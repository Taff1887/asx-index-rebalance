"""Pure ALPHA event study — is there a real, risk-adjusted index-rebalance signal?

No trading strategy. We measure market-ADJUSTED abnormal returns (AR) around the
announcement and test whether additions/removals earn statistically significant
abnormal returns versus the ASX 200.

    AR_it   = R_it - R_market,t            (market-adjusted model, beta=1)
    CAR     = sum of AR over an event window
    market  = S&P/ASX 200 price index (^AXJO) daily return

We split every event four ways because the mechanism differs:

    SCHEDULED  (quarterly rank review)  -> the PURE index-demand signal
    OFF-CYCLE  (M&A / demerger driven)  -> contaminated by takeover premium

For each (action x cohort x window) we report n, mean CAR, cross-sectional
t-stat, median, % in expected direction, Wilcoxon p, sign p, and a PLACEBO p
(random announcement dates in the same names) — the luck test.

Outputs:
    outputs/alpha_events.csv     per-event ARs / CARs (for independent re-check)
    outputs/alpha_car.csv        per group x window cross-sectional statistics
    docs/figures/alpha_car_path.png    mean CAR path t-5..t+30 by group
    docs/figures/alpha_car_windows.png CAR by window x group, significance-flagged
"""

from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import stats  # noqa: E402

from asxrebalance.paths import (  # noqa: E402
    OUTPUTS_DIR, PROCESSED_BENCHMARK_DIR, PROCESSED_RECONCILED_DIR, RAW_YAHOO_DIR, REPO_ROOT,
)

DOCS = REPO_ROOT / "docs" / "figures"
RNG = np.random.default_rng(20260609)
N_PLACEBO = 3000
PRE, POST = 5, 30          # event-time path window (trading bars around announcement)
WINDOWS = {                # CAR windows, in bar offsets from announcement t0
    "ann[-1,+1]": (-1, 1),
    "ann[0,+1]": (0, 1),
    "post[+2,+5]": (2, 5),
    "post[+2,+10]": (2, 10),
    "run[0,+10]": (0, 10),
    "run[0,+20]": (0, 20),
}


def _load_price(tk):
    for d in (PROCESSED_RECONCILED_DIR / "prices", RAW_YAHOO_DIR):
        p = d / f"{tk}.csv"
        if p.exists():
            df = pd.read_csv(p, parse_dates=["date"]).sort_values("date").reset_index(drop=True)
            if not df.empty:
                return df
    return None


def main() -> None:
    master = pd.read_csv(OUTPUTS_DIR / "asx200_events_master.csv",
                         parse_dates=["announcement_date", "effective_date"])
    mkt = pd.read_csv(PROCESSED_BENCHMARK_DIR / "asx200_benchmark.csv", parse_dates=["date"])
    mkt = mkt.sort_values("date").set_index("date")["adjusted_close"].pct_change()

    lo_off, hi_off = -PRE, POST
    rows, path_acc = [], {}   # path_acc[(action,cohort)] -> list of AR series indexed by offset

    for _, e in master.iterrows():
        tk = str(e["ticker"])
        df = _load_price(tk)
        if df is None:
            continue
        df = df.set_index("date")
        ann = pd.Timestamp(e["announcement_date"])
        if pd.isna(ann):
            continue
        idx = df.index
        after = idx[idx >= ann]
        if len(after) == 0:
            continue
        t0 = after[0]
        # DATA-HYGIENE GATE: reject events whose price file has no bar near the
        # announcement (searchsorted would otherwise jump the window years
        # forward and measure an unrelated date). Found by adversarial review.
        if (t0 - ann).days > 7:
            continue
        pos = idx.get_loc(t0)
        if isinstance(pos, slice):
            pos = pos.start
        lo, hi = pos + lo_off, pos + hi_off + 1
        if lo < 1 or hi > len(df):
            # allow partial but need at least announcement..+1
            lo, hi = max(1, lo), min(len(df), hi)
        win = df.iloc[lo:hi]
        stock_ret = win["adjusted_close"].pct_change()
        ar = (stock_ret - mkt.reindex(win.index)) * 100.0
        ar.index = range(lo - pos, hi - pos)   # offset relative to t0
        action = str(e["action"]).title()
        cohort = "off_cycle" if bool(e["off_cycle"]) else "scheduled"

        rec = {"ticker": tk, "action": action, "cohort": cohort,
               "announcement_date": ann.date()}
        for name, (a, b) in WINDOWS.items():
            seg = ar[(ar.index >= a) & (ar.index <= b)].dropna()
            width = b - a + 1
            # require >=60% of window days actually present, else NaN (don't let
            # pandas sum an all-NaN window to a fake 0.0 that pollutes the stats).
            rec[name] = float(seg.sum()) if len(seg) >= max(1, int(0.6 * width)) else np.nan
        rows.append(rec)
        path_acc.setdefault((action, cohort), []).append(ar)

    events = pd.DataFrame(rows)
    events.to_csv(OUTPUTS_DIR / "alpha_events.csv", index=False)

    # ---- cross-sectional stats per group x window ----
    def placebo_p(sub_tickers, a, b, observed, direction):
        """Random-date null for CAR over window [a,b]; one-sided in `direction`."""
        recs = []
        for tk in sub_tickers:
            df = _load_price(tk)
            if df is None:
                continue
            sr = df.set_index("date")["adjusted_close"].pct_change()
            ar_full = (sr - mkt.reindex(sr.index)) * 100.0
            arr = ar_full.to_numpy()
            recs.append(arr)
        if len(recs) < 5:
            return np.nan
        width = b - a + 1
        null = np.empty(N_PLACEBO)
        for k in range(N_PLACEBO):
            vals = []
            for arr in recs:
                n = len(arr)
                if n < width + 6:
                    continue
                s = RNG.integers(3, n - width - 3)
                vals.append(np.nansum(arr[s:s + width]))
            null[k] = np.mean(vals) if vals else np.nan
        null = null[np.isfinite(null)]
        if direction > 0:
            return (1 + np.sum(null >= observed)) / (len(null) + 1)
        return (1 + np.sum(null <= observed)) / (len(null) + 1)

    out = []
    for action in ("Addition", "Removal"):
        for cohort in ("scheduled", "off_cycle"):
            sub = events[(events.action == action) & (events.cohort == cohort)]
            if len(sub) < 8:
                continue
            tickers = sub["ticker"].tolist()
            direction = 1 if action == "Addition" else -1  # expected sign of abnormal return
            for name, (a, b) in WINDOWS.items():
                x = sub[name].dropna().to_numpy(float)
                if len(x) < 8:
                    continue
                t, p_t = stats.ttest_1samp(x, 0.0)
                try:
                    _, p_w = stats.wilcoxon(x)
                except ValueError:
                    p_w = np.nan
                in_dir = np.mean((x * direction) > 0) * 100
                p_sign = stats.binomtest(int((x > 0).sum()), len(x), 0.5).pvalue
                pp = placebo_p(tickers, a, b, float(x.mean()), direction) if name in (
                    "ann[0,+1]", "run[0,+10]", "post[+2,+10]") else np.nan
                out.append({
                    "action": action, "cohort": cohort, "window": name, "n": len(x),
                    "mean_car_pct": round(float(x.mean()), 3),
                    "median_car_pct": round(float(np.median(x)), 3),
                    "t_stat": round(float(t), 2), "p_ttest": round(float(p_t), 4),
                    "p_wilcoxon": round(float(p_w), 4) if pd.notna(p_w) else np.nan,
                    "pct_expected_dir": round(in_dir, 1), "p_sign": round(float(p_sign), 4),
                    "p_placebo": round(float(pp), 4) if pd.notna(pp) else np.nan,
                })
    car = pd.DataFrame(out)
    car.to_csv(OUTPUTS_DIR / "alpha_car.csv", index=False)

    # ---- chart 1: mean CAR path ----
    fig, ax = plt.subplots(figsize=(12, 7))
    styles = {("Addition", "scheduled"): ("#2e7d32", "-"),
              ("Removal", "scheduled"): ("#c62828", "-"),
              ("Addition", "off_cycle"): ("#66bb6a", "--"),
              ("Removal", "off_cycle"): ("#ef9a9a", "--")}
    for key, paths in path_acc.items():
        if len(paths) < 8:
            continue
        m = pd.concat(paths, axis=1).reindex(range(lo_off, hi_off + 1)).mean(axis=1)
        car_path = m.fillna(0).cumsum()
        c, ls = styles.get(key, ("#555", ":"))
        ax.plot(car_path.index, car_path.values, color=c, ls=ls, lw=2,
                label=f"{key[0]} · {key[1]} (n={len(paths)})")
    ax.axvline(0, color="black", ls="--", lw=1, label="Announcement (t0)")
    ax.axvline(1, color="#1f4e79", ls=":", lw=1, label="t+1 (info confirmed)")
    ax.axhline(0, color="black", lw=0.6)
    ax.set_xlabel("Trading days from announcement")
    ax.set_ylabel("Mean cumulative ABNORMAL return vs ASX 200 (%)")
    ax.set_title("Index-rebalance ALPHA — market-adjusted CAR around announcement\n"
                 "Scheduled = pure index demand · Off-cycle = M&A-driven (takeover premium contaminates removals)")
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3)
    fig.tight_layout(); fig.savefig(DOCS / "alpha_car_path.png", dpi=140); plt.close(fig)
    print("  -> docs/figures/alpha_car_path.png")

    # ---- chart 2: CAR by window for SCHEDULED (the pure signal) ----
    sched = car[car.cohort == "scheduled"].copy()
    wlist = list(WINDOWS.keys())
    fig, ax = plt.subplots(figsize=(13, 6.8))
    x = np.arange(len(wlist)); w = 0.38
    for i, (action, color) in enumerate([("Addition", "#2e7d32"), ("Removal", "#c62828")]):
        d = sched[sched.action == action].set_index("window").reindex(wlist)
        vals = d["mean_car_pct"].values
        bars = ax.bar(x + (i - 0.5) * w, np.nan_to_num(vals), w, color=color, label=action)
        for xi, (val, row) in enumerate(zip(vals, d.itertuples())):
            if pd.isna(val):
                continue
            sig = (pd.notna(row.p_wilcoxon) and row.p_wilcoxon < 0.05)
            star = "*" if sig else ""
            ax.annotate(f"{val:+.1f}{star}", (x[xi] + (i - 0.5) * w, val),
                        ha="center", va="bottom" if val >= 0 else "top", fontsize=7.5)
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xticks(x); ax.set_xticklabels(wlist, fontsize=8)
    ax.set_ylabel("Mean cumulative abnormal return (%)")
    ax.set_title("SCHEDULED index changes — abnormal return by event window\n"
                 "* = Wilcoxon p<0.05. This is the pure index-demand alpha (no M&A contamination).")
    ax.legend(); ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout(); fig.savefig(DOCS / "alpha_car_windows.png", dpi=140); plt.close(fig)
    print("  -> docs/figures/alpha_car_windows.png")

    # ---- console ----
    pd.set_option("display.width", 220)
    print(f"\nAnalysable events: {len(events)}  "
          f"(scheduled {(events.cohort=='scheduled').sum()}, off-cycle {(events.cohort=='off_cycle').sum()})")
    print("\nALPHA — market-adjusted CAR by action x cohort x window:")
    print(car.to_string(index=False))


if __name__ == "__main__":
    main()
