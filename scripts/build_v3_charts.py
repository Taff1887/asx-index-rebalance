"""Charts for v3: horizon (superfund test), extended event study, and the
'hold ASX 200 instead of cash' total-return comparison vs ASX 50/100/200 TR.

No cumulative-return-over-cash line chart. The one time-series here is the
extended EVENT-TIME study (t-5..t+45), which is an event-window average, not
a calendar cumulative curve.
"""

from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from asxrebalance.paths import (  # noqa: E402
    OUTPUTS_DIR, PROCESSED_BENCHMARK_DIR, PROCESSED_RECONCILED_DIR,
    PROCESSED_LABELS_DIR, RAW_YAHOO_DIR, REPO_ROOT,
)

DOCS = REPO_ROOT / "docs" / "figures"
EXIT_ORDER = ["eff", "eff5", "eff10", "eff20", "eff30", "eff40"]
EXIT_DAYS = {"eff": 10, "eff5": 15, "eff10": 20, "eff20": 30, "eff30": 40, "eff40": 50}


def _save(fig, name):
    fig.tight_layout(); fig.savefig(DOCS / name, dpi=140); plt.close(fig)
    print(f"  -> docs/figures/{name}")


def horizon_chart():
    h = pd.read_csv(OUTPUTS_DIR / "v3_horizon.csv")
    a = h[h.tier == "ASX200"]
    fig, ax = plt.subplots(figsize=(11, 6.5))
    xs = [EXIT_DAYS[e] for e in EXIT_ORDER]
    for side, c in (("long", "#2e7d32"), ("short", "#c62828")):
        s = a[a.side == side].set_index("exit").reindex(EXIT_ORDER)
        ax.plot(xs, s["mean_pct"], marker="o", color=c, lw=2, label=f"{side} (mean/trade)")
        for x, y, n in zip(xs, s["mean_pct"], s["n"]):
            if pd.notna(y):
                ax.annotate(f"{y:+.1f}%", (x, y), textcoords="offset points",
                            xytext=(0, 8 if side == "short" else -14), ha="center", fontsize=8, color=c)
    ax.axhline(0, color="black", lw=0.6)
    ax.axvline(10, color="grey", ls=":", label="effective date")
    ax.set_xlabel("Trading days held (announcement+1 entry → exit)")
    ax.set_ylabel("ASX 200 mean return per trade (%)")
    ax.set_title("Does holding longer help?  ASX 200 per-trade return vs holding period\n"
                 "(tests the 'funds keep buying additions' hypothesis)")
    ax.legend(); ax.grid(True, alpha=0.3)
    _save(fig, "v3_horizon.png")


def _load_price(tk):
    for d in (PROCESSED_RECONCILED_DIR / "prices", RAW_YAHOO_DIR):
        p = d / f"{tk}.csv"
        if p.exists():
            return pd.read_csv(p, parse_dates=["date"]).sort_values("date").reset_index(drop=True)
    return None


def event_study_long():
    """Extended event study t-5..t+45 for ASX200 adds/removes (valid trades)."""
    labels = pd.read_csv(PROCESSED_LABELS_DIR / "rebalance_labels.csv",
                          parse_dates=["announcement_date", "effective_date"])
    labels = labels[labels["index"] == "ASX200"]
    bench = pd.read_csv(PROCESSED_BENCHMARK_DIR / "asx200_benchmark.csv", parse_dates=["date"]).sort_values("date")
    bench = bench.set_index("date")["adjusted_close"].pct_change()
    valid = set(pd.read_csv(OUTPUTS_DIR / "v2_trades.csv")["ticker"])  # only validated tickers

    paths = {"Addition": [], "Removal": []}
    for _, e in labels.iterrows():
        if e["ticker"] not in valid:
            continue
        df = _load_price(e["ticker"])
        if df is None:
            continue
        df = df.set_index("date")
        ann = pd.Timestamp(e["announcement_date"])
        if ann not in df.index:
            aft = df.index[df.index >= ann]
            if len(aft) == 0:
                continue
            ann = aft[0]
        pos = df.index.get_loc(ann)
        lo, hi = max(0, pos - 5), min(len(df), pos + 46)
        win = df.iloc[lo:hi]
        ar = win["adjusted_close"].pct_change() - bench.reindex(win.index)
        ar.index = range(lo - pos, hi - pos)
        paths[e["action"]].append(ar)

    fig, ax = plt.subplots(figsize=(12, 6.5))
    for action, c in (("Addition", "#2e7d32"), ("Removal", "#c62828")):
        if not paths[action]:
            continue
        m = pd.concat(paths[action], axis=1).mean(axis=1).cumsum() * 100
        ax.plot(m.index, m.values, color=c, lw=2, label=f"{action}s (n={len(paths[action])})")
    ax.axvline(0, color="black", ls="--", lw=1.2, label="Announcement (t=0)")
    ax.axvline(1, color="#1f4e79", ls=":", lw=1.5, label="Strategy entry (t+1)")
    ax.axvline(10, color="grey", ls=":", lw=1.2, label="Effective (~t+10)")
    ax.axhline(0, color="black", lw=0.5)
    ax.set_xlabel("Trading days from announcement")
    ax.set_ylabel("Mean cumulative abnormal return vs ASX 200 (%)")
    ax.set_title("Extended ASX 200 event study (t-5 → t+45)\n"
                 "Do additions recover as super funds keep buying? (the long-side question)")
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3)
    _save(fig, "v3_event_study_long.png")


def overlay_vs_benchmark_chart():
    ov = pd.read_csv(OUTPUTS_DIR / "v3_overlay.csv")

    def tr_total(name):
        f = PROCESSED_BENCHMARK_DIR / f"{name.lower()}_tr.csv"
        df = pd.read_csv(f, parse_dates=["date"]).sort_values("date")
        d = pd.read_csv(OUTPUTS_DIR / "v3_daily.csv", parse_dates=["date"])
        df = df[(df.date >= d.date.min()) & (df.date <= d.date.max())]
        return (df["adjusted_close"].iloc[-1] / df["adjusted_close"].iloc[0] - 1) * 100

    def get(sub, col):
        return ov[ov.series.str.contains(sub)][col].iloc[0]

    # benchmarks (single bar), then strategy variants (gross + ex-outlier paired)
    bench = [("ASX 50 TR", tr_total("ASX50"), "#9c27b0"),
             ("ASX 100 TR", tr_total("ASX100"), "#ff9800"),
             ("ASX 200 TR\n(buy & hold)", tr_total("ASX200"), "#616161")]
    variants = [("Switch LONG\nadditions", "LONG additions", "#2e7d32"),
                ("Switch SHORT\nremovals", "SHORT removals", "#c62828"),
                ("Switch\nLONG/SHORT", "LONG/SHORT", "#1f4e79")]

    fig, ax = plt.subplots(figsize=(14, 7.5))
    x = 0; xticks = []; xlabels = []
    for label, val, c in bench:
        ax.bar(x, val, 0.6, color=c)
        ax.text(x, val + 8, f"{val:+.0f}%", ha="center", fontweight="bold")
        xticks.append(x); xlabels.append(label); x += 1
    x += 0.5
    for label, key, c in variants:
        g = get(key, "all_trades_pct"); r = get(key, "robust_ex_outlier_pct")
        ax.bar(x - 0.21, g, 0.4, color=c, alpha=0.45, label="gross (incl. 2013 outlier)" if x < 5 else None)
        ax.bar(x + 0.21, r, 0.4, color=c, label="robust (ex-outlier)" if x < 5 else None)
        ax.text(x - 0.21, g + 8, f"{g:+.0f}%", ha="center", fontsize=8)
        ax.text(x + 0.21, r + 8, f"{r:+.0f}%", ha="center", fontsize=8, fontweight="bold")
        xticks.append(x); xlabels.append(label); x += 1.1
    ax.axhline(tr_total("ASX200"), color="#616161", ls="--", lw=1, label="ASX 200 buy & hold")
    ax.set_xticks(xticks); ax.set_xticklabels(xlabels)
    ax.set_ylabel("Total return over 2012-09 → 2026-01 (%)")
    ax.set_title("'Hold ASX 200 instead of cash' switching strategy vs total-return benchmarks\n"
                 "GROSS of costs & survivorship-biased; robust bar removes the one 2013 trade (PRU) "
                 "that drives ~half the gross short")
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(True, axis="y", alpha=0.3); ax.axhline(0, color="black", lw=0.5)
    _save(fig, "v3_switch_vs_benchmark.png")


def main():
    horizon_chart()
    event_study_long()
    overlay_vs_benchmark_chart()


if __name__ == "__main__":
    main()
