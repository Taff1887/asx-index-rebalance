"""PREDICT ASX 200 inclusions/removals BEFORE the announcement, then pre-position.

The alpha (§2) lands *at* the announcement and is gone by the time you can trade
on the public news. The only way to capture it is to **predict** the change and
be positioned **before** S&P announces. S&P's rule is mechanical — float-adjusted
**market-cap rank** — so we predict from size:

  * prediction date  P = 10 trading days BEFORE each quarterly announcement D
  * rank the boundary universe (ASX 200/300 names) by market cap at P
  * predicted ADDITIONS = the K largest names NOT currently in the ASX 200
  * predicted REMOVALS  = the K smallest names currently IN the ASX 200
  (membership tracked from the scheduled-rebalance event history)

We then (a) score the prediction (precision / recall vs what S&P actually did)
and (b) back-test pre-positioning: go long predicted adds / short predicted
removals at the P close, and exit at the announcement-day close, the next day
(through the pop), and the effective date. Real prices + real FMP market caps.

Outputs: outputs/predictor_events.csv · outputs/predictor_summary.csv
         docs/figures/predictor.png
"""

from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from asxrebalance.paths import (  # noqa: E402
    OUTPUTS_DIR, PROCESSED_BENCHMARK_DIR, PROCESSED_LABELS_DIR,
    PROCESSED_RECONCILED_DIR, RAW_FMP_DIR, RAW_YAHOO_DIR, REPO_ROOT,
)

DOCS = REPO_ROOT / "docs" / "figures"
MKTCAP_DIR = RAW_FMP_DIR.parent / "fmp_mktcap"
K = 5            # predictions per side per rebalance
LOOKBACK = 10    # trading days before the announcement to predict & enter


def _price(tk):
    for d in (PROCESSED_RECONCILED_DIR / "prices", RAW_YAHOO_DIR):
        p = d / f"{tk}.csv"
        if p.exists():
            df = pd.read_csv(p, parse_dates=["date"]).sort_values("date")
            if not df.empty:
                return df.set_index("date")
    return None


def main() -> None:
    labels = pd.read_csv(PROCESSED_LABELS_DIR / "rebalance_labels.csv",
                         parse_dates=["announcement_date"])
    a200 = labels[labels["index"] == "ASX200"].copy()
    uni = sorted(labels[labels["index"].isin(["ASX200", "ASX300"])]["ticker"].astype(str).unique())

    # market-cap series for the universe
    mc = {}
    for t in uni:
        p = MKTCAP_DIR / f"{t}.csv"
        if p.exists():
            s = pd.read_csv(p, parse_dates=["date"]).sort_values("date").set_index("date")["market_cap"]
            if len(s) > 50:
                mc[t] = s
    print(f"universe with market caps: {len(mc)} / {len(uni)}")

    # ASX 200 membership from scheduled events: last event before date wins
    evs = {t: a200[a200.ticker == t].sort_values("announcement_date") for t in a200.ticker.unique()}

    def is_member(t, date):
        e = evs.get(t)
        if e is None:
            return False
        past = e[e.announcement_date < date]
        if past.empty:
            return False
        return past.iloc[-1]["action"] == "Addition"

    cal = pd.DatetimeIndex(pd.read_csv(PROCESSED_BENCHMARK_DIR / "asx200_benchmark.csv",
                                       parse_dates=["date"])["date"]).sort_values()

    def mc_at(t, date):
        s = mc.get(t)
        if s is None:
            return np.nan
        s = s[s.index <= date]
        return float(s.iloc[-1]) if len(s) else np.nan

    def ret(tk, d_entry, d_exit, sgn):
        df = _price(tk)
        if df is None:
            return np.nan
        ein = df[df.index >= d_entry]
        eout = df[df.index >= d_exit]
        if ein.empty or eout.empty:
            return np.nan
        p0 = float(ein.iloc[0]["adjusted_close"]); p1 = float(eout.iloc[0]["adjusted_close"])
        if p0 <= 0 or p1 <= 0:
            return np.nan
        return sgn * (p1 / p0 - 1) * 100

    rows, pools = [], []
    for D in sorted(a200.announcement_date.unique()):
        D = pd.Timestamp(D)
        pos = cal.searchsorted(D)
        if pos - LOOKBACK < 0 or pos >= len(cal):
            continue
        P = cal[pos - LOOKBACK]                 # prediction / entry date
        Dpost = cal[min(pos + 1, len(cal) - 1)]  # day after announcement (capture the pop)

        caps = {t: mc_at(t, P) for t in mc}
        caps = {t: v for t, v in caps.items() if np.isfinite(v)}
        if len(caps) < 30:
            continue
        members = {t: v for t, v in caps.items() if is_member(t, P)}
        nonmem = {t: v for t, v in caps.items() if not is_member(t, P)}
        if len(members) < 5:
            continue
        # A real addition sits near the index boundary, not among the mega-caps.
        # Our membership map only tracks boundary names, so the huge stable members
        # (no Addition event in our data) leak into `nonmem`. Exclude any non-member
        # bigger than the 75th percentile of tracked members — it is surely already
        # in the index. Predicted additions = the largest of the REMAINING (boundary)
        # non-members; predicted removals = the smallest current members.
        ceil = float(np.percentile(list(members.values()), 75))
        add_pool = {t: v for t, v in nonmem.items() if v < ceil}
        pred_add = [t for t, _ in sorted(add_pool.items(), key=lambda x: -x[1])[:K]]
        pred_rem = [t for t, _ in sorted(members.items(), key=lambda x: x[1])[:K]]

        actual = a200[a200.announcement_date == D]
        act_add = set(actual[actual.action == "Addition"].ticker)
        act_rem = set(actual[actual.action == "Removal"].ticker)
        pools.append({"side": "add", "pool": len(add_pool), "actual": len(act_add)})
        pools.append({"side": "rem", "pool": len(members), "actual": len(act_rem)})

        for side, preds, actual_set, sgn in (("add", pred_add, act_add, 1.0),
                                             ("rem", pred_rem, act_rem, -1.0)):
            for tk in preds:
                hit = tk in actual_set
                rows.append({
                    "announcement_date": D.date(), "side": side, "ticker": tk, "hit": hit,
                    "entry_date": P.date(),
                    "ret_to_announce_pct": ret(tk, P, D, sgn),       # P -> announcement close (pre-news)
                    "ret_through_pop_pct": ret(tk, P, Dpost, sgn),   # P -> day after (captures the pop)
                })
    ev = pd.DataFrame(rows)
    ev.to_csv(OUTPUTS_DIR / "predictor_events.csv", index=False)

    # ---- scoring ----
    n_reb = ev.announcement_date.nunique()
    act_add_total = a200[(a200.action == "Addition")
                         & (a200.announcement_date.isin(pd.to_datetime(ev.announcement_date.unique())))].shape[0]
    act_rem_total = a200[(a200.action == "Removal")
                         & (a200.announcement_date.isin(pd.to_datetime(ev.announcement_date.unique())))].shape[0]

    pf = pd.DataFrame(pools)

    def summ(side, actual_total):
        d = ev[ev.side == side]
        n = len(d); hits = int(d.hit.sum())
        # random baseline precision = mean over rebalances of actual/pool (chance
        # of a random pick from the pool being a real change).
        ps = pf[(pf.side == side) & (pf.pool > 0)]
        rand = round(100 * (ps["actual"] / ps["pool"]).mean(), 1) if len(ps) else 0.0
        return {
            "side": "Additions" if side == "add" else "Removals",
            "predictions": n, "hits": hits,
            "precision_pct": round(100 * hits / n, 1) if n else 0,
            "random_precision_pct": rand,
            "recall_pct": round(100 * hits / actual_total, 1) if actual_total else 0,
            "actual_changes": actual_total,
            "ret_to_announce_pct": round(d["ret_to_announce_pct"].mean(), 3),
            "ret_through_pop_pct": round(d["ret_through_pop_pct"].mean(), 3),
        }
    s = pd.DataFrame([summ("add", act_add_total), summ("rem", act_rem_total)])
    s.to_csv(OUTPUTS_DIR / "predictor_summary.csv", index=False)

    # ---- chart ----
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6.2))
    x = np.arange(2); lbl = ["Additions", "Removals"]
    ax1.bar(x, s["precision_pct"], 0.5, color=["#2e7d32", "#c62828"])
    for xi, v, h, n in zip(x, s["precision_pct"], s["hits"], s["predictions"]):
        ax1.annotate(f"{v:.0f}%\n({h}/{n})", (xi, v), ha="center", va="bottom", fontweight="bold")
    ax1.set_xticks(x); ax1.set_xticklabels(lbl); ax1.set_ylabel("Hit rate (precision, %)")
    ax1.set_title(f"Can we PREDICT the change?  top/bottom {K} by market cap\n"
                  f"{LOOKBACK} trading days before the announcement")
    ax1.grid(True, axis="y", alpha=0.3)

    w = 0.38
    ax2.bar(x - w / 2, s["ret_to_announce_pct"], w, color="#90caf9", label="all predicted (P→announce)")
    ax2.bar(x + w / 2, s["ret_through_pop_pct"], w, color="#1565c0", label="all predicted (P→day after)")
    for xi, a, b in zip(x, s["ret_to_announce_pct"], s["ret_through_pop_pct"]):
        ax2.annotate(f"{a:+.1f}%", (xi - w / 2, a), ha="center", va="bottom", fontsize=8)
        ax2.annotate(f"{b:+.1f}%", (xi + w / 2, b), ha="center", va="bottom", fontsize=8)
    ax2.axhline(0, color="black", lw=0.7)
    ax2.set_xticks(x); ax2.set_xticklabels(lbl)
    ax2.set_ylabel("Mean return of predicted trades (%)")
    ax2.set_title("Pre-positioning return (long adds / short removals)\nenter 10 days early, exit at / after the announcement")
    ax2.legend(fontsize=8); ax2.grid(True, axis="y", alpha=0.3)
    fig.tight_layout(); fig.savefig(DOCS / "predictor.png", dpi=140); plt.close(fig)
    print("  -> docs/figures/predictor.png")

    pd.set_option("display.width", 200)
    print("\nPREDICTOR — score & pre-positioning return:")
    print(s.to_string(index=False))
    print(f"\n(rebalances scored: {ev.announcement_date.nunique()}, K={K} per side, "
          f"entry {LOOKBACK} trading days before announcement)")


if __name__ == "__main__":
    main()
