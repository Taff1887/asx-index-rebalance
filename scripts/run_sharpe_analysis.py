"""Sharpe ratios: the standalone rebalance strategy (cash between trades) vs simply
buying-and-holding the ASX 200, vs holding the risk-free rate (2.5%).

NO 'switch into the index' overlay. Between trades the book sits in CASH at the
risk-free rate. Each trade is HELD TO ITS EXIT (no daily rebalancing) — we spread
each trade's realised hold-return evenly across its holding days, so the daily
series reproduces the real per-trade returns exactly and does NOT suffer the
volatility-drag artifact of a daily-rebalanced equal-weight book. Concurrent
trades are equal-weighted; idle days earn the risk-free rate.

We report the SHORT-removal, LONG-addition and LONG/SHORT books (ASX 200, exit
eff+5), each WITH and WITHOUT the one 2013 PRU outlier, plus buy & hold and cash.

Outputs: outputs/sharpe_table.csv · docs/figures/sharpe_ratios.png
"""

from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from asxrebalance.paths import OUTPUTS_DIR, PROCESSED_BENCHMARK_DIR, REPO_ROOT  # noqa: E402

DOCS = REPO_ROOT / "docs" / "figures"
RF_ANNUAL = 0.025
RF_DAILY = RF_ANNUAL / 252.0
TD = 252


def _daily_series(trades: pd.DataFrame, idx: pd.DatetimeIndex, exclude=frozenset()):
    """Hold-to-exit, equal-weight concurrent, cash (rf) when idle.

    Each trade's realised return is spread evenly across its holding days as a
    constant daily rate; the day's book return is the mean across active trades.
    """
    acc = pd.Series(0.0, index=idx)   # sum of active daily rates
    cnt = pd.Series(0.0, index=idx)   # number of active trades
    for _, t in trades.iterrows():
        if t["ticker"] in exclude:
            continue
        d0, d1 = pd.Timestamp(t["entry_date"]), pd.Timestamp(t["exit_eff5_date"])
        days = idx[(idx > d0) & (idx <= d1)]
        if len(days) == 0:
            continue
        rate = (1 + t["ret"] / 100.0) ** (1.0 / len(days)) - 1.0
        acc.loc[days] += rate
        cnt.loc[days] += 1
    book = (acc / cnt).where(cnt > 0, RF_DAILY)   # equal-weight active, else cash
    return book


def stats(r: pd.Series):
    r = r.dropna()
    n = len(r)
    cagr = (1 + r).prod() ** (TD / n) - 1
    vol = r.std() * np.sqrt(TD)
    sharpe = (r.mean() - RF_DAILY) / r.std() * np.sqrt(TD) if r.std() > 0 else np.nan
    return cagr * 100, vol * 100, sharpe


def main() -> None:
    t = pd.read_csv(OUTPUTS_DIR / "v2_trades.csv",
                    parse_dates=["entry_date", "exit_eff5_date"])
    t = t[t["tier"] == "ASX200"].copy()
    t["ret"] = t["trade_eff5_pct"]          # GROSS per-trade % (eff+5), already signed
    t = t[t["ret"].notna() & t["exit_eff5_date"].notna()]

    bench = pd.read_csv(PROCESSED_BENCHMARK_DIR / "asx200_tr.csv", parse_dates=["date"]).sort_values("date")
    idx = pd.DatetimeIndex(bench["date"])
    tr_ret = bench.set_index("date")["adjusted_close"].pct_change()

    books = {
        "short": t[t.side == "short"],
        "long": t[t.side == "long"],
        "ls": t,   # both; longs +, shorts already +signed when stock falls
    }
    labels = {"short": "SHORT removals\n+ cash (2.5%)", "long": "LONG additions\n+ cash (2.5%)",
              "ls": "LONG+SHORT\n+ cash (2.5%)"}

    rows = []
    for key, tr in books.items():
        ca, va, sa = stats(_daily_series(tr, idx))
        cr, vr, sr = stats(_daily_series(tr, idx, exclude={"PRU"}))
        rows.append({"variant": labels[key].replace("\n", " "), "col": key,
                     "cagr_all_pct": round(ca, 2), "vol_all_pct": round(va, 2), "sharpe_all": round(sa, 3),
                     "cagr_robust_pct": round(cr, 2), "vol_robust_pct": round(vr, 2), "sharpe_robust": round(sr, 3)})
    cab, vab, sab = stats(tr_ret)
    rows.insert(0, {"variant": "ASX 200 buy & hold", "col": "bh",
                    "cagr_all_pct": round(cab, 2), "vol_all_pct": round(vab, 2), "sharpe_all": round(sab, 3),
                    "cagr_robust_pct": round(cab, 2), "vol_robust_pct": round(vab, 2), "sharpe_robust": round(sab, 3)})
    rows.append({"variant": "Hold risk-free (2.5%/yr)", "col": "rf",
                 "cagr_all_pct": 2.5, "vol_all_pct": 0.0, "sharpe_all": 0.0,
                 "cagr_robust_pct": 2.5, "vol_robust_pct": 0.0, "sharpe_robust": 0.0})
    tab = pd.DataFrame(rows)
    tab.to_csv(OUTPUTS_DIR / "sharpe_table.csv", index=False)

    # ---- chart ----
    plot = tab[~tab.col.isin(["rf"])]
    x = np.arange(len(plot)); w = 0.38
    fig, ax = plt.subplots(figsize=(12.5, 7))
    colors = ["#616161" if c == "bh" else "#1565c0" for c in plot["col"]]
    b1 = ax.bar(x - w / 2, plot["sharpe_all"], w, color="#90caf9", label="with 2013 PRU outlier")
    b2 = ax.bar(x + w / 2, plot["sharpe_robust"], w, color="#1565c0", label="without PRU")
    for bars, key in ((b1, "sharpe_all"), (b2, "sharpe_robust")):
        for r, v in zip(bars, plot[key]):
            ax.annotate(f"{v:.2f}", (r.get_x() + r.get_width() / 2, v), ha="center",
                        va="bottom" if v >= 0 else "top", fontsize=8.5, fontweight="bold")
    bh = tab[tab.col == "bh"]["sharpe_all"].iloc[0]
    ax.axhline(bh, color="#616161", ls="--", lw=1.2, label=f"ASX 200 buy & hold = {bh:.2f}")
    ax.axhline(0, color="#c62828", ls=":", lw=1.4, label="risk-free (2.5%) = 0")
    ax.set_xticks(x); ax.set_xticklabels(plot["variant"].str.replace(" ", "\n", 1), fontsize=8.5)
    ax.set_ylabel("Annualised Sharpe ratio (rf = 2.5%/yr)")
    ax.set_title("Sharpe ratios — standalone rebalance strategy (cash between trades) vs buy & hold vs cash\n"
                 "Trades held to exit (no daily-rebalance drag); ASX 200, eff+5, gross of costs")
    ax.legend(fontsize=9); ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout(); fig.savefig(DOCS / "sharpe_ratios.png", dpi=140); plt.close(fig)
    print("  -> docs/figures/sharpe_ratios.png")

    pd.set_option("display.width", 200)
    print("\nSHARPE / RETURN / VOL (annualised; rf = 2.5%):")
    show = tab.copy(); show["variant"] = show["variant"].str.replace("→", "->", regex=False)
    print(show[["variant", "cagr_all_pct", "vol_all_pct", "sharpe_all", "sharpe_robust"]].to_string(index=False))


if __name__ == "__main__":
    main()
