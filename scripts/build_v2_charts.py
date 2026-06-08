"""Charts for the validated v2 strategy. NO cumulative-return-over-calendar line.

Produces (all in docs/figures/):
  v2_mean_per_trade.png  - grouped bars: mean per-trade % by tier x side (exit=eff)
  v2_win_rate.png        - grouped bars: win rate % by tier x side
  v2_exit_window.png     - ASX200 long/short mean per-trade by exit window
  v2_event_study.png     - average price path in EVENT TIME (t-5..t+10) for
                           ASX200 adds/removes, marking the announcement-day pop
                           that the ann+1 entry skips
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from asxrebalance.paths import (  # noqa: E402
    OUTPUTS_DIR, PROCESSED_BENCHMARK_DIR, PROCESSED_RECONCILED_DIR,
    RAW_YAHOO_DIR, REPO_ROOT,
)

DOCS = REPO_ROOT / "docs" / "figures"
DOCS.mkdir(parents=True, exist_ok=True)
TIERS = ["ASX20", "ASX50", "ASX100", "ASX200"]
PAL = {"Long-only": "#2e7d32", "Short-only": "#c62828", "Long & short": "#1f4e79"}


def _save(fig, name):
    fig.tight_layout()
    fig.savefig(DOCS / name, dpi=140)
    plt.close(fig)
    print(f"  -> docs/figures/{name}")


def mean_per_trade_chart(summary):
    sub = summary[summary["exit_window"] == "eff"]
    fig, ax = plt.subplots(figsize=(12, 7))
    x = np.arange(len(TIERS)); w = 0.26
    for i, v in enumerate(("Long-only", "Short-only", "Long & short")):
        vals, ns = [], []
        for tier in TIERS:
            rr = sub[(sub.tier == tier) & (sub.variant == v)]
            vals.append(rr.mean_pct.iloc[0] if not rr.empty else 0)
            ns.append(int(rr.n_trades.iloc[0]) if not rr.empty else 0)
        bars = ax.bar(x + (i - 1) * w, vals, w, color=PAL[v], label=v)
        for b, val, n in zip(bars, vals, ns):
            ax.text(b.get_x() + b.get_width() / 2, val + (0.08 if val >= 0 else -0.08),
                    f"{val:+.2f}%\nn={n}", ha="center",
                    va="bottom" if val >= 0 else "top", fontsize=8)
    ax.set_xticks(x); ax.set_xticklabels(TIERS)
    ax.axhline(0, color="black", lw=0.6)
    ax.set_ylabel("Mean return PER TRADE (%)")
    ax.set_title("Average per-trade return by index tier and side  (entry = announcement+1 close, exit = effective close)")
    ax.legend(); ax.grid(True, axis="y", alpha=0.3)
    _save(fig, "v2_mean_per_trade.png")


def win_rate_chart(summary):
    sub = summary[summary["exit_window"] == "eff"]
    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(TIERS)); w = 0.26
    for i, v in enumerate(("Long-only", "Short-only", "Long & short")):
        vals = []
        for tier in TIERS:
            rr = sub[(sub.tier == tier) & (sub.variant == v)]
            vals.append(rr.win_rate_pct.iloc[0] if not rr.empty else 0)
        bars = ax.bar(x + (i - 1) * w, vals, w, color=PAL[v], label=v)
        for b, val in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, val + 0.6, f"{val:.0f}%", ha="center", fontsize=8)
    ax.set_xticks(x); ax.set_xticklabels(TIERS)
    ax.axhline(50, color="grey", ls="--", lw=1, label="50% (coin flip)")
    ax.set_ylabel("Win rate (% of trades profitable)")
    ax.set_title("Win rate by index tier and side  (exit = effective close)")
    ax.legend(); ax.grid(True, axis="y", alpha=0.3)
    _save(fig, "v2_win_rate.png")


def exit_window_chart(summary):
    """ASX200 long vs short mean per-trade across the 3 exit windows."""
    fig, ax = plt.subplots(figsize=(10, 6))
    windows = ["eff", "eff5", "eff10"]
    wl = ["Exit at\neffective", "Exit at\neff + 5 bd", "Exit at\neff + 10 bd"]
    x = np.arange(len(windows)); w = 0.35
    for i, (v, c) in enumerate((("Long-only", PAL["Long-only"]), ("Short-only", PAL["Short-only"]))):
        vals = []
        for win in windows:
            rr = summary[(summary.tier == "ASX200") & (summary.variant == v) & (summary.exit_window == win)]
            vals.append(rr.mean_pct.iloc[0] if not rr.empty else 0)
        bars = ax.bar(x + (i - 0.5) * w, vals, w, color=c, label=v)
        for b, val in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, val + (0.06 if val >= 0 else -0.06),
                    f"{val:+.2f}%", ha="center", va="bottom" if val >= 0 else "top", fontsize=9)
    ax.set_xticks(x); ax.set_xticklabels(wl)
    ax.axhline(0, color="black", lw=0.6)
    ax.set_ylabel("ASX 200 mean return per trade (%)")
    ax.set_title("Does holding past the effective date help?  (ASX 200)")
    ax.legend(); ax.grid(True, axis="y", alpha=0.3)
    _save(fig, "v2_exit_window.png")


def _load_price(tk):
    for d in (PROCESSED_RECONCILED_DIR / "prices", RAW_YAHOO_DIR):
        p = d / f"{tk}.csv"
        if p.exists():
            return pd.read_csv(p, parse_dates=["date"]).sort_values("date").reset_index(drop=True)
    return None


def event_study_chart(trades):
    """Average abnormal price path in event time for ASX200 adds/removes.

    Uses only VALID trades (so no corrupted series). Day 0 = announcement.
    Plots mean cumulative abnormal return (stock minus ASX200) from t-5 to t+12.
    """
    bench = pd.read_csv(PROCESSED_BENCHMARK_DIR / "asx200_benchmark.csv", parse_dates=["date"]).sort_values("date")
    bench = bench.set_index("date")["adjusted_close"].pct_change()

    paths = {"Addition": [], "Removal": []}
    for _, tr in trades[trades.tier == "ASX200"].iterrows():
        df = _load_price(tr["ticker"])
        if df is None:
            continue
        df = df.set_index("date")
        ann = pd.Timestamp(tr["announcement_date"])
        if ann not in df.index:
            after = df.index[df.index >= ann]
            if len(after) == 0:
                continue
            ann = after[0]
        pos = df.index.get_loc(ann)
        lo, hi = max(0, pos - 5), min(len(df), pos + 13)
        win = df.iloc[lo:hi]
        ar = win["adjusted_close"].pct_change() - bench.reindex(win.index)
        ar.index = range(lo - pos, hi - pos)
        paths[tr["action"]].append(ar)

    fig, ax = plt.subplots(figsize=(11, 6.5))
    for action, color in (("Addition", "#2e7d32"), ("Removal", "#c62828")):
        if not paths[action]:
            continue
        m = pd.concat(paths[action], axis=1).mean(axis=1).cumsum() * 100
        ax.plot(m.index, m.values, color=color, lw=2, marker="o", ms=3,
                label=f"{action}s (n={len(paths[action])})")
    ax.axvline(0, color="black", ls="--", lw=1.2, label="Announcement (t=0)")
    ax.axvline(1, color="#1f4e79", ls=":", lw=1.5, label="Strategy ENTRY (t+1)")
    ax.axvline(10, color="grey", ls=":", lw=1.2, label="Effective (~t+10)")
    ax.axhline(0, color="black", lw=0.5)
    ax.set_xlabel("Trading days from announcement")
    ax.set_ylabel("Mean cumulative abnormal return vs ASX 200 (%)")
    ax.set_title("Where does the move happen?  ASX 200 event study (valid trades only)\n"
                 "The addition pop is on t=0; entering at t+1 skips it.")
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3)
    _save(fig, "v2_event_study.png")


def main():
    summary = pd.read_csv(OUTPUTS_DIR / "v2_summary.csv")
    trades = pd.read_csv(OUTPUTS_DIR / "v2_trades.csv",
                          parse_dates=["announcement_date", "effective_date"])
    mean_per_trade_chart(summary)
    win_rate_chart(summary)
    exit_window_chart(summary)
    event_study_chart(trades)


if __name__ == "__main__":
    main()
