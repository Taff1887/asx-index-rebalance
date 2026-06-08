"""Clean implementation of the rebalance strategy on the verified S&P PDF labels.

Entry rule (per user spec):  end of the next trading day AFTER the announcement.
Exit rules: try three options and compare —
    A. effective_date close                     (textbook, ~10 bdays after entry)
    B. effective_date + 5 bdays close           (hold one extra week)
    C. effective_date + 10 bdays close          (hold two extra weeks)

Strategy variants (per index):
    * long_only      — buy every Addition, no shorts
    * short_only     — short every Removal, no longs
    * long_short     — buy Adds, short Removals

Each trade uses identical notional sized as 1 unit. Per-trade return is
the raw % move between entry and exit close (no leverage assumption, no
position cap). Strategy total return = compound product of per-trade
returns within each variant. The output is the apples-to-apples %
return per variant — what the user asked for.

Outputs:
    outputs/clean_strategy_trades.csv     - every trade with entry/exit price
    outputs/clean_strategy_summary.csv    - per-variant total return + Sharpe
    docs/figures/clean_strategy_bars.png  - the bar chart (% only)
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from asxrebalance.paths import (  # noqa: E402
    OUTPUTS_DIR,
    PROCESSED_BENCHMARK_DIR,
    PROCESSED_LABELS_DIR,
    PROCESSED_RECONCILED_DIR,
    RAW_YAHOO_DIR,
    REPO_ROOT,
)

DOCS = REPO_ROOT / "docs" / "figures"
DOCS.mkdir(parents=True, exist_ok=True)


def _load_price(ticker: str) -> pd.DataFrame | None:
    for d in (PROCESSED_RECONCILED_DIR / "prices", RAW_YAHOO_DIR):
        p = d / f"{ticker}.csv"
        if p.exists():
            return pd.read_csv(p, parse_dates=["date"]).sort_values("date").reset_index(drop=True)
    return None


def _next_close(df: pd.DataFrame, target: pd.Timestamp) -> tuple[pd.Timestamp, float] | None:
    """Return (date, adjusted_close) at or after `target`."""
    sub = df[df["date"] >= target]
    if sub.empty:
        return None
    return sub.iloc[0]["date"], float(sub.iloc[0]["adjusted_close"])


def _calc_trade(df: pd.DataFrame, ann_date: pd.Timestamp,
                 eff_date: pd.Timestamp, exit_offset_bdays: int = 0,
                 ) -> dict | None:
    """Compute one trade: entry at next bday close after ann_date, exit at
    eff_date + `exit_offset_bdays` close (or first trading day after).
    """
    entry_target = ann_date + pd.tseries.offsets.BDay(1)
    exit_target = eff_date + pd.tseries.offsets.BDay(exit_offset_bdays)
    entry = _next_close(df, entry_target)
    exit_ = _next_close(df, exit_target)
    if entry is None or exit_ is None:
        return None
    entry_date, entry_px = entry
    exit_date, exit_px = exit_
    raw_pct = exit_px / entry_px - 1
    return {
        "entry_date": entry_date, "entry_close": round(entry_px, 4),
        "exit_date": exit_date,   "exit_close": round(exit_px, 4),
        "raw_pct": raw_pct,
        "holding_bdays": int(np.busday_count(entry_date.date(), exit_date.date())),
    }


def _portfolio_return(per_trade_pct: pd.Series, n_trades_per_event: int = 1) -> float:
    """Equal-weight buy-and-hold portfolio of per-trade returns.

    Treat every trade as 1 unit; equal-weighted compound return is the
    mean per-trade return aggregated as `(1 + r_avg)^n_buckets - 1`. Since
    we want a single 'strategy total return' number for comparison, we
    just chain the mean per-trade returns: each rebalance you allocate
    your capital equally across that quarter's trades, then re-allocate
    next quarter. With that convention, total return = product over
    quarters of (1 + mean_quarter_return) - 1.
    """
    return float((1 + per_trade_pct.fillna(0)).prod() - 1)


def _by_quarter(per_trade: pd.Series, ann_dates: pd.Series) -> pd.Series:
    """Average per-trade return within each announcement date."""
    return per_trade.groupby(ann_dates).mean()


def _benchmark_return(name: str, start: pd.Timestamp, end: pd.Timestamp) -> float:
    f = PROCESSED_BENCHMARK_DIR / f"{name.lower()}_benchmark.csv"
    if not f.exists():
        return float("nan")
    df = pd.read_csv(f, parse_dates=["date"]).sort_values("date")
    sub = df[(df["date"] >= start) & (df["date"] <= end)]
    if len(sub) < 2:
        return float("nan")
    return float(sub["adjusted_close"].iloc[-1] / sub["adjusted_close"].iloc[0] - 1)


def main() -> None:
    labels = pd.read_csv(PROCESSED_LABELS_DIR / "rebalance_labels.csv",
                          parse_dates=["announcement_date", "effective_date"])
    labels = labels[labels["index"] == "ASX200"].copy()
    print(f"Labels loaded: {len(labels)} ASX 200 events across "
          f"{labels['announcement_date'].nunique()} rebalances "
          f"({labels['announcement_date'].min().date()} -> "
          f"{labels['announcement_date'].max().date()})\n")

    rows = []
    skipped = 0
    for _, evt in labels.iterrows():
        df = _load_price(evt["ticker"])
        if df is None or df.empty:
            skipped += 1
            continue
        # Three exit windows for comparison.
        results = {}
        for label, offset in (("exit_eff", 0), ("exit_eff+5", 5), ("exit_eff+10", 10)):
            r = _calc_trade(df, evt["announcement_date"], evt["effective_date"], offset)
            if r is None:
                continue
            results[label] = r

        if "exit_eff" not in results:
            skipped += 1
            continue

        side = "long" if evt["action"] == "Addition" else "short"
        base = results["exit_eff"]
        row = {
            "announcement_date": evt["announcement_date"],
            "effective_date": evt["effective_date"],
            "ticker": evt["ticker"],
            "company_name": evt["company_name"],
            "action": evt["action"],
            "side": side,
            "entry_date": base["entry_date"],
            "entry_close": base["entry_close"],
            "exit_date_eff": base["exit_date"],
            "exit_close_eff": base["exit_close"],
            "holding_bdays_eff": base["holding_bdays"],
            "raw_pct_eff": base["raw_pct"],
            "trade_pct_eff": base["raw_pct"] if side == "long" else -base["raw_pct"],
        }
        for label, off in (("exit_eff+5", 5), ("exit_eff+10", 10)):
            if label in results:
                r = results[label]
                row[f"exit_date_+{off}"] = r["exit_date"]
                row[f"exit_close_+{off}"] = r["exit_close"]
                row[f"raw_pct_+{off}"] = r["raw_pct"]
                row[f"trade_pct_+{off}"] = r["raw_pct"] if side == "long" else -r["raw_pct"]
            else:
                row[f"exit_date_+{off}"] = pd.NaT
                row[f"exit_close_+{off}"] = float("nan")
                row[f"raw_pct_+{off}"] = float("nan")
                row[f"trade_pct_+{off}"] = float("nan")
        rows.append(row)

    df_trades = pd.DataFrame(rows)
    print(f"Trades computable: {len(df_trades)} (skipped {skipped} for missing prices)\n")

    # ---- aggregate per variant -------------------------------------------------
    summary_rows = []
    for exit_label, col in (("Exit at effective (t+10)", "trade_pct_eff"),
                              ("Exit at t+15 (+5 bdays)",  "trade_pct_+5"),
                              ("Exit at t+20 (+10 bdays)", "trade_pct_+10")):
        for variant_label, mask in (
            ("Long-only (additions)", df_trades["side"] == "long"),
            ("Short-only (removals)", df_trades["side"] == "short"),
            ("Long/short (all)",      pd.Series(True, index=df_trades.index)),
        ):
            sub = df_trades.loc[mask, ["announcement_date", col]].dropna()
            if sub.empty:
                continue
            by_q = sub.groupby("announcement_date")[col].mean()
            total = (1 + by_q).prod() - 1
            mean_per_trade = float(sub[col].mean())
            std_per_trade = float(sub[col].std())
            n = len(sub)
            summary_rows.append({
                "exit_window": exit_label,
                "variant": variant_label,
                "n_trades": n,
                "mean_per_trade_pct": mean_per_trade * 100,
                "median_per_trade_pct": float(sub[col].median()) * 100,
                "std_per_trade_pct": std_per_trade * 100,
                "compound_total_return_pct": total * 100,
            })
    summary = pd.DataFrame(summary_rows)

    # ---- benchmark -------------------------------------------------------------
    start = df_trades["announcement_date"].min()
    end = (df_trades["exit_date_eff"].max() if not df_trades["exit_date_eff"].isna().all()
           else df_trades["effective_date"].max())
    bench = []
    for name in ("ASX50", "ASX100", "ASX200"):
        ret = _benchmark_return(name, start, end)
        bench.append({"benchmark": name, "total_return_pct": ret * 100 if ret == ret else None})
    bench_df = pd.DataFrame(bench)

    # ---- save ------------------------------------------------------------------
    df_trades.to_csv(OUTPUTS_DIR / "clean_strategy_trades.csv", index=False)
    summary.to_csv(OUTPUTS_DIR / "clean_strategy_summary.csv", index=False)
    bench_df.to_csv(OUTPUTS_DIR / "clean_strategy_benchmarks.csv", index=False)

    print("=" * 75)
    print("SUMMARY (compounded total return = product over quarterly buckets)")
    print("=" * 75)
    print(summary.round(2).to_string(index=False))
    print()
    print("Real ASX index benchmarks over the same window:")
    print(bench_df.round(2).to_string(index=False))
    print()

    # ---- bar chart -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(13, 7))
    palette = {
        "Long-only (additions)":   "#2e7d32",
        "Short-only (removals)":   "#c62828",
        "Long/short (all)":        "#1f4e79",
    }

    exit_labels = list(summary["exit_window"].unique())
    variants = list(palette.keys())
    n_groups = len(exit_labels)
    n_vars = len(variants)
    width = 0.22
    x = np.arange(n_groups)
    for i, v in enumerate(variants):
        vals = [summary.loc[(summary["exit_window"] == e) & (summary["variant"] == v),
                              "compound_total_return_pct"].iloc[0]
                if not summary.loc[(summary["exit_window"] == e) & (summary["variant"] == v)].empty
                else 0 for e in exit_labels]
        bars = ax.bar(x + (i - n_vars / 2 + 0.5) * width, vals, width=width,
                       color=palette[v], label=v)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, val + (1.5 if val >= 0 else -2.5),
                     f"{val:+.1f}%", ha="center",
                     va="bottom" if val >= 0 else "top", fontweight="bold", fontsize=9)
    # Add benchmark lines.
    for i, row in bench_df.iterrows():
        if pd.isna(row["total_return_pct"]):
            continue
        ax.axhline(row["total_return_pct"], linestyle="--",
                    color={"ASX50": "#9c27b0", "ASX100": "#ff9800",
                           "ASX200": "#616161"}[row["benchmark"]],
                    linewidth=1.5, label=f"{row['benchmark']} ETF: {row['total_return_pct']:+.1f}%")

    ax.set_xticks(x)
    ax.set_xticklabels(exit_labels, rotation=10)
    ax.set_ylabel("Compound total return (%)")
    ax.set_title(f"Clean ASX 200 rebalance strategy — total return by exit window, "
                  f"{start.date()} → {end.date()}")
    ax.legend(loc="best", fontsize=9)
    ax.grid(True, axis="y", alpha=0.3)
    ax.axhline(0, color="black", linewidth=0.5)
    fig.tight_layout()
    fig.savefig(DOCS / "clean_strategy_bars.png", dpi=140)
    plt.close(fig)
    print("-> docs/figures/clean_strategy_bars.png")


if __name__ == "__main__":
    main()
