"""Per-tier ASX rebalance strategy with the user's spec.

Entry:  end of next trading day AFTER announcement (announcement + 1 bday close)
Exit A: effective close
Exit B: effective + 5 bdays close
Exit C: effective + 10 bdays close

Variants per tier:
    long_only   - buy every Addition
    short_only  - short every Removal
    long_short  - buy adds, short removals

Index tiers traded: ASX 20, ASX 50, ASX 100, ASX 200.

For each (tier, variant, exit) combo we compute the EQUAL-WEIGHT QUARTERLY
COMPOUND return: each quarter the per-trade returns within that tier are
averaged (equal capital across the trades that day), then the quarterly
returns are compounded across the strategy lifetime.

Benchmarks (real Yahoo Finance index data):
    ASX 50:  ^AFLI
    ASX 100: ^ATLI
    ASX 200: ^AXJO

Everything reported in percent. No simulated data.

Outputs:
    outputs/per_tier_trades.csv      - every individual trade with prices
    outputs/per_tier_summary.csv     - per-tier per-variant total return %
    docs/figures/per_tier_bars.png   - the bar chart
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

TIERS = ["ASX20", "ASX50", "ASX100", "ASX200"]
BENCHMARK_SYMBOLS = {"ASX50": "ASX50", "ASX100": "ASX100", "ASX200": "ASX200"}


def _load_price(ticker: str) -> pd.DataFrame | None:
    for d in (PROCESSED_RECONCILED_DIR / "prices", RAW_YAHOO_DIR):
        p = d / f"{ticker}.csv"
        if p.exists():
            df = pd.read_csv(p, parse_dates=["date"]).sort_values("date").reset_index(drop=True)
            return df
    return None


def _close_on_or_after(df: pd.DataFrame, target: pd.Timestamp) -> tuple[pd.Timestamp, float] | None:
    sub = df[df["date"] >= target]
    if sub.empty:
        return None
    return sub.iloc[0]["date"], float(sub.iloc[0]["adjusted_close"])


def compute_trade(df: pd.DataFrame, ann: pd.Timestamp, eff: pd.Timestamp,
                  exit_bday_offset: int) -> dict | None:
    entry_target = ann + pd.tseries.offsets.BDay(1)
    exit_target = eff + pd.tseries.offsets.BDay(exit_bday_offset)
    entry = _close_on_or_after(df, entry_target)
    exit_ = _close_on_or_after(df, exit_target)
    if entry is None or exit_ is None:
        return None
    e_date, e_px = entry
    x_date, x_px = exit_
    return {
        "entry_date": e_date, "entry_close": e_px,
        "exit_date": x_date,  "exit_close": x_px,
        "raw_pct": (x_px / e_px - 1) * 100,  # percent
    }


def _benchmark_total_pct(name: str, start: pd.Timestamp, end: pd.Timestamp) -> float | None:
    f = PROCESSED_BENCHMARK_DIR / f"{name.lower()}_benchmark.csv"
    if not f.exists():
        return None
    df = pd.read_csv(f, parse_dates=["date"]).sort_values("date")
    df = df[(df["date"] >= start) & (df["date"] <= end)]
    if len(df) < 2:
        return None
    return float((df["adjusted_close"].iloc[-1] / df["adjusted_close"].iloc[0] - 1) * 100)


def main() -> None:
    labels = pd.read_csv(PROCESSED_LABELS_DIR / "rebalance_labels.csv",
                          parse_dates=["announcement_date", "effective_date"])
    labels = labels[labels["index"].isin(TIERS)].copy()
    print(f"Loaded {len(labels)} events across {labels['announcement_date'].nunique()} quarters\n")

    # Build every trade with all three exit windows.
    rows = []
    skipped = 0
    for _, evt in labels.iterrows():
        df = _load_price(evt["ticker"])
        if df is None or df.empty:
            skipped += 1
            continue
        side = "long" if evt["action"] == "Addition" else "short"
        results = {}
        for label, off in (("eff", 0), ("eff5", 5), ("eff10", 10)):
            r = compute_trade(df, evt["announcement_date"], evt["effective_date"], off)
            if r is not None:
                results[label] = r
        if "eff" not in results:
            skipped += 1
            continue
        base = results["eff"]
        row = {
            "tier": evt["index"],
            "announcement_date": evt["announcement_date"],
            "effective_date": evt["effective_date"],
            "ticker": evt["ticker"],
            "company_name": evt["company_name"],
            "action": evt["action"],
            "side": side,
            "entry_date": base["entry_date"],
            "entry_close": round(base["entry_close"], 4),
        }
        for k, lbl in (("eff", "exit_eff"), ("eff5", "exit_eff+5"), ("eff10", "exit_eff+10")):
            if k in results:
                r = results[k]
                raw = r["raw_pct"]
                row[f"{lbl}_date"] = r["exit_date"]
                row[f"{lbl}_close"] = round(r["exit_close"], 4)
                row[f"{lbl}_raw_pct"] = round(raw, 2)
                row[f"{lbl}_trade_pct"] = round(raw if side == "long" else -raw, 2)
            else:
                row[f"{lbl}_date"] = pd.NaT
                row[f"{lbl}_close"] = float("nan")
                row[f"{lbl}_raw_pct"] = float("nan")
                row[f"{lbl}_trade_pct"] = float("nan")
        rows.append(row)

    trades = pd.DataFrame(rows)
    print(f"Trades computable: {len(trades)} (skipped {skipped} for missing prices)\n")
    print("Per-tier trade counts:")
    print(trades.groupby(["tier", "side"]).size().unstack(fill_value=0).to_string())

    # ---- per-tier per-variant compounded totals --------------------------------
    summary_rows = []
    for tier in TIERS:
        for exit_lbl, col in (("exit_eff", "exit_eff_trade_pct"),
                                 ("exit_eff+5", "exit_eff+5_trade_pct"),
                                 ("exit_eff+10", "exit_eff+10_trade_pct")):
            for variant_lbl, mask in (
                ("Long-only",   trades["side"] == "long"),
                ("Short-only",  trades["side"] == "short"),
                ("Long & short", pd.Series(True, index=trades.index)),
            ):
                sub = trades.loc[(trades["tier"] == tier) & mask, ["announcement_date", col]].dropna()
                if sub.empty:
                    continue
                # Quarterly equal-weight then compound
                by_q = sub.groupby("announcement_date")[col].mean() / 100  # convert % to fraction
                total = (1 + by_q).prod() - 1
                summary_rows.append({
                    "tier": tier,
                    "exit_window": exit_lbl,
                    "variant": variant_lbl,
                    "n_trades": len(sub),
                    "n_quarters": by_q.shape[0],
                    "mean_per_trade_pct": float(sub[col].mean()),
                    "median_per_trade_pct": float(sub[col].median()),
                    "compound_total_return_pct": total * 100,
                })

    summary = pd.DataFrame(summary_rows)

    # ---- benchmark over the same window ----------------------------------------
    start = trades["announcement_date"].min()
    end = trades["exit_eff_date"].max() if not trades["exit_eff_date"].isna().all() else trades["effective_date"].max()
    bench_rows = []
    for tier in ("ASX50", "ASX100", "ASX200"):
        ret = _benchmark_total_pct(tier, start, end)
        bench_rows.append({"tier": tier, "total_return_pct": ret})
    bench = pd.DataFrame(bench_rows)

    # ---- save ------------------------------------------------------------------
    trades.to_csv(OUTPUTS_DIR / "per_tier_trades.csv", index=False)
    summary.to_csv(OUTPUTS_DIR / "per_tier_summary.csv", index=False)
    bench.to_csv(OUTPUTS_DIR / "per_tier_benchmarks.csv", index=False)

    print("\n" + "=" * 80)
    print("PER-TIER SUMMARY (exit at effective)")
    print("=" * 80)
    pivot = (summary[summary["exit_window"] == "exit_eff"]
             .pivot_table(index="tier", columns="variant",
                           values="compound_total_return_pct").round(2))
    pivot = pivot.reindex(TIERS)
    print(pivot.to_string())
    print("\nBenchmarks (% total return over same window):")
    print(bench.round(2).to_string(index=False))

    # ---- bar chart -------------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(16, 7), sharey=True)
    palettes = {
        "Long-only":    "#2e7d32",
        "Short-only":   "#c62828",
        "Long & short": "#1f4e79",
    }
    for ax, (exit_lbl, exit_title) in zip(
        axes,
        [("exit_eff", "Exit at effective (textbook)"),
         ("exit_eff+5",  "Exit at effective + 5 business days"),
         ("exit_eff+10", "Exit at effective + 10 business days")],
    ):
        sub = summary[summary["exit_window"] == exit_lbl]
        positions = np.arange(len(TIERS))
        width = 0.25
        for i, variant in enumerate(("Long-only", "Short-only", "Long & short")):
            vals = []
            for tier in TIERS:
                r = sub[(sub["tier"] == tier) & (sub["variant"] == variant)]
                vals.append(r["compound_total_return_pct"].iloc[0] if not r.empty else 0)
            bars = ax.bar(positions + (i - 1) * width, vals, width=width,
                           color=palettes[variant], label=variant)
            for bar, val in zip(bars, vals):
                if abs(val) > 0.01:
                    ax.text(bar.get_x() + bar.get_width() / 2,
                             val + (1.5 if val >= 0 else -3),
                             f"{val:+.1f}%", ha="center",
                             va="bottom" if val >= 0 else "top",
                             fontsize=8, fontweight="bold")

        ax.set_xticks(positions)
        ax.set_xticklabels(TIERS)
        ax.set_title(exit_title, fontsize=11)
        if ax is axes[0]:
            ax.set_ylabel("Compound total return (%)")
            ax.legend(loc="best", fontsize=9)
        ax.grid(True, axis="y", alpha=0.3)
        ax.axhline(0, color="black", linewidth=0.5)
        # Benchmark dashed lines per tier.
        for tier in ("ASX50", "ASX100", "ASX200"):
            if tier not in TIERS:
                continue
            i_tier = TIERS.index(tier)
            row = bench[bench["tier"] == tier]
            if row.empty or pd.isna(row["total_return_pct"].iloc[0]):
                continue
            val = row["total_return_pct"].iloc[0]
            ax.plot([i_tier - 0.4, i_tier + 0.4], [val, val],
                     color="#616161", linestyle="--", linewidth=1.6)

    fig.suptitle(f"ASX rebalance strategy by tier — all percentages, {start.date()} → {end.date()}",
                  fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(DOCS / "per_tier_bars.png", dpi=140)
    plt.close(fig)
    print(f"\n-> docs/figures/per_tier_bars.png")


if __name__ == "__main__":
    main()
