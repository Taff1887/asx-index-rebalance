"""Six-way comparison: 3 strategy variants vs real ASX 50/100/200 indices.

Reads the strategy daily-returns CSVs and the real-benchmark CSVs (produced
by `scripts/fetch_real_benchmarks.py`), computes Sharpe/vol/MD on a common
calendar denominator, builds:

  * `docs/figures/total_return_bars.png` — bar chart of total return.
  * `docs/figures/six_way_comparison.png` — line chart over real benchmarks.
  * `docs/figures/six_way_drawdown.png` — drawdown chart for all six.
  * `outputs/six_way_metrics.csv` — Sharpe / vol / Calmar / max DD / total
    return table for all six.
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
    REPO_ROOT,
)
from asxrebalance.strategy.performance import summary_metrics  # noqa: E402

DOCS_FIGURES = REPO_ROOT / "docs" / "figures"
DOCS_FIGURES.mkdir(parents=True, exist_ok=True)

PALETTE = {
    "long_short":   "#1f4e79",
    "long_only":    "#2e7d32",
    "short_only":   "#c62828",
    "asx50":        "#9c27b0",
    "asx100":       "#ff9800",
    "asx200":       "#616161",
}
FIGSIZE = (12, 6.5)


def _load_strategy(variant_slug: str) -> pd.Series:
    f = OUTPUTS_DIR / f"strategy_returns_{variant_slug}.csv"
    if not f.exists():
        return pd.Series(dtype=float, name=variant_slug)
    df = pd.read_csv(f, parse_dates=["date"]).sort_values("date").set_index("date")
    return df["return"].rename(variant_slug)


def _load_benchmark(name: str) -> pd.Series:
    f = PROCESSED_BENCHMARK_DIR / f"{name.lower()}_benchmark.csv"
    if not f.exists():
        return pd.Series(dtype=float, name=name)
    df = pd.read_csv(f, parse_dates=["date"]).sort_values("date").set_index("date")
    return df["adjusted_close"].pct_change().rename(name)


def main() -> None:
    series = {
        "Long/short":      (_load_strategy("announcement_long_short"), PALETTE["long_short"]),
        "Long-only":       (_load_strategy("additions_only"),          PALETTE["long_only"]),
        "Short-only":      (_load_strategy("removals_only"),           PALETTE["short_only"]),
        "ASX 50":          (_load_benchmark("ASX50"),                  PALETTE["asx50"]),
        "ASX 100":         (_load_benchmark("ASX100"),                 PALETTE["asx100"]),
        "ASX 200":         (_load_benchmark("ASX200"),                 PALETTE["asx200"]),
    }

    # Restrict every series to the strategy's date window so metrics are
    # apples-to-apples.
    strategy_returns = series["Long/short"][0]
    if not strategy_returns.empty:
        win_start = strategy_returns.index.min()
        win_end = strategy_returns.index.max()
        for name, (s, color) in list(series.items()):
            if s.empty:
                continue
            series[name] = (s.loc[(s.index >= win_start) & (s.index <= win_end)], color)

    # ---- Metrics table ------------------------------------------------------
    rows = []
    for name, (s, _) in series.items():
        if s.empty:
            continue
        s = s.dropna()
        metrics = summary_metrics(s)
        total = float((1 + s).prod() - 1)
        rows.append({
            "series": name,
            "total_return": total,
            "cagr": metrics["cagr"],
            "vol": metrics["vol"],
            "sharpe": metrics["sharpe"],
            "sortino": metrics["sortino"],
            "max_drawdown": metrics["max_drawdown"],
            "calmar": metrics["calmar"],
        })
    table = pd.DataFrame(rows)
    table.to_csv(OUTPUTS_DIR / "six_way_metrics.csv", index=False)
    print(table.round(4).to_string(index=False))

    # ---- Bar chart of total returns ----------------------------------------
    fig, ax = plt.subplots(figsize=FIGSIZE)
    colors = [PALETTE.get(name.lower().replace(" ", "").replace("/", "_").replace("-", "_"),
                          "#777") for name in table["series"]]
    # Manual colour map by display name.
    colour_map = {
        "Long/short": PALETTE["long_short"],
        "Long-only":  PALETTE["long_only"],
        "Short-only": PALETTE["short_only"],
        "ASX 50":     PALETTE["asx50"],
        "ASX 100":    PALETTE["asx100"],
        "ASX 200":    PALETTE["asx200"],
    }
    colors = [colour_map[name] for name in table["series"]]
    bars = ax.bar(table["series"], table["total_return"] * 100, color=colors)
    for bar, value in zip(bars, table["total_return"]):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + (1 if bar.get_height() >= 0 else -3),
                f"{value*100:+.1f}%", ha="center",
                va="bottom" if bar.get_height() >= 0 else "top",
                fontweight="bold")
    ax.axhline(0, color="black", linewidth=0.5)
    win_start = strategy_returns.index.min().date()
    win_end = strategy_returns.index.max().date()
    ax.set_ylabel(f"Total return over {win_start} to {win_end} (%)")
    ax.set_title("Total return — 3 rebalance strategies vs real ASX 50 / 100 / 200")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(DOCS_FIGURES / "total_return_bars.png", dpi=140)
    plt.close(fig)
    print(f"  -> docs/figures/total_return_bars.png")

    # ---- Line chart of cumulative returns ----------------------------------
    fig, ax = plt.subplots(figsize=FIGSIZE)
    for name, (s, color) in series.items():
        if s.empty:
            continue
        s = s.dropna()
        cum = (1 + s).cumprod() - 1
        ax.plot(cum.index, cum.values * 100, color=color, linewidth=2, label=name,
                linestyle="--" if name.startswith("ASX") else "-")
    ax.set_title("Cumulative return — strategies vs real ASX 50 / 100 / 200")
    ax.set_ylabel("Cumulative return (%)")
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(DOCS_FIGURES / "six_way_comparison.png", dpi=140)
    plt.close(fig)
    print(f"  -> docs/figures/six_way_comparison.png")

    # ---- Drawdowns ---------------------------------------------------------
    fig, ax = plt.subplots(figsize=FIGSIZE)
    for name, (s, color) in series.items():
        if s.empty:
            continue
        s = s.dropna()
        cum = (1 + s).cumprod()
        dd = (cum / cum.cummax() - 1) * 100
        ax.plot(dd.index, dd.values, color=color, linewidth=2, label=name,
                linestyle="--" if name.startswith("ASX") else "-")
    ax.set_title("Drawdowns — strategies vs real ASX 50 / 100 / 200")
    ax.set_ylabel("Drawdown (%)")
    ax.legend(loc="lower left")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(DOCS_FIGURES / "six_way_drawdown.png", dpi=140)
    plt.close(fig)
    print(f"  -> docs/figures/six_way_drawdown.png")


if __name__ == "__main__":
    main()
