"""Bar chart: baseline (exit at effective) vs optimal (data-driven exit) per variant,
plus real ASX 50 / 100 / 200 indices. Total return only — no cumulative line chart.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from asxrebalance.paths import OUTPUTS_DIR, PROCESSED_BENCHMARK_DIR, REPO_ROOT
from asxrebalance.strategy.performance import summary_metrics

DOCS_FIGURES = REPO_ROOT / "docs" / "figures"
DOCS_FIGURES.mkdir(parents=True, exist_ok=True)


def _strategy_metrics(slug: str) -> dict:
    f = OUTPUTS_DIR / f"strategy_performance_summary_{slug}.csv"
    s = pd.read_csv(f).iloc[0]
    return {
        "total_return": float(s["strategy_total_return"]),
        "cagr": float(s["cagr"]),
        "vol": float(s["vol"]),
        "sharpe": float(s["sharpe"]),
        "max_drawdown": float(s["max_drawdown"]),
        "alpha": float(s["alpha"]),
    }


def _benchmark_metrics(name: str, start: pd.Timestamp, end: pd.Timestamp) -> dict:
    f = PROCESSED_BENCHMARK_DIR / f"{name.lower()}_benchmark.csv"
    df = pd.read_csv(f, parse_dates=["date"]).sort_values("date")
    df = df[(df["date"] >= start) & (df["date"] <= end)]
    s = df.set_index("date")["adjusted_close"].pct_change().dropna()
    m = summary_metrics(s)
    return {
        "total_return": float((1 + s).prod() - 1),
        "cagr": float(m["cagr"]),
        "vol": float(m["vol"]),
        "sharpe": float(m["sharpe"]),
        "max_drawdown": float(m["max_drawdown"]),
        "alpha": None,
    }


def main() -> None:
    base_returns = pd.read_csv(OUTPUTS_DIR / "strategy_returns_announcement_long_short_baseline.csv",
                                parse_dates=["date"])
    start = base_returns["date"].min()
    end = base_returns["date"].max()

    rows = [
        ("Long-only — exit t+10 (textbook)", _strategy_metrics("additions_only_baseline"),    "#2e7d32"),
        ("Long-only — exit t+28 (data-driven)", _strategy_metrics("additions_only_optimal"), "#81c784"),
        ("Short-only — exit t+10 (textbook)", _strategy_metrics("removals_only_baseline"),    "#c62828"),
        ("Short-only — exit t+18 (data-driven)", _strategy_metrics("removals_only_optimal"),  "#ef5350"),
        ("Long/short — exit t+10 (textbook)", _strategy_metrics("announcement_long_short_baseline"), "#1f4e79"),
        ("Long/short — exit t+18 (data-driven)", _strategy_metrics("announcement_long_short_optimal"), "#5b8fbe"),
        ("ASX 50",  _benchmark_metrics("ASX50", start, end),  "#9c27b0"),
        ("ASX 100", _benchmark_metrics("ASX100", start, end), "#ff9800"),
        ("ASX 200", _benchmark_metrics("ASX200", start, end), "#616161"),
    ]

    table = pd.DataFrame([{
        "series": label, "total_return": m["total_return"], "cagr": m["cagr"],
        "vol": m["vol"], "sharpe": m["sharpe"], "max_drawdown": m["max_drawdown"],
        "alpha": m["alpha"],
    } for label, m, _ in rows])
    table.to_csv(OUTPUTS_DIR / "baseline_vs_optimal_metrics.csv", index=False)
    print(table.round(4).to_string(index=False))

    # ---- Bar chart of total returns ---------------------------------------
    fig, ax = plt.subplots(figsize=(14, 7))
    labels = [r[0] for r in rows]
    colors = [r[2] for r in rows]
    vals = [r[1]["total_return"] * 100 for r in rows]
    bars = ax.bar(labels, vals, color=colors)
    for bar, value in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2,
                value + (1.5 if value >= 0 else -2.5),
                f"{value:+.1f}%", ha="center",
                va="bottom" if value >= 0 else "top", fontweight="bold")
    ax.axhline(0, color="black", linewidth=0.5)
    ax.set_ylabel(f"Total return over {start.date()} to {end.date()} (%)")
    ax.set_title("Total return — baseline (textbook) vs data-driven exit window vs real ASX indices")
    ax.grid(True, axis="y", alpha=0.3)
    plt.xticks(rotation=20, ha="right")
    fig.tight_layout()
    fig.savefig(DOCS_FIGURES / "baseline_vs_optimal_bars.png", dpi=140)
    plt.close(fig)
    print(f"\n-> docs/figures/baseline_vs_optimal_bars.png")

    # ---- Sharpe bar chart -------------------------------------------------
    fig, ax = plt.subplots(figsize=(14, 6))
    vals = [r[1]["sharpe"] for r in rows]
    bars = ax.bar(labels, vals, color=colors)
    for bar, value in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2,
                value + (0.04 if value >= 0 else -0.07),
                f"{value:.2f}", ha="center",
                va="bottom" if value >= 0 else "top", fontweight="bold")
    ax.axhline(0, color="black", linewidth=0.5)
    ax.set_ylabel("Sharpe ratio")
    ax.set_title("Sharpe ratio — baseline (textbook) vs data-driven exit window vs real ASX indices")
    ax.grid(True, axis="y", alpha=0.3)
    plt.xticks(rotation=20, ha="right")
    fig.tight_layout()
    fig.savefig(DOCS_FIGURES / "baseline_vs_optimal_sharpe.png", dpi=140)
    plt.close(fig)
    print(f"-> docs/figures/baseline_vs_optimal_sharpe.png")


if __name__ == "__main__":
    main()
