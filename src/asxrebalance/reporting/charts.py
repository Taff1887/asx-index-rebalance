"""Chart generation. Matplotlib is used so the repo has no GUI dependency."""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

_DEFAULT_FIGSIZE = (10, 6)


def _save(fig, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


def cumulative_return_chart(strategy: pd.DataFrame, benchmark: pd.DataFrame,
                             path: Path, title: str = "Strategy vs ASX 200 buy-and-hold") -> Path:
    fig, ax = plt.subplots(figsize=_DEFAULT_FIGSIZE)
    if not strategy.empty:
        ax.plot(strategy["date"], (1 + strategy["return"].fillna(0)).cumprod() - 1,
                label="Strategy", color="#1f4e79")
    if not benchmark.empty:
        ax.plot(benchmark["date"],
                (1 + benchmark["benchmark_return"].fillna(0)).cumprod() - 1,
                label="ASX 200 buy-and-hold", color="#a6a6a6", linestyle="--")
    ax.set_title(title)
    ax.set_ylabel("Cumulative return")
    ax.set_xlabel("Date")
    ax.legend()
    ax.grid(True, alpha=0.3)
    return _save(fig, path)


def drawdown_chart(strategy: pd.DataFrame, benchmark: pd.DataFrame, path: Path) -> Path:
    fig, ax = plt.subplots(figsize=_DEFAULT_FIGSIZE)
    for df, col, label, color in (
        (strategy, "return", "Strategy", "#1f4e79"),
        (benchmark, "benchmark_return", "ASX 200 buy-and-hold", "#a6a6a6"),
    ):
        if df.empty or col not in df.columns:
            continue
        cum = (1 + df[col].fillna(0)).cumprod()
        dd = cum / cum.cummax() - 1
        ax.plot(df["date"], dd, label=label, color=color)
    ax.set_title("Drawdowns")
    ax.set_ylabel("Drawdown")
    ax.legend()
    ax.grid(True, alpha=0.3)
    return _save(fig, path)


def rolling_metric_chart(daily_returns: pd.Series, window: int, metric: str,
                          path: Path, title: str | None = None) -> Path:
    fig, ax = plt.subplots(figsize=_DEFAULT_FIGSIZE)
    if metric == "return":
        roll = (1 + daily_returns).rolling(window).apply(np.prod, raw=True) - 1
    elif metric == "sharpe":
        roll = (daily_returns.rolling(window).mean() * 252) / (
            daily_returns.rolling(window).std() * np.sqrt(252)
        )
    else:
        raise ValueError(f"Unknown rolling metric: {metric}")
    roll.index.name = "date"
    ax.plot(roll.index, roll.values, color="#1f4e79")
    ax.set_title(title or f"Rolling {window}-day {metric}")
    ax.grid(True, alpha=0.3)
    return _save(fig, path)


def monthly_heatmap(monthly: pd.DataFrame, path: Path) -> Path:
    fig, ax = plt.subplots(figsize=_DEFAULT_FIGSIZE)
    if monthly.empty:
        ax.text(0.5, 0.5, "No data", ha="center", va="center")
        return _save(fig, path)
    monthly = monthly.copy()
    monthly["year"] = monthly.index.year
    monthly["month"] = monthly.index.month
    pivot = monthly.pivot(index="year", columns="month", values="return")
    im = ax.imshow(pivot.values, cmap="RdYlGn", aspect="auto")
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    ax.set_title("Monthly return heatmap")
    fig.colorbar(im, ax=ax, format="%.1%%")
    return _save(fig, path)


def rebalance_pnl_chart(trades: pd.DataFrame, path: Path) -> Path:
    fig, ax = plt.subplots(figsize=_DEFAULT_FIGSIZE)
    if trades.empty or "announcement_date" not in trades.columns:
        ax.text(0.5, 0.5, "No trades", ha="center", va="center")
        return _save(fig, path)
    grp = (trades.assign(month=pd.to_datetime(trades["announcement_date"]).dt.to_period("M"))
                 .groupby("month")["net_pnl_aud"].sum())
    ax.bar(grp.index.astype(str), grp.values, color="#1f4e79")
    ax.set_title("PnL by rebalance month (AUD)")
    plt.xticks(rotation=45, ha="right")
    return _save(fig, path)


def bar_chart(df: pd.DataFrame, label_col: str, value_col: str, path: Path,
              title: str) -> Path:
    fig, ax = plt.subplots(figsize=_DEFAULT_FIGSIZE)
    ax.bar(df[label_col].astype(str), df[value_col].astype(float), color="#1f4e79")
    ax.set_title(title)
    plt.xticks(rotation=45, ha="right")
    ax.grid(True, axis="y", alpha=0.3)
    return _save(fig, path)


def scatter_chart(df: pd.DataFrame, x: str, y: str, path: Path, title: str) -> Path:
    fig, ax = plt.subplots(figsize=_DEFAULT_FIGSIZE)
    ax.scatter(df[x], df[y], alpha=0.6)
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    return _save(fig, path)


def calibration_chart(df: pd.DataFrame, path: Path) -> Path:
    fig, ax = plt.subplots(figsize=_DEFAULT_FIGSIZE)
    if df.empty:
        ax.text(0.5, 0.5, "No data", ha="center", va="center")
        return _save(fig, path)
    ax.plot([0, 1], [0, 1], "--", color="grey", label="Perfect calibration")
    ax.scatter(df["mean_pred"], df["empirical"], s=df["n"], color="#1f4e79", alpha=0.7,
               label="Bucketed predictions")
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Empirical hit rate")
    ax.set_title("Forecast probability calibration")
    ax.legend()
    ax.grid(True, alpha=0.3)
    return _save(fig, path)


__all__ = [
    "cumulative_return_chart",
    "drawdown_chart",
    "rolling_metric_chart",
    "monthly_heatmap",
    "rebalance_pnl_chart",
    "bar_chart",
    "scatter_chart",
    "calibration_chart",
]
