"""Benchmark daily-return panel for strategy comparison."""

from __future__ import annotations

import pandas as pd


def benchmark_daily_returns(benchmark_panel: pd.DataFrame) -> pd.DataFrame:
    if benchmark_panel.empty:
        return pd.DataFrame(columns=["date", "benchmark_return"])
    df = benchmark_panel.sort_values("date").copy()
    df["benchmark_return"] = df["adjusted_close"].pct_change()
    return df[["date", "benchmark_return"]].reset_index(drop=True)


__all__ = ["benchmark_daily_returns"]
