"""Rank universe by float-adjusted market cap and tag rank changes."""

from __future__ import annotations

import pandas as pd

from ..config import load_methodology_config


def rank_universe(panel: pd.DataFrame) -> pd.DataFrame:
    """Rank by `avg_float_market_cap` descending; ties broken by liquidity."""
    method = load_methodology_config()["ranking"]
    primary = method["primary_metric"]
    tie = method["tie_breaker"]
    ascending = bool(method.get("ascending", False))

    out = panel.sort_values(
        [primary, tie], ascending=[ascending, ascending], na_position="last",
    ).reset_index(drop=True)
    out["fmc_rank"] = out.index + 1
    return out


def rank_change(prev_ranks: pd.DataFrame, current_ranks: pd.DataFrame,
                suffix: str = "_1m") -> pd.DataFrame:
    """Compute change in rank between two ranking snapshots."""
    a = prev_ranks[["ticker", "fmc_rank"]].rename(columns={"fmc_rank": f"fmc_rank_prev{suffix}"})
    b = current_ranks[["ticker", "fmc_rank"]]
    merged = b.merge(a, on="ticker", how="left")
    merged[f"rank_change{suffix}"] = merged[f"fmc_rank_prev{suffix}"] - merged["fmc_rank"]
    return merged


__all__ = ["rank_universe", "rank_change"]
