"""Passive flow / index impact estimation."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import load_strategy_config


def expected_index_weights(fmc_panel: pd.DataFrame, index_name: str,
                           target_count: int) -> pd.DataFrame:
    """Compute target index weights for the top `target_count` securities by FMC."""
    top = fmc_panel.sort_values("avg_float_market_cap", ascending=False).head(target_count).copy()
    total = top["avg_float_market_cap"].fillna(0).sum()
    if total <= 0:
        top["expected_weight"] = np.nan
    else:
        top["expected_weight"] = top["avg_float_market_cap"] / total
    top["index"] = index_name
    return top[["ticker", "index", "avg_float_market_cap", "expected_weight"]]


def passive_flow(events: pd.DataFrame, passive_aum_by_index: dict | None = None,
                 weight_col: str = "expected_weight") -> pd.DataFrame:
    """Translate weight changes into passive AUD flow.

    `events` columns: ticker, index, action (Addition/Removal/Promotion/Demotion),
    expected_weight (post-rebalance), prior_weight (pre-rebalance, 0 if new).
    """
    aum = passive_aum_by_index or load_strategy_config()["passive_aum"]
    out = events.copy()
    out["aum_assumed"] = out["index"].map(aum).astype(float)
    out["weight_change"] = out[weight_col].fillna(0) - out.get("prior_weight",
                                                                pd.Series(0, index=out.index))
    out["passive_flow_estimate"] = out["weight_change"] * out["aum_assumed"]
    return out


def flow_to_adv(flow_df: pd.DataFrame, adv_panel: pd.DataFrame,
                price_col: str = "ref_price") -> pd.DataFrame:
    """Convert AUD flow to days-of-ADV.

    `adv_panel` is expected to contain ADV_20d and ADV_60d in shares and `ref_price` in AUD.
    """
    merged = flow_df.merge(adv_panel, on="ticker", how="left")
    for window in ("20d", "60d"):
        col = f"ADV_{window}"
        if col in merged.columns:
            merged[f"passive_flow_to_ADV_{window}"] = np.where(
                (merged[col] > 0) & merged[price_col].notna(),
                merged["passive_flow_estimate"].abs()
                    / (merged[col] * merged[price_col]),
                np.nan,
            )
    return merged


__all__ = ["expected_index_weights", "passive_flow", "flow_to_adv"]
