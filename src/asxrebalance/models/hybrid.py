"""Combine rules-engine signal, ML probability and flow pressure into a hybrid score."""

from __future__ import annotations

import numpy as np
import pandas as pd


DEFAULT_WEIGHTS = {
    "rules_signal": 0.30,
    "ml_probability": 0.40,
    "rank_buffer_margin": 0.10,
    "liquidity_pass": 0.05,
    "flow_pressure": 0.10,
    "historical_hit_rate": 0.025,
    "data_quality_score": 0.025,
}


def _scale(series: pd.Series) -> pd.Series:
    if series.empty:
        return series
    s = series.astype(float).fillna(series.median()).replace([np.inf, -np.inf], np.nan)
    s = s.fillna(s.median())
    rng = s.max() - s.min()
    if rng == 0:
        return pd.Series(0.5, index=s.index)
    return (s - s.min()) / rng


def hybrid_score(panel: pd.DataFrame, weights: dict | None = None) -> pd.DataFrame:
    """Compute hybrid_score in [0, 1] by combining the configured components."""
    w = weights or DEFAULT_WEIGHTS
    out = panel.copy()

    rules_signal = (out.get("predicted_action", "no change")
                       .isin(["Addition", "Promotion"]).astype(float))
    ml_proba = out.get("ml_probability", pd.Series(0.5, index=out.index)).astype(float)
    buffer_margin = -_scale(out.get("rank_buffer_margin", pd.Series(0, index=out.index)))
    liq_pass = out.get("liquidity_pass", pd.Series(True, index=out.index)).astype(float)
    flow_pressure = _scale(out.get("passive_flow_to_ADV_20d",
                                    pd.Series(0, index=out.index)).abs())
    hit_rate = out.get("historical_hit_rate", pd.Series(0.5, index=out.index)).astype(float)
    data_quality = out.get("data_quality_score", pd.Series(1.0, index=out.index)).astype(float)

    score = (
        w.get("rules_signal", 0) * rules_signal
        + w.get("ml_probability", 0) * ml_proba
        + w.get("rank_buffer_margin", 0) * buffer_margin
        + w.get("liquidity_pass", 0) * liq_pass
        + w.get("flow_pressure", 0) * flow_pressure
        + w.get("historical_hit_rate", 0) * hit_rate
        + w.get("data_quality_score", 0) * data_quality
    )
    total_w = sum(w.values()) or 1.0
    out["hybrid_probability"] = (score / total_w).clip(0, 1)
    out["hybrid_score_raw"] = score
    return out


def hybrid_confidence_bucket(p: float) -> str:
    if p >= 0.75:
        return "high conviction"
    if p >= 0.60:
        return "medium conviction"
    if p >= 0.45:
        return "borderline"
    return "no change"


__all__ = ["DEFAULT_WEIGHTS", "hybrid_score", "hybrid_confidence_bucket"]
