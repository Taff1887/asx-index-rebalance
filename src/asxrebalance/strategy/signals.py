"""Generate strategy signals from forecast events."""

from __future__ import annotations

from datetime import date, timedelta
from typing import Iterable

import pandas as pd


def build_event_signals(forecast: pd.DataFrame,
                        variant: str = "announcement_long_short",
                        exit_offset_days: int = 0) -> pd.DataFrame:
    """Convert a forecast frame into long/short signals with entry and exit dates.

    Required columns on `forecast`:
        ticker, index, predicted_action, hybrid_probability,
        announcement_date, effective_date, passive_flow_to_ADV_20d.
    Optional column: confidence_bucket.

    Parameters
    ----------
    exit_offset_days
        Business-day offset applied to the effective date to derive the exit.
        Negative values exit before the effective close, positive values hold
        past it. Set in ``config/strategy.yaml::exit_timing.exit_offset_days``.
    """
    df = forecast.copy()
    df["side"] = df["predicted_action"].map({
        "Addition": "long",
        "Promotion": "long",
        "Removal": "short",
        "Demotion": "short",
    }).fillna("none")
    df = df.loc[df["side"] != "none"].copy()

    if variant == "pre_announcement":
        df["entry_date"] = df["announcement_date"] - pd.Timedelta(days=5)
    elif variant == "additions_only":
        df = df[df["side"] == "long"].copy()
        df["entry_date"] = df["announcement_date"]
    elif variant == "removals_only":
        df = df[df["side"] == "short"].copy()
        df["entry_date"] = df["announcement_date"]
    elif variant == "flow_pressure":
        df = df[df["passive_flow_to_ADV_20d"].fillna(0).abs() >= 0.5].copy()
        df["entry_date"] = df["announcement_date"]
    else:
        df["entry_date"] = df["announcement_date"]

    df["exit_date"] = df["effective_date"]
    if exit_offset_days:
        df["exit_date"] = df["effective_date"] + pd.tseries.offsets.BDay(exit_offset_days)
        # Never exit before entry — clamp.
        df["exit_date"] = df[["entry_date", "exit_date"]].max(axis=1)

    df["variant"] = variant
    return df.reset_index(drop=True)


def apply_filters(signals: pd.DataFrame, min_prob: float = 0.0,
                  min_flow_to_adv: float = 0.0,
                  high_conviction_only: bool = False,
                  exclude_high_severity: bool = False) -> pd.DataFrame:
    """Apply the configured filters from `strategy.yaml`."""
    s = signals.copy()
    if min_prob:
        s = s[s.get("hybrid_probability", 0).fillna(0) >= min_prob]
    if min_flow_to_adv:
        s = s[s.get("passive_flow_to_ADV_20d", 0).fillna(0).abs() >= min_flow_to_adv]
    if high_conviction_only and "confidence_bucket" in s.columns:
        s = s[s["confidence_bucket"].isin(["high conviction", "medium conviction"])]
    if exclude_high_severity and "data_quality_flag" in s.columns:
        s = s[s["data_quality_flag"] != "high"]
    return s.reset_index(drop=True)


def top_k_signals(signals: pd.DataFrame, k: int) -> pd.DataFrame:
    """Per (rebalance_month, index, side) keep only the top-k by hybrid_probability."""
    if "rebalance_month" not in signals.columns:
        signals = signals.copy()
        signals["rebalance_month"] = signals["announcement_date"].dt.to_period("M").dt.to_timestamp()
    return (signals.sort_values("hybrid_probability", ascending=False)
                   .groupby(["rebalance_month", "index", "side"]).head(k)
                   .reset_index(drop=True))


__all__ = ["build_event_signals", "apply_filters", "top_k_signals"]
