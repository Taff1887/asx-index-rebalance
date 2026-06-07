"""Liquidity features used by the eligibility check and the flow model."""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd

from ..config import load_methodology_config


def daily_value_traded(panel: pd.DataFrame) -> pd.DataFrame:
    out = panel.copy()
    out["daily_value_traded"] = out["close"] * out["volume"]
    return out


def median_daily_value_traded(panel: pd.DataFrame, asof: date,
                              lookback_business_days: int | None = None) -> pd.DataFrame:
    method = load_methodology_config()["liquidity"]
    lookback = lookback_business_days or int(method["lookback_business_days"])
    panel = daily_value_traded(panel)
    end = pd.Timestamp(asof)
    start = end - pd.tseries.offsets.BDay(lookback)
    sub = panel[(panel["date"] >= start) & (panel["date"] <= end)]
    return (sub.groupby("ticker")["daily_value_traded"]
              .median().rename("median_daily_value_traded").reset_index())


def adv(panel: pd.DataFrame, asof: date, days: int) -> pd.DataFrame:
    end = pd.Timestamp(asof)
    start = end - pd.tseries.offsets.BDay(days)
    sub = panel[(panel["date"] >= start) & (panel["date"] <= end)]
    if sub.empty:
        return pd.DataFrame(columns=["ticker", f"ADV_{days}d"])
    return (sub.groupby("ticker")["volume"]
              .mean().rename(f"ADV_{days}d").reset_index())


def stock_liquidity_ratio(mdvt: pd.DataFrame, avg_fmc: pd.DataFrame) -> pd.DataFrame:
    """Median daily value traded divided by average float-adjusted market cap."""
    merged = mdvt.merge(avg_fmc, on="ticker", how="inner")
    merged["stock_liquidity_ratio"] = (
        merged["median_daily_value_traded"] / merged["avg_float_market_cap"]
    )
    return merged


def market_liquidity(ratio: pd.DataFrame, weights: pd.DataFrame) -> float:
    """Cap-weighted average of stock_liquidity_ratio across the eligible universe."""
    if "avg_float_market_cap" in ratio.columns:
        w = ratio["avg_float_market_cap"].fillna(0).astype(float)
        s = ratio["stock_liquidity_ratio"].fillna(0).astype(float)
    else:
        merged = ratio.merge(weights, on="ticker", how="inner")
        w = merged["avg_float_market_cap"].fillna(0).astype(float)
        s = merged["stock_liquidity_ratio"].fillna(0).astype(float)
    if w.sum() <= 0:
        return float("nan")
    return float((s * w).sum() / w.sum())


def relative_liquidity(ratio: pd.DataFrame, weights: pd.DataFrame) -> pd.DataFrame:
    """Per-stock liquidity ratio relative to the market liquidity."""
    mkt = market_liquidity(ratio, weights)
    out = ratio.copy()
    out["relative_liquidity"] = out["stock_liquidity_ratio"] / mkt if mkt else np.nan
    method = load_methodology_config()["liquidity"]
    threshold = float(method.get("minimum_relative_liquidity", 0.5))
    abs_min = float(method.get("minimum_absolute_daily_value_aud", 0))
    out["liquidity_pass"] = (
        (out["relative_liquidity"] >= threshold)
        & (out["median_daily_value_traded"] >= abs_min)
    )
    return out


__all__ = [
    "daily_value_traded",
    "median_daily_value_traded",
    "adv",
    "stock_liquidity_ratio",
    "market_liquidity",
    "relative_liquidity",
]
