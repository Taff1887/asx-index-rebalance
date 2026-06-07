"""Transaction cost models."""

from __future__ import annotations

import numpy as np

from ..config import load_costs_config


BPS = 1e-4


def fixed_costs_bps() -> float:
    """Sum of brokerage + half-spread + slippage + exchange + stamp duty (bps)."""
    c = load_costs_config()["costs"]
    return (
        float(c.get("brokerage_bps", 0))
        + float(c.get("half_spread_bps", 0))
        + float(c.get("slippage_bps", 0))
        + float(c.get("exchange_fees_bps", 0))
        + float(c.get("stamp_duty_bps", 0))
    )


def market_impact_bps(trade_value_aud: float, adv_aud: float) -> float:
    """Simple square-root market-impact model: coef * (trade/ADV)^exponent (bps).

    Falls back to zero when ADV is missing (NaN / None / non-positive) so that
    a missing ADV doesn't poison the trade ledger with NaN costs.
    """
    cfg = load_costs_config()["costs"].get("market_impact", {})
    if not cfg.get("enabled", True):
        return 0.0
    if adv_aud is None or trade_value_aud is None:
        return 0.0
    if not np.isfinite(adv_aud) or adv_aud <= 0:
        return 0.0
    if not np.isfinite(trade_value_aud) or trade_value_aud <= 0:
        return 0.0
    coef = float(cfg.get("coefficient", 0.1))
    expo = float(cfg.get("exponent", 0.5))
    return coef * (trade_value_aud / adv_aud) ** expo * 10000  # convert fraction to bps


def borrow_cost_daily_bps() -> float:
    c = load_costs_config()["costs"]
    return float(c.get("borrow_cost_annual_bps", 0)) / 252.0


def per_trade_cost(trade_value_aud: float, adv_aud: float, side: str) -> float:
    """Return AUD cost for a single trade entry or exit."""
    impact = market_impact_bps(trade_value_aud, adv_aud)
    cost_bps = fixed_costs_bps() + impact
    return abs(trade_value_aud) * cost_bps * BPS


def borrow_cost_for_day(short_value_aud: float) -> float:
    return abs(short_value_aud) * borrow_cost_daily_bps() * BPS


__all__ = [
    "fixed_costs_bps",
    "market_impact_bps",
    "borrow_cost_daily_bps",
    "per_trade_cost",
    "borrow_cost_for_day",
]
