from __future__ import annotations

import pandas as pd

from asxrebalance.features.liquidity import (
    adv,
    daily_value_traded,
    market_liquidity,
    median_daily_value_traded,
    relative_liquidity,
    stock_liquidity_ratio,
)


def test_daily_value_traded(small_price_panel):
    out = daily_value_traded(small_price_panel)
    assert "daily_value_traded" in out.columns
    assert (out["daily_value_traded"] >= 0).all()


def test_median_daily_value_traded(small_price_panel):
    mdvt = median_daily_value_traded(small_price_panel, pd.Timestamp("2024-04-01").date())
    assert "median_daily_value_traded" in mdvt.columns
    assert len(mdvt) > 0


def test_relative_liquidity(small_price_panel):
    mdvt = median_daily_value_traded(small_price_panel, pd.Timestamp("2024-04-01").date())
    fmc = pd.DataFrame({"ticker": mdvt["ticker"],
                        "avg_float_market_cap": [1e8] * len(mdvt)})
    ratio = stock_liquidity_ratio(mdvt, fmc)
    rel = relative_liquidity(ratio, fmc)
    assert "relative_liquidity" in rel.columns
    assert "liquidity_pass" in rel.columns


def test_adv(small_price_panel):
    out = adv(small_price_panel, pd.Timestamp("2024-03-01").date(), 20)
    assert "ADV_20d" in out.columns
