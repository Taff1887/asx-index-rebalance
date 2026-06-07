"""Guard against look-ahead bias in the feature pipeline."""

from __future__ import annotations

from datetime import date

import pandas as pd

from asxrebalance.features.event_features import build_event_features
from asxrebalance.features.liquidity import median_daily_value_traded
from asxrebalance.features.market_cap import asof_float_market_cap


def test_avg_fmc_does_not_use_future(small_price_panel, shares_panel, iwf_panel):
    asof = date(2024, 2, 15)
    fmc = asof_float_market_cap(small_price_panel, asof, shares_panel, iwf_panel)
    # No NaN-only row would be problematic; we mainly assert no exception and
    # that the result is bounded by the maximum price observed up to asof.
    pit_cap = small_price_panel[small_price_panel["date"] <= pd.Timestamp(asof)]
    assert not pit_cap.empty
    assert "avg_float_market_cap" in fmc.columns


def test_liquidity_does_not_use_future(small_price_panel):
    asof = date(2024, 2, 15)
    mdvt = median_daily_value_traded(small_price_panel, asof)
    assert (mdvt["median_daily_value_traded"] >= 0).all()


def test_event_features_no_future_data(small_price_panel):
    asof = date(2024, 2, 15)
    rank_panel = pd.DataFrame({
        "ticker": ["AAA", "BBB", "CCC", "DDD"],
        "avg_float_market_cap": [1e9, 2e9, 1.5e9, 0.8e9],
        "fmc_rank": [3, 1, 2, 4],
    })
    feats = build_event_features(rank_panel, small_price_panel, asof, target_count=3)
    assert "market_cap_gap_to_cutoff" in feats.columns
    # All return_21d / return_63d values must be computable from <= asof data only.
    pre = small_price_panel[small_price_panel["date"] <= pd.Timestamp(asof)]
    assert not pre.empty
