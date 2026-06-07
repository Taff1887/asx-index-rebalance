from __future__ import annotations

import pandas as pd

from asxrebalance.features.rankings import rank_change, rank_universe


def test_rank_universe_orders_by_market_cap():
    df = pd.DataFrame({
        "ticker": ["A", "B", "C"],
        "avg_float_market_cap": [100, 300, 200],
        "median_daily_value_traded": [1, 2, 3],
    })
    ranked = rank_universe(df)
    assert ranked.loc[ranked["ticker"] == "B", "fmc_rank"].iloc[0] == 1
    assert ranked.loc[ranked["ticker"] == "C", "fmc_rank"].iloc[0] == 2
    assert ranked.loc[ranked["ticker"] == "A", "fmc_rank"].iloc[0] == 3


def test_rank_change():
    prev = pd.DataFrame({"ticker": ["A", "B"], "fmc_rank": [2, 1]})
    cur = pd.DataFrame({"ticker": ["A", "B"], "fmc_rank": [1, 2]})
    out = rank_change(prev, cur, suffix="_1m")
    assert int(out.loc[out["ticker"] == "A", "rank_change_1m"].iloc[0]) == 1
    assert int(out.loc[out["ticker"] == "B", "rank_change_1m"].iloc[0]) == -1
