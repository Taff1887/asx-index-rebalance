from __future__ import annotations

import pandas as pd

from asxrebalance.models.rules_engine import predict_rebalance


def _panel(size: int = 250) -> pd.DataFrame:
    mc = [size - i for i in range(size)]
    return pd.DataFrame({
        "ticker": [f"T{i:03d}" for i in range(size)],
        "avg_float_market_cap": [v * 1_000_000 for v in mc],
        "median_daily_value_traded": [1_000_000 + 1000 * i for i in range(size)],
        "relative_liquidity": 1.0,
        "liquidity_pass": True,
        "fmc_rank": list(range(1, size + 1)),
    })


def test_predict_with_buffers_marks_additions():
    p = _panel(250)
    # Non-members hold the top 20 ranks; the remaining 200 are members.
    # A non-member with rank <= 179 (the addition buffer) must be added.
    members = {p.loc[i, "ticker"] for i in range(20, 220)}
    out = predict_rebalance(p, "ASX200", members)
    additions = out[out["predicted_action"].isin(["Addition", "Promotion"])]
    assert not additions.empty


def test_liquidity_fail_forces_removal():
    p = _panel(250)
    p.loc[p["ticker"] == "T000", "liquidity_pass"] = False
    members = {p.loc[i, "ticker"] for i in range(200)}
    out = predict_rebalance(p, "ASX200", members)
    row = out[out["ticker"] == "T000"].iloc[0]
    assert row["predicted_action"] == "Removal"


def test_pairs_additions_and_deletions():
    p = _panel(250)
    members = {p.loc[i, "ticker"] for i in range(200)}
    out = predict_rebalance(p, "ASX200", members)
    n_add = (out["predicted_action"].isin(["Addition", "Promotion"])).sum()
    n_del = (out["predicted_action"].isin(["Removal", "Demotion"])).sum()
    assert n_add == n_del
