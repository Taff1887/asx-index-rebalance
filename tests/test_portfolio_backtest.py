from __future__ import annotations

import pandas as pd

from asxrebalance.strategy.execution import backtest_positions
from asxrebalance.strategy.portfolio import enforce_exposure, expand_to_positions, size_positions


def _signals() -> pd.DataFrame:
    return pd.DataFrame({
        "ticker": ["A", "B"],
        "index": ["ASX200"] * 2,
        "side": ["long", "short"],
        "hybrid_probability": [0.9, 0.7],
        "volatility_63d": [0.2, 0.3],
        "passive_flow_to_ADV_20d": [0.5, 0.4],
        "entry_date": pd.to_datetime(["2024-03-04", "2024-03-04"]),
        "exit_date": pd.to_datetime(["2024-03-15", "2024-03-15"]),
        "announcement_date": pd.to_datetime(["2024-03-04"] * 2),
        "effective_date": pd.to_datetime(["2024-03-15"] * 2),
        "rebalance_month": pd.to_datetime(["2024-03-01"] * 2),
        "variant": ["announcement_long_short"] * 2,
    })


def _prices() -> pd.DataFrame:
    dates = pd.bdate_range("2024-03-01", "2024-03-20")
    rows = []
    for t, start in [("A", 100), ("B", 50)]:
        for i, d in enumerate(dates):
            rows.append({"date": d, "ticker": t,
                         "adjusted_close": start * (1 + 0.001 * i)})
    return pd.DataFrame(rows)


def test_size_positions_assigns_weights():
    s = size_positions(_signals())
    assert "weight" in s.columns
    assert (s.loc[s["side"] == "short", "weight"] <= 0).all()


def test_backtest_returns_daily_panel():
    sized = enforce_exposure(size_positions(_signals()))
    positions = expand_to_positions(sized)
    result = backtest_positions(positions, _prices())
    assert not result["daily_returns"].empty
    assert {"date", "return", "cum_return"} <= set(result["daily_returns"].columns)
    assert isinstance(result["summary"], dict)
