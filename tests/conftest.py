"""Shared pytest fixtures and synthetic frames."""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def small_price_panel() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    dates = pd.bdate_range("2024-01-02", periods=80)
    tickers = ["AAA", "BBB", "CCC", "DDD"]
    rows = []
    for i, t in enumerate(tickers):
        base = 10 + i * 5
        for j, d in enumerate(dates):
            close = base * np.exp(0.0001 * j + 0.01 * rng.standard_normal())
            rows.append({
                "date": d, "ticker": t, "source": "fmp",
                "open": close * 0.999, "high": close * 1.01, "low": close * 0.99,
                "close": close, "adjusted_close": close, "volume": 1000 + 10 * j,
                "vwap": close,
            })
    return pd.DataFrame(rows)


@pytest.fixture
def yahoo_panel(small_price_panel) -> pd.DataFrame:
    df = small_price_panel.copy()
    df["source"] = "yahoo"
    df["close"] = df["close"] * 1.001  # 10 bp difference
    return df


@pytest.fixture
def shares_panel() -> pd.DataFrame:
    return pd.DataFrame([
        {"date": pd.Timestamp("2024-01-01"), "ticker": "AAA",
         "shares_outstanding": 1_000_000, "source": "synthetic", "quality_flag": "ok"},
        {"date": pd.Timestamp("2024-01-01"), "ticker": "BBB",
         "shares_outstanding": 2_000_000, "source": "synthetic", "quality_flag": "ok"},
        {"date": pd.Timestamp("2024-01-01"), "ticker": "CCC",
         "shares_outstanding": 1_500_000, "source": "synthetic", "quality_flag": "ok"},
        {"date": pd.Timestamp("2024-01-01"), "ticker": "DDD",
         "shares_outstanding": 800_000, "source": "synthetic", "quality_flag": "ok"},
    ])


@pytest.fixture
def iwf_panel() -> pd.DataFrame:
    return pd.DataFrame([
        {"date": pd.Timestamp("2024-01-01"), "ticker": "AAA",
         "iwf": 0.7, "source": "synthetic", "quality_flag": "ok"},
        {"date": pd.Timestamp("2024-01-01"), "ticker": "BBB",
         "iwf": 0.5, "source": "synthetic", "quality_flag": "ok"},
        {"date": pd.Timestamp("2024-01-01"), "ticker": "CCC",
         "iwf": 1.0, "source": "synthetic", "quality_flag": "ok"},
        {"date": pd.Timestamp("2024-01-01"), "ticker": "DDD",
         "iwf": 1.0, "source": "synthetic", "quality_flag": "ok"},
    ])
