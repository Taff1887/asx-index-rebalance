from __future__ import annotations

import pandas as pd

from asxrebalance.data.reconciliation import (
    choose_price_source,
    choose_volume_source,
    create_reconciled_price_panel,
    reconcile_ohlcv,
)


def test_choose_price_prefers_configured_source():
    row = pd.Series({"close_fmp": 10.0, "close_yahoo": 10.05})
    val, src, _ = choose_price_source(row, "fmp")
    assert val == 10.0 and src == "fmp"
    val, src, _ = choose_price_source(row, "yahoo")
    assert val == 10.05 and src == "yahoo"


def test_choose_volume_falls_back_when_missing():
    row = pd.Series({"volume_fmp": float("nan"), "volume_yahoo": 1234})
    val, src, _ = choose_volume_source(row, "fmp")
    assert val == 1234 and src == "yahoo"


def test_reconcile_ohlcv_preserves_provenance(small_price_panel, yahoo_panel):
    panel = reconcile_ohlcv(small_price_panel, yahoo_panel)
    expected = {"chosen_price_source", "chosen_volume_source",
                "price_quality_flag", "volume_quality_flag",
                "reconciliation_notes"}
    assert expected <= set(panel.columns)


def test_create_panel_drops_unreliable(small_price_panel, yahoo_panel):
    panel = create_reconciled_price_panel(small_price_panel, yahoo_panel,
                                          drop_unreliable=True)
    assert "drop" not in panel.columns
    assert len(panel) > 0
