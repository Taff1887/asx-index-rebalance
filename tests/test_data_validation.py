from __future__ import annotations

import pandas as pd

from asxrebalance.data.validation import (
    compare_price_sources,
    compare_volume_sources,
    detect_missing_observations,
    detect_stale_prices,
    detect_suspicious_jumps,
    generate_data_quality_report,
    load_thresholds,
)


def test_compare_price_sources_severity(small_price_panel, yahoo_panel):
    th = load_thresholds()
    cmp = compare_price_sources(small_price_panel, yahoo_panel, th)
    assert {"close", "adjusted_close"} == set(cmp["field"].unique())
    assert cmp["severity"].isin(["none", "low", "medium", "high", "missing"]).all()


def test_compare_volume_sources(small_price_panel, yahoo_panel):
    th = load_thresholds()
    cmp = compare_volume_sources(small_price_panel, yahoo_panel, th)
    assert "pct_diff" in cmp.columns


def test_detect_missing_observations(small_price_panel, yahoo_panel):
    # Drop one row from yahoo to force a missing observation.
    yahoo_short = yahoo_panel.drop(yahoo_panel.index[0])
    missing = detect_missing_observations(small_price_panel, yahoo_short)
    assert len(missing) >= 1


def test_detect_stale_prices(small_price_panel):
    df = small_price_panel.copy()
    df.loc[df["ticker"] == "AAA", "close"] = 12.0  # constant for AAA
    flagged = detect_stale_prices(df)
    assert (flagged["ticker"] == "AAA").any()


def test_detect_suspicious_jumps(small_price_panel):
    df = small_price_panel.copy()
    df.loc[df.index[3], "adjusted_close"] = df.loc[df.index[3], "adjusted_close"] * 5
    flagged = detect_suspicious_jumps(df)
    assert not flagged.empty


def test_full_report_shape(small_price_panel, yahoo_panel):
    report = generate_data_quality_report(small_price_panel, yahoo_panel)
    assert {"summary", "issues", "price_differences", "volume_differences"} <= set(report.keys())
    assert isinstance(report["summary"], pd.DataFrame)
