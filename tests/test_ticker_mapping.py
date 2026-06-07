from __future__ import annotations

import pytest

from asxrebalance.data.ticker_mapping import (
    normalise_asx_ticker,
    to_bloomberg_ticker_placeholder,
    to_fmp_ticker,
    to_yahoo_ticker,
)


@pytest.mark.parametrize("raw,expected", [
    ("BHP", "BHP"),
    ("BHP.AX", "BHP"),
    ("BHP.AU", "BHP"),
    ("BHP AU EQUITY", "BHP"),
    ("bhp", "BHP"),
])
def test_normalise(raw: str, expected: str) -> None:
    assert normalise_asx_ticker(raw) == expected


def test_yahoo_form() -> None:
    assert to_yahoo_ticker("BHP.AX") == "BHP.AX"


def test_fmp_form() -> None:
    assert to_fmp_ticker("BHP") == "BHP.AX"


def test_bloomberg_placeholder() -> None:
    assert to_bloomberg_ticker_placeholder("BHP") == "BHP AU Equity"


def test_invalid_raises() -> None:
    with pytest.raises(ValueError):
        normalise_asx_ticker("BAD-TICKER!!!")
