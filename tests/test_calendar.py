"""Tests for the rebalance calendar."""

from __future__ import annotations

from datetime import date

import pytest

from asxrebalance.calendar import (
    get_announcement_date,
    get_effective_date,
    get_rebalance_months,
    get_rebalance_window,
    get_reference_date,
    iter_rebalance_windows,
)


def test_rebalance_months_quarterly():
    assert get_rebalance_months("ASX200") == [3, 6, 9, 12]
    assert get_rebalance_months("ASX100") == [3, 6, 9, 12]
    assert get_rebalance_months("ASX50") == [3, 6, 9, 12]


def test_announcement_date_is_first_friday():
    # March 2024: first Friday is the 1st.
    assert get_announcement_date(date(2024, 3, 1)) == date(2024, 3, 1)
    # June 2024: first Friday is the 7th.
    assert get_announcement_date(date(2024, 6, 1)) == date(2024, 6, 7)


def test_effective_date_is_third_friday():
    assert get_effective_date(date(2024, 3, 1)) == date(2024, 3, 15)
    assert get_effective_date(date(2024, 9, 1)) == date(2024, 9, 20)


def test_reference_date_is_before_announcement():
    ann = get_announcement_date(date(2024, 6, 1))
    ref = get_reference_date(date(2024, 6, 1))
    assert ref < ann


def test_window_bundles_dates():
    win = get_rebalance_window("ASX200", date(2024, 3, 1))
    assert win.index_name == "ASX200"
    assert win.announcement_date < win.effective_date


def test_invalid_month_raises():
    with pytest.raises(ValueError):
        get_rebalance_window("ASX200", date(2024, 4, 1))


def test_iter_windows_within_range():
    wins = iter_rebalance_windows("ASX200", date(2024, 1, 1), date(2024, 12, 31))
    assert len(wins) == 4
