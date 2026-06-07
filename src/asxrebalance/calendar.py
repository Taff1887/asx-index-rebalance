"""Rebalance calendar for the S&P/ASX indices.

The S&P/ASX Australian Indices methodology rebalances four times per year (March,
June, September, December). The convention encoded here:

* announcement date — first Friday of the rebalance month;
* effective date — after-close on the third Friday of the rebalance month;
* reference date — two Fridays before the announcement, approximating the
  "second-last Friday of the month prior" guideline used in the methodology.

The methodology is interpreted from public S&P documentation. All dates and
offsets are configurable in ``config/methodology.yaml`` so the user can shift
them when the official methodology or exchange holidays change.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta

from .config import load_indices_config, load_methodology_config

_WEEKDAYS = {
    "monday": 0,
    "tuesday": 1,
    "wednesday": 2,
    "thursday": 3,
    "friday": 4,
    "saturday": 5,
    "sunday": 6,
}


@dataclass(frozen=True)
class RebalanceWindow:
    """Key dates around a single rebalance event."""

    index_name: str
    rebalance_month: date
    reference_date: date
    announcement_date: date
    effective_date: date


def get_rebalance_months(index_name: str) -> list[int]:
    """Return the months of the year on which the named index rebalances."""
    cfg = load_indices_config()
    if index_name not in cfg["indices"]:
        raise KeyError(f"Unknown index: {index_name}")
    freq = cfg["indices"][index_name].get("rebalance_frequency", "quarterly")
    method = load_methodology_config().get("calendar", {})
    months_quarterly = list(method.get("rebalance_months", [3, 6, 9, 12]))
    if freq == "quarterly":
        return months_quarterly
    if freq == "semi-annual":
        return [months_quarterly[0], months_quarterly[2]]
    if freq == "annual":
        return [months_quarterly[1]]
    raise ValueError(f"Unsupported rebalance frequency for {index_name}: {freq}")


def _nth_weekday(year: int, month: int, weekday: int, n: int) -> date:
    """Return the n-th occurrence (1-indexed) of `weekday` in (year, month)."""
    if n < 1:
        raise ValueError("n must be >= 1")
    first = date(year, month, 1)
    shift = (weekday - first.weekday()) % 7
    return first + timedelta(days=shift + 7 * (n - 1))


def get_announcement_date(rebalance_month: date) -> date:
    """Return the announcement date for the rebalance month."""
    method = load_methodology_config().get("calendar", {})
    weekday = _WEEKDAYS[method.get("announcement_weekday", "friday").lower()]
    n = int(method.get("announcement_week_of_month", 1))
    return _nth_weekday(rebalance_month.year, rebalance_month.month, weekday, n)


def get_effective_date(rebalance_month: date) -> date:
    """Return the effective date for the rebalance month."""
    method = load_methodology_config().get("calendar", {})
    weekday = _WEEKDAYS[method.get("effective_weekday", "friday").lower()]
    n = int(method.get("effective_week_of_month", 3))
    return _nth_weekday(rebalance_month.year, rebalance_month.month, weekday, n)


def get_reference_date(rebalance_month: date) -> date:
    """Return the reference (ranking) date used by the methodology."""
    announcement = get_announcement_date(rebalance_month)
    method = load_methodology_config().get("calendar", {})
    offset_weeks = int(method.get("reference_date_offset_weeks", -2))
    return announcement + timedelta(weeks=offset_weeks)


def get_rebalance_window(index_name: str, rebalance_month: date) -> RebalanceWindow:
    """Return the bundled calendar dates for the given index + month."""
    months = get_rebalance_months(index_name)
    if rebalance_month.month not in months:
        raise ValueError(
            f"{rebalance_month:%Y-%m} is not a rebalance month for {index_name}. "
            f"Allowed months: {months}"
        )
    return RebalanceWindow(
        index_name=index_name,
        rebalance_month=date(rebalance_month.year, rebalance_month.month, 1),
        reference_date=get_reference_date(rebalance_month),
        announcement_date=get_announcement_date(rebalance_month),
        effective_date=get_effective_date(rebalance_month),
    )


def iter_rebalance_windows(index_name: str, start: date, end: date) -> list[RebalanceWindow]:
    """Yield all rebalance windows for an index between `start` and `end` inclusive."""
    months = get_rebalance_months(index_name)
    out: list[RebalanceWindow] = []
    year = start.year
    while year <= end.year:
        for month in months:
            ref_month = date(year, month, 1)
            window = get_rebalance_window(index_name, ref_month)
            if start <= window.effective_date <= end:
                out.append(window)
        year += 1
    return out


__all__ = [
    "RebalanceWindow",
    "get_rebalance_months",
    "get_announcement_date",
    "get_effective_date",
    "get_reference_date",
    "get_rebalance_window",
    "iter_rebalance_windows",
]
