"""Map an ASX ticker between canonical, Yahoo, FMP and Bloomberg forms.

The canonical form is the bare ASX code (``"BHP"``) with no suffix. The Yahoo
and FMP forms append ``.AX``. The Bloomberg placeholder appends `` AU Equity``
and is kept here so that paid-data integrations can be added without touching
the rest of the codebase.
"""

from __future__ import annotations

import re

import pandas as pd

from ..paths import PROCESSED_RECONCILED_DIR

_ASX_TICKER_RE = re.compile(r"^[A-Z0-9]{1,8}$")


def normalise_asx_ticker(ticker: str) -> str:
    """Strip suffix/whitespace and uppercase. Accepts BHP, BHP.AX, BHP AU Equity, BHP.AU."""
    if ticker is None:
        raise ValueError("Ticker is None")
    t = str(ticker).strip().upper()
    if not t:
        raise ValueError("Empty ticker")
    for suffix in (" AU EQUITY", ".AU", ".AX", " AU"):
        if t.endswith(suffix):
            t = t[: -len(suffix)]
            break
    t = t.replace(" ", "").replace("/", "")
    if not _ASX_TICKER_RE.match(t):
        raise ValueError(f"Invalid ASX ticker after normalisation: '{t}' (input: '{ticker}')")
    return t


def to_yahoo_ticker(ticker: str) -> str:
    return f"{normalise_asx_ticker(ticker)}.AX"


def to_fmp_ticker(ticker: str) -> str:
    return f"{normalise_asx_ticker(ticker)}.AX"


def to_bloomberg_ticker_placeholder(ticker: str) -> str:
    """Return the Bloomberg form. Not used by the model but kept for paid-data extensions."""
    return f"{normalise_asx_ticker(ticker)} AU Equity"


def load_ticker_master(path=None) -> pd.DataFrame:
    """Load the canonical ticker master CSV, or return an empty frame if missing."""
    p = path or (PROCESSED_RECONCILED_DIR / "ticker_master.csv")
    if not p.exists():
        return pd.DataFrame(
            columns=[
                "canonical_ticker", "yahoo_ticker", "fmp_ticker",
                "company_name", "sector", "industry",
                "first_seen_date", "last_seen_date", "active_flag", "notes",
            ]
        )
    return pd.read_csv(p, parse_dates=["first_seen_date", "last_seen_date"])


def save_ticker_master(df: pd.DataFrame, path=None) -> None:
    p = path or (PROCESSED_RECONCILED_DIR / "ticker_master.csv")
    p.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(p, index=False)


__all__ = [
    "normalise_asx_ticker",
    "to_yahoo_ticker",
    "to_fmp_ticker",
    "to_bloomberg_ticker_placeholder",
    "load_ticker_master",
    "save_ticker_master",
]
