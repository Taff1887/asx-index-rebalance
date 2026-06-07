"""Shares outstanding loader.

Reads a long-form panel of (date, ticker, shares_outstanding, source) from either
FMP (via a cached CSV in ``data/raw/fmp/``) or a manual CSV in ``data/raw/manual``.
The methodology lets the user override or drop observations when the implied
market cap disagrees materially with the vendor-reported market cap.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd

from ..paths import RAW_FMP_DIR, RAW_MANUAL_DIR
from .ticker_mapping import normalise_asx_ticker


SHARES_COLUMNS = ["date", "ticker", "shares_outstanding", "source", "quality_flag"]


def load_shares_outstanding(path: Path | None = None) -> pd.DataFrame:
    """Load the shares panel. Falls back to an empty frame with the right columns."""
    if path is None:
        # Prefer manual, then aggregated FMP cache.
        manual = RAW_MANUAL_DIR / "shares.csv"
        fmp = RAW_FMP_DIR / "shares.csv"
        path = manual if manual.exists() else fmp
    p = Path(path)
    if not p.exists():
        return pd.DataFrame(columns=SHARES_COLUMNS)
    df = pd.read_csv(p, parse_dates=["date"])
    df["ticker"] = df["ticker"].astype(str).map(normalise_asx_ticker)
    if "quality_flag" not in df.columns:
        df["quality_flag"] = "ok"
    if "source" not in df.columns:
        df["source"] = "csv"
    return df[SHARES_COLUMNS]


def asof_shares(shares: pd.DataFrame, asof: date) -> pd.DataFrame:
    """Return the most recent shares_outstanding per ticker as of `asof`."""
    if shares.empty:
        return shares
    sub = shares.loc[shares["date"] <= pd.Timestamp(asof)]
    if sub.empty:
        return sub
    latest = sub.sort_values("date").groupby("ticker").tail(1)
    return latest.reset_index(drop=True)


__all__ = ["SHARES_COLUMNS", "load_shares_outstanding", "asof_shares"]
