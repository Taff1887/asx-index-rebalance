"""Investable Weight Factor (IWF) / free-float loader.

The S&P/ASX methodology float-adjusts the market cap by an Investable Weight
Factor between 0 and 1. We support a CSV with one row per ticker per effective
date, and we fall back to an IWF of 1.0 with a warning when the data is missing.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd

from ..paths import RAW_MANUAL_DIR
from .ticker_mapping import normalise_asx_ticker


IWF_COLUMNS = ["date", "ticker", "iwf", "source", "quality_flag"]


def load_iwf(path: Path | None = None) -> pd.DataFrame:
    p = path or (RAW_MANUAL_DIR / "iwf.csv")
    if not p.exists():
        return pd.DataFrame(columns=IWF_COLUMNS)
    df = pd.read_csv(p, parse_dates=["date"])
    df["ticker"] = df["ticker"].astype(str).map(normalise_asx_ticker)
    if "source" not in df.columns:
        df["source"] = "csv"
    if "quality_flag" not in df.columns:
        df["quality_flag"] = "ok"
    return df[IWF_COLUMNS]


def asof_iwf(iwf: pd.DataFrame, asof: date, tickers: list[str] | None = None) -> pd.DataFrame:
    """Return the most recent IWF per ticker as of `asof`, defaulting missing to 1.0."""
    if iwf.empty:
        if tickers is None:
            return pd.DataFrame(columns=IWF_COLUMNS)
        return pd.DataFrame({
            "date": pd.Timestamp(asof),
            "ticker": tickers,
            "iwf": 1.0,
            "source": "fallback",
            "quality_flag": "iwf_fallback_one",
        })
    sub = iwf.loc[iwf["date"] <= pd.Timestamp(asof)]
    latest = sub.sort_values("date").groupby("ticker").tail(1)
    if tickers is not None:
        missing = sorted(set(tickers) - set(latest["ticker"]))
        if missing:
            fallback = pd.DataFrame({
                "date": pd.Timestamp(asof),
                "ticker": missing,
                "iwf": 1.0,
                "source": "fallback",
                "quality_flag": "iwf_fallback_one",
            })
            latest = pd.concat([latest, fallback], ignore_index=True)
    return latest.reset_index(drop=True)


__all__ = ["IWF_COLUMNS", "load_iwf", "asof_iwf"]
