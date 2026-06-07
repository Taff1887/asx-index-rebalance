"""Yahoo Finance loader (yfinance) with a local cache fallback.

Yahoo data is used as the cross-check for FMP data and as the source of adjusted
prices for total-return calculations.
"""

from __future__ import annotations

import logging
from datetime import date
from pathlib import Path

import pandas as pd

from ..config import load_data_sources_config
from ..paths import RAW_YAHOO_DIR
from .ticker_mapping import normalise_asx_ticker, to_yahoo_ticker

log = logging.getLogger(__name__)


class YahooFinancePriceLoader:
    """Load OHLCV from Yahoo Finance via yfinance, falling back to local cache."""

    SOURCE = "yahoo"

    def __init__(self, cache_dir: Path | None = None) -> None:
        cfg = load_data_sources_config().get("yahoo", {})
        self.cache_dir = Path(cache_dir or cfg.get("cache_dir", RAW_YAHOO_DIR))
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.auto_adjust = bool(cfg.get("auto_adjust", False))

    def _cache_path(self, canonical: str) -> Path:
        return self.cache_dir / f"{canonical}.csv"

    def get_prices(self, ticker: str, start: date, end: date) -> pd.DataFrame:
        canonical = normalise_asx_ticker(ticker)
        cache_path = self._cache_path(canonical)
        df: pd.DataFrame | None = None
        try:
            import yfinance as yf  # type: ignore
            data = yf.download(
                tickers=to_yahoo_ticker(canonical),
                start=start.isoformat(),
                end=(pd.Timestamp(end) + pd.Timedelta(days=1)).date().isoformat(),
                progress=False,
                auto_adjust=self.auto_adjust,
                actions=False,
                threads=False,
            )
            if isinstance(data, pd.DataFrame) and not data.empty:
                if isinstance(data.columns, pd.MultiIndex):
                    # Recent yfinance returns (field, ticker) multi-index columns
                    # even for single-ticker downloads.  Flatten to the field name.
                    data.columns = data.columns.get_level_values(0)
                df = data.reset_index().rename(columns={
                    "Date": "date",
                    "Open": "open",
                    "High": "high",
                    "Low": "low",
                    "Close": "close",
                    "Adj Close": "adjusted_close",
                    "Volume": "volume",
                })
                if "adjusted_close" not in df.columns:
                    df["adjusted_close"] = df["close"]
                df["vwap"] = pd.NA
                df.to_csv(cache_path, index=False)
        except Exception as exc:
            log.warning("Yahoo fetch failed for %s: %s", canonical, exc)

        if df is None:
            if cache_path.exists():
                df = pd.read_csv(cache_path, parse_dates=["date"])
            else:
                return self._empty()

        df["date"] = pd.to_datetime(df["date"])
        df["ticker"] = canonical
        df["source"] = self.SOURCE
        start_ts = pd.Timestamp(start)
        end_ts = pd.Timestamp(end)
        mask = (df["date"] >= start_ts) & (df["date"] <= end_ts)
        cols = ["date", "ticker", "source", "open", "high", "low",
                "close", "adjusted_close", "volume", "vwap"]
        for c in cols:
            if c not in df.columns:
                df[c] = pd.NA
        return df.loc[mask, cols].sort_values("date").reset_index(drop=True)

    @staticmethod
    def _empty() -> pd.DataFrame:
        return pd.DataFrame(columns=[
            "date", "ticker", "source", "open", "high", "low",
            "close", "adjusted_close", "volume", "vwap",
        ])


__all__ = ["YahooFinancePriceLoader"]
