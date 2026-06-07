"""Financial Modeling Prep loader.

The loader gracefully falls back to the local FMP cache directory when the
``FMP_API_KEY`` environment variable is empty. This is the path used by the
synthetic-data pipeline and by users who pre-stage CSVs in ``data/raw/fmp/``.
"""

from __future__ import annotations

import logging
import time
from datetime import date
from pathlib import Path

import pandas as pd
import requests

from ..config import get_env, load_data_sources_config
from ..paths import RAW_FMP_DIR
from .ticker_mapping import normalise_asx_ticker, to_fmp_ticker

log = logging.getLogger(__name__)


class FMPPriceLoader:
    """Load OHLCV and reference data from Financial Modeling Prep."""

    SOURCE = "fmp"

    def __init__(self, api_key: str | None = None, cache_dir: Path | None = None) -> None:
        cfg = load_data_sources_config().get("fmp", {})
        self.api_key = api_key if api_key is not None else (get_env("FMP_API_KEY") or "")
        self.base_url = cfg.get("base_url", "https://financialmodelingprep.com/api/v3")
        self.retries = int(cfg.get("retries", 3))
        self.backoff = float(cfg.get("retry_backoff_seconds", 2.0))
        self.cache_dir = Path(cache_dir or cfg.get("cache_dir", RAW_FMP_DIR))
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    # ----- prices --------------------------------------------------------------
    def _cache_path(self, ticker: str) -> Path:
        return self.cache_dir / f"{normalise_asx_ticker(ticker)}.csv"

    def get_prices(self, ticker: str, start: date, end: date) -> pd.DataFrame:
        """Return OHLCV for `ticker` between `start` and `end` inclusive.

        If the API key is missing the loader will read from `self.cache_dir`.
        Cached files are CSVs named ``<canonical>.csv`` with columns
        ``date, open, high, low, close, adjusted_close, volume, vwap``.
        """
        canonical = normalise_asx_ticker(ticker)
        cache_path = self._cache_path(canonical)

        if not self.api_key:
            if cache_path.exists():
                df = pd.read_csv(cache_path, parse_dates=["date"])
            else:
                return self._empty()
        else:
            df = self._fetch_remote(canonical, start, end)
            if df is not None and not df.empty:
                df.to_csv(cache_path, index=False)
            elif cache_path.exists():
                df = pd.read_csv(cache_path, parse_dates=["date"])
            else:
                return self._empty()

        df = df.sort_values("date").reset_index(drop=True)
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
        return df.loc[mask, cols].reset_index(drop=True)

    def _fetch_remote(self, canonical: str, start: date, end: date) -> pd.DataFrame | None:
        endpoint = f"{self.base_url}/historical-price-full/{to_fmp_ticker(canonical)}"
        params = {"from": start.isoformat(), "to": end.isoformat(), "apikey": self.api_key}
        for attempt in range(1, self.retries + 1):
            try:
                r = requests.get(endpoint, params=params, timeout=30)
                if r.status_code == 200:
                    payload = r.json()
                    hist = payload.get("historical", [])
                    if not hist:
                        return self._empty()
                    df = pd.DataFrame(hist)
                    df = df.rename(columns={"adjClose": "adjusted_close"})
                    df["date"] = pd.to_datetime(df["date"])
                    for col in ("open", "high", "low", "close", "adjusted_close", "volume", "vwap"):
                        if col not in df.columns:
                            df[col] = pd.NA
                    return df[["date", "open", "high", "low", "close",
                               "adjusted_close", "volume", "vwap"]]
                log.warning("FMP %s returned %s, attempt %d", canonical, r.status_code, attempt)
            except requests.RequestException as exc:
                log.warning("FMP request failed for %s: %s", canonical, exc)
            time.sleep(self.backoff * attempt)
        return None

    @staticmethod
    def _empty() -> pd.DataFrame:
        return pd.DataFrame(columns=[
            "date", "ticker", "source", "open", "high", "low",
            "close", "adjusted_close", "volume", "vwap",
        ])

    # ----- reference data ------------------------------------------------------
    def get_company_profile(self, ticker: str) -> pd.DataFrame:
        """Best-effort company-profile fetch (sector/industry/market cap snapshot)."""
        canonical = normalise_asx_ticker(ticker)
        cache_path = self.cache_dir / f"{canonical}_profile.csv"
        if not self.api_key and cache_path.exists():
            return pd.read_csv(cache_path)
        if not self.api_key:
            return pd.DataFrame(columns=["symbol", "companyName", "sector", "industry", "mktCap"])
        try:
            r = requests.get(
                f"{self.base_url}/profile/{to_fmp_ticker(canonical)}",
                params={"apikey": self.api_key}, timeout=30,
            )
            if r.status_code == 200:
                df = pd.DataFrame(r.json())
                df.to_csv(cache_path, index=False)
                return df
        except requests.RequestException as exc:
            log.warning("FMP profile failed for %s: %s", canonical, exc)
        return pd.DataFrame(columns=["symbol", "companyName", "sector", "industry", "mktCap"])

    def get_shares_outstanding(self, ticker: str) -> pd.DataFrame:
        """Return a date / shares_outstanding time series. Falls back to local CSV."""
        canonical = normalise_asx_ticker(ticker)
        cache_path = self.cache_dir / f"{canonical}_shares.csv"
        if cache_path.exists():
            return pd.read_csv(cache_path, parse_dates=["date"])
        return pd.DataFrame(columns=["date", "ticker", "shares_outstanding", "source"])


__all__ = ["FMPPriceLoader"]
