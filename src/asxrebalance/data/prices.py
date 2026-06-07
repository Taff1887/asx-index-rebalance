"""CSV-based loader and a placeholder for paid-data vendors (Bloomberg, FactSet, ...)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .ticker_mapping import normalise_asx_ticker


REQUIRED_PRICE_COLUMNS = [
    "date", "ticker", "source", "open", "high", "low",
    "close", "adjusted_close", "volume", "vwap",
]


class CSVPriceLoader:
    """Read prices from a CSV directory or single file. Useful for offline runs."""

    SOURCE = "csv"

    def __init__(self, path: Path) -> None:
        self.path = Path(path)

    def get_prices(self, ticker: str, start, end) -> pd.DataFrame:
        canonical = normalise_asx_ticker(ticker)
        if self.path.is_dir():
            f = self.path / f"{canonical}.csv"
            if not f.exists():
                return pd.DataFrame(columns=REQUIRED_PRICE_COLUMNS)
            df = pd.read_csv(f, parse_dates=["date"])
        else:
            df = pd.read_csv(self.path, parse_dates=["date"])
            df = df.loc[df["ticker"] == canonical]
        for c in REQUIRED_PRICE_COLUMNS:
            if c not in df.columns:
                df[c] = pd.NA
        df["ticker"] = canonical
        df["source"] = df.get("source", "csv").fillna("csv")
        start_ts = pd.Timestamp(start)
        end_ts = pd.Timestamp(end)
        return df.loc[(df["date"] >= start_ts) & (df["date"] <= end_ts),
                      REQUIRED_PRICE_COLUMNS].reset_index(drop=True)


class PaidDataLoader:
    """Stub for Bloomberg / FactSet / Refinitiv / S&P paid feeds.

    A real implementation would talk to the vendor's API, return data in the same
    schema as :class:`CSVPriceLoader`, and never silently overwrite raw data.
    """

    def __init__(self, vendor: str | None = None) -> None:
        self.vendor = vendor

    def get_prices(self, ticker: str, start, end) -> pd.DataFrame:  # pragma: no cover - stub
        raise NotImplementedError(
            "PaidDataLoader is a placeholder. Implement vendor-specific logic before use."
        )


__all__ = ["CSVPriceLoader", "PaidDataLoader", "REQUIRED_PRICE_COLUMNS"]
