"""Fetch real S&P/ASX 50, 100, 200 index data from Yahoo Finance.

Replaces the synthetic ASX 200 proxy under
`data/processed/benchmark/asx200_benchmark.csv` with the actual S&P/ASX 200
index, and writes ASX 50 and ASX 100 alongside.

Symbols:
    ^AXJO -> S&P/ASX 200
    ^ATLI -> S&P/ASX 100
    ^AFLI -> S&P/ASX 50

Run after `python scripts/generate_synthetic_data.py` to swap the synthetic
benchmark with real-market prices. The strategy itself still uses synthetic
trade-level data (real historical rebalance announcements require a paid feed),
but every chart and table that references the benchmark now reflects the
real ASX market.
"""

from __future__ import annotations

import pandas as pd
import yfinance as yf

from asxrebalance.paths import PROCESSED_BENCHMARK_DIR


INDICES = {
    "ASX200": {"symbol": "^AXJO", "filename": "asx200_benchmark.csv"},
    "ASX100": {"symbol": "^ATLI", "filename": "asx100_benchmark.csv"},
    "ASX50":  {"symbol": "^AFLI", "filename": "asx50_benchmark.csv"},
}


def _fetch(symbol: str, start: str, end: str) -> pd.DataFrame:
    df = yf.download(symbol, start=start, end=end,
                      progress=False, auto_adjust=False, threads=False)
    if not isinstance(df, pd.DataFrame) or df.empty:
        raise RuntimeError(f"Empty download for {symbol}")
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.reset_index().rename(columns={
        "Date": "date", "Open": "open", "High": "high", "Low": "low",
        "Close": "close", "Adj Close": "adjusted_close", "Volume": "volume",
    })
    return df


def main(start: str = "2018-01-01", end: str = "2025-12-31") -> None:
    PROCESSED_BENCHMARK_DIR.mkdir(parents=True, exist_ok=True)
    for name, meta in INDICES.items():
        symbol = meta["symbol"]
        print(f"Fetching {name} ({symbol})...")
        df = _fetch(symbol, start, end)
        df["ticker"] = name
        df["source"] = "yfinance"
        keep = ["date", "ticker", "open", "high", "low", "close",
                "adjusted_close", "volume", "source"]
        df = df[keep].sort_values("date").reset_index(drop=True)
        path = PROCESSED_BENCHMARK_DIR / meta["filename"]
        df.to_csv(path, index=False)
        first = df["adjusted_close"].iloc[0]
        last = df["adjusted_close"].iloc[-1]
        print(f"  {len(df)} rows  {df['date'].min().date()} -> {df['date'].max().date()}")
        print(f"  total return: {(last / first - 1) * 100:+.1f}%")
        print(f"  -> {path}")


if __name__ == "__main__":
    main()
