"""Benchmark loader used for buy-and-hold ASX 200 comparison and abnormal returns."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd

from ..config import load_data_sources_config
from ..paths import PROCESSED_BENCHMARK_DIR
from .fmp import FMPPriceLoader
from .yahoo import YahooFinancePriceLoader


def load_benchmark(start: date, end: date, ticker: str | None = None) -> pd.DataFrame:
    """Load the benchmark (default: STW.AX) with CSV / FMP / Yahoo fallback.

    The CSV fallback is preferred when present so offline runs are deterministic
    and don't depend on network availability.
    """
    cfg = load_data_sources_config().get("benchmark", {})
    ticker = ticker or cfg.get("primary", "STW.AX")

    csv_fallback = Path(cfg.get("csv_fallback_path",
                                 PROCESSED_BENCHMARK_DIR / "asx200_benchmark.csv"))
    if csv_fallback.exists():
        df = pd.read_csv(csv_fallback, parse_dates=["date"])
        if "ticker" not in df.columns:
            df["ticker"] = ticker.replace(".AX", "")
        df = df[(df["date"] >= pd.Timestamp(start)) & (df["date"] <= pd.Timestamp(end))]
        if not df.empty:
            return _finalise(df, df["ticker"].iloc[0], source="csv")

    fmp = FMPPriceLoader()
    yahoo = YahooFinancePriceLoader()
    candidates = [ticker, *cfg.get("fallbacks", [])]
    for cand in candidates:
        symbol = cand.replace(".AX", "")
        try:
            df = fmp.get_prices(symbol, start, end)
            if not df.empty:
                return _finalise(df, symbol, source="fmp")
        except Exception:
            pass
        try:
            df = yahoo.get_prices(symbol, start, end)
            if not df.empty:
                return _finalise(df, symbol, source="yahoo")
        except Exception:
            pass

    return pd.DataFrame(columns=["date", "ticker", "close", "adjusted_close", "source"])


def _finalise(df: pd.DataFrame, ticker: str, source: str) -> pd.DataFrame:
    keep = ["date", "ticker", "close", "adjusted_close", "volume"]
    for c in keep:
        if c not in df.columns:
            df[c] = pd.NA
    df = df[keep].copy()
    df["source"] = source
    return df.sort_values("date").reset_index(drop=True)


def buy_and_hold_returns(df: pd.DataFrame) -> pd.DataFrame:
    """Compute daily returns from the benchmark price panel."""
    df = df.sort_values("date").copy()
    df["return"] = df["adjusted_close"].pct_change()
    df["cum_return"] = (1 + df["return"].fillna(0)).cumprod() - 1
    return df


__all__ = ["load_benchmark", "buy_and_hold_returns"]
