"""Recover delisted / Yahoo-missing ASX tickers from Financial Modeling Prep.

Yahoo Finance purges most delisted ASX names. FMP (with the user's premium key)
still carries many acquired/delisted constituents. This script:

  1. Reads the FMP key from this repo's .env (FMP_API_KEY).
  2. For every ticker in the labels that has NO Yahoo price file, queries FMP's
     stable EOD endpoints:
       full              -> raw OHLCV + vwap
       dividend-adjusted -> adjClose (comparable to Yahoo adjusted_close)
  3. Saves recovered series to data/raw/fmp/<T>.csv AND
     data/processed/reconciled/prices/<T>.csv (so the strategy loaders find them),
     in the standard schema: date,open,high,low,close,adjusted_close,volume,vwap.

FMP standard plans cover currently-listed and many acquired names but not the
oldest delistings; the script reports exactly which tickers were recovered and
which remain unavailable.
"""

from __future__ import annotations

import time

import pandas as pd
import requests
from dotenv import load_dotenv

load_dotenv()

from asxrebalance.config import get_env  # noqa: E402
from asxrebalance.paths import (
    PROCESSED_LABELS_DIR, PROCESSED_RECONCILED_DIR, RAW_FMP_DIR, RAW_YAHOO_DIR,
)

BASE = "https://financialmodelingprep.com/stable"
TIERS = ["ASX20", "ASX50", "ASX100", "ASX200"]


def _fetch(endpoint: str, symbol: str, key: str) -> list[dict] | None:
    url = f"{BASE}/{endpoint}"
    params = {"symbol": symbol, "from": "2011-01-01", "to": "2026-02-01", "apikey": key}
    for attempt in range(3):
        try:
            r = requests.get(url, params=params, timeout=30)
            if r.status_code == 200:
                data = r.json()
                return data if isinstance(data, list) else None
            if r.status_code == 429:
                time.sleep(2 * (attempt + 1)); continue
            return None
        except requests.RequestException:
            time.sleep(1.5 * (attempt + 1))
    return None


def recover_ticker(ticker: str, key: str) -> pd.DataFrame | None:
    sym = f"{ticker}.AX"
    full = _fetch("historical-price-eod/full", sym, key)
    if not full:
        return None
    df = pd.DataFrame(full)
    if df.empty or "close" not in df.columns:
        return None
    df["date"] = pd.to_datetime(df["date"])
    adj = _fetch("historical-price-eod/dividend-adjusted", sym, key)
    if adj:
        a = pd.DataFrame(adj)
        a["date"] = pd.to_datetime(a["date"])
        df = df.merge(a[["date", "adjClose"]], on="date", how="left")
        df["adjusted_close"] = df["adjClose"].fillna(df["close"])
    else:
        df["adjusted_close"] = df["close"]
    for c in ("open", "high", "low", "vwap", "volume"):
        if c not in df.columns:
            df[c] = pd.NA
    out = df[["date", "open", "high", "low", "close", "adjusted_close", "volume", "vwap"]]
    return out.sort_values("date").reset_index(drop=True)


def main() -> None:
    key = get_env("FMP_API_KEY")
    if not key:
        raise SystemExit("No FMP_API_KEY in .env — add it and re-run.")

    labels = pd.read_csv(PROCESSED_LABELS_DIR / "rebalance_labels.csv")
    labels = labels[labels["index"].isin(TIERS)]
    all_tickers = sorted(labels["ticker"].astype(str).unique())
    have = {p.stem for p in RAW_YAHOO_DIR.glob("*.csv")} | {
        p.stem for p in (PROCESSED_RECONCILED_DIR / "prices").glob("*.csv")}
    missing = [t for t in all_tickers if t not in have and t not in ("shares", "iwf")]
    print(f"{len(all_tickers)} tickers; {len(missing)} missing from Yahoo. Trying FMP...\n")

    (PROCESSED_RECONCILED_DIR / "prices").mkdir(parents=True, exist_ok=True)
    RAW_FMP_DIR.mkdir(parents=True, exist_ok=True)
    recovered, failed = [], []
    for t in missing:
        df = recover_ticker(t, key)
        if df is not None and len(df) > 50:
            df.to_csv(RAW_FMP_DIR / f"{t}.csv", index=False)
            df.to_csv(PROCESSED_RECONCILED_DIR / "prices" / f"{t}.csv", index=False)
            recovered.append(t)
            print(f"  recovered {t}: {len(df)} rows "
                  f"({df.date.min().date()} -> {df.date.max().date()})")
        else:
            failed.append(t)

    print(f"\nRecovered {len(recovered)} of {len(missing)} via FMP.")
    print(f"Still unavailable ({len(failed)}): {', '.join(failed)}")
    pd.DataFrame({"ticker": recovered, "source": "fmp"}).to_csv(
        PROCESSED_RECONCILED_DIR / "fmp_recovered.csv", index=False)


if __name__ == "__main__":
    main()
