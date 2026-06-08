"""Fetch yfinance prices for every ticker that appears in
data/processed/labels/rebalance_labels.csv but doesn't already have a CSV
under data/raw/yahoo/.

Pulls roughly announcement_date − 60 bdays through effective_date + 60 bdays
for each event, then writes one CSV per ticker into data/raw/yahoo/ and
mirrors to data/raw/fmp/ so the existing reconciliation pipeline works.
"""

from __future__ import annotations

import pandas as pd
import yfinance as yf

from asxrebalance.paths import (
    PROCESSED_LABELS_DIR,
    RAW_FMP_DIR,
    RAW_YAHOO_DIR,
)


def _flatten(df: pd.DataFrame) -> pd.DataFrame:
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df


def main() -> None:
    labels = pd.read_csv(PROCESSED_LABELS_DIR / "rebalance_labels.csv",
                          parse_dates=["announcement_date", "effective_date"])
    tickers = sorted(labels["ticker"].astype(str).unique())
    print(f"{len(tickers)} unique tickers in labels CSV")

    existing = {p.stem for p in RAW_YAHOO_DIR.glob("*.csv")
                 if p.stem not in ("shares", "iwf")}
    missing = [t for t in tickers if t not in existing]
    print(f"{len(existing)} already have data, {len(missing)} need fetching")

    if not missing:
        print("Nothing to do.")
        return

    start = (labels["announcement_date"].min() - pd.Timedelta(days=90)).date().isoformat()
    end = (labels["effective_date"].max() + pd.Timedelta(days=90)).date().isoformat()
    print(f"Fetch window: {start} -> {end}\n")

    fetched, empty = 0, []
    for tkr in missing:
        symbol = f"{tkr}.AX"
        try:
            df = yf.download(symbol, start=start, end=end,
                              progress=False, auto_adjust=False, threads=False)
        except Exception as exc:
            print(f"  {tkr}: ERROR {exc}")
            empty.append(tkr)
            continue
        if not isinstance(df, pd.DataFrame) or df.empty:
            print(f"  {tkr}: empty (likely delisted)")
            empty.append(tkr)
            continue
        df = _flatten(df).reset_index().rename(columns={
            "Date": "date", "Open": "open", "High": "high", "Low": "low",
            "Close": "close", "Adj Close": "adjusted_close", "Volume": "volume",
        })
        df["vwap"] = pd.NA
        keep = ["date", "open", "high", "low", "close", "adjusted_close",
                "volume", "vwap"]
        df = df[keep].sort_values("date").reset_index(drop=True)
        df.to_csv(RAW_YAHOO_DIR / f"{tkr}.csv", index=False)
        df.to_csv(RAW_FMP_DIR / f"{tkr}.csv", index=False)
        fetched += 1
        if fetched % 20 == 0:
            print(f"  ...fetched {fetched} so far")

    print(f"\nFetched {fetched} new tickers.")
    print(f"Empty / delisted: {len(empty)}")
    for t in empty:
        print(f"  {t}")


if __name__ == "__main__":
    main()
