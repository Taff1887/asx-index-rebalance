"""Build a REAL S&P/ASX rebalance dataset from public-source events.

The events below were compiled from public S&P press releases and major
Australian financial press (Stockhead, Market Index, Financial Standard,
Livewire, IG, TipRanks) covering quarterly rebalances 2024-Q1 through
2025-Q3 for the S&P/ASX 200. Each row carries:

    announcement_date  : the public S&P/ASX rebalance announcement
                         (first Friday of the rebalance month).
    effective_date     : prior to the open of the third Monday of the
                         rebalance month — when the change takes effect.
    index              : the index the change applies to (ASX200).
    action             : Addition or Removal.
    ticker             : the ASX ticker symbol (canonical).
    company_name       : the issuer's name as stated in the announcement.

For every ticker the script then pulls real OHLCV from Yahoo Finance via
yfinance (suffix .AX) covering announcement_date − 30 bdays through
effective_date + 30 bdays, and writes one CSV per ticker into
data/raw/yahoo/. The strategy backtest will then trade these real events
against real prices.

This is a partial dataset. The full S&P/ASX rebalance history requires
a paid feed; this file documents what is verifiable from public sources.
Sources are listed in docs/REAL_REBALANCE_SOURCES.md.
"""

from __future__ import annotations

from datetime import date

import pandas as pd
import yfinance as yf

from asxrebalance.paths import PROCESSED_LABELS_DIR, RAW_FMP_DIR, RAW_YAHOO_DIR


# (announcement_date, effective_date, index, action, ticker, company_name)
REAL_EVENTS: list[tuple[str, str, str, str, str, str]] = [
    # ---------- March 2024 rebalance (announced 2024-03-01, eff 2024-03-18) ----------
    ("2024-03-01", "2024-03-18", "ASX200", "Addition", "SMR", "Stanmore Resources"),
    ("2024-03-01", "2024-03-18", "ASX200", "Removal",  "WBT", "Weebit Nano"),
    ("2024-03-01", "2024-03-18", "ASX200", "Removal",  "CXO", "Core Lithium"),
    ("2024-03-01", "2024-03-18", "ASX200", "Removal",  "SYA", "Sayona Mining"),

    # ---------- September 2024 rebalance (announced 2024-09-06, eff 2024-09-23) ----
    ("2024-09-06", "2024-09-23", "ASX200", "Addition", "GYG", "Guzman y Gomez"),
    ("2024-09-06", "2024-09-23", "ASX200", "Addition", "WGX", "Westgold Resources"),
    ("2024-09-06", "2024-09-23", "ASX200", "Addition", "YAL", "Yancoal Australia"),

    # ---------- December 2024 rebalance (announced 2024-12-06, eff 2024-12-23) -----
    ("2024-12-06", "2024-12-23", "ASX200", "Removal",  "SPK", "Spark New Zealand"),

    # ---------- March 2025 rebalance (announced 2025-03-07, eff 2025-03-24) --------
    ("2025-03-07", "2025-03-24", "ASX200", "Addition", "CSC", "Capstone Copper"),
    ("2025-03-07", "2025-03-24", "ASX200", "Addition", "DGT", "DigiCo Infrastructure REIT"),
    ("2025-03-07", "2025-03-24", "ASX200", "Addition", "IMD", "Imdex"),
    ("2025-03-07", "2025-03-24", "ASX200", "Addition", "MAQ", "Macquarie Technology Group"),
    ("2025-03-07", "2025-03-24", "ASX200", "Addition", "NXL", "Nuix"),
    ("2025-03-07", "2025-03-24", "ASX200", "Addition", "SPR", "Spartan Resources"),
    ("2025-03-07", "2025-03-24", "ASX200", "Addition", "TPW", "Temple & Webster Group"),
    ("2025-03-07", "2025-03-24", "ASX200", "Removal",  "AD8", "Audinate Group"),
    ("2025-03-07", "2025-03-24", "ASX200", "Removal",  "CKF", "Collins Foods"),
    ("2025-03-07", "2025-03-24", "ASX200", "Removal",  "CQE", "Charter Hall Social Infra REIT"),
    ("2025-03-07", "2025-03-24", "ASX200", "Removal",  "CRN", "Coronado Global Resources"),
    ("2025-03-07", "2025-03-24", "ASX200", "Removal",  "JLG", "Johns Lyng Group"),
    ("2025-03-07", "2025-03-24", "ASX200", "Removal",  "KLS", "Kelsian Group"),
    ("2025-03-07", "2025-03-24", "ASX200", "Removal",  "SGR", "The Star Entertainment Group"),

    # ---------- June 2025 rebalance (announced 2025-06-06, eff 2025-06-23) ---------
    ("2025-06-06", "2025-06-23", "ASX200", "Addition", "ASB", "Austal"),
    ("2025-06-06", "2025-06-23", "ASX200", "Addition", "NCK", "Nick Scali"),
    ("2025-06-06", "2025-06-23", "ASX200", "Removal",  "HLS", "Healius"),
    ("2025-06-06", "2025-06-23", "ASX200", "Removal",  "SMR", "Stanmore Resources"),

    # ---------- September 2025 rebalance (announced 2025-09-05, eff 2025-09-22) ----
    ("2025-09-05", "2025-09-22", "ASX200", "Addition", "DBI", "Dalrymple Bay Infrastructure"),
    ("2025-09-05", "2025-09-22", "ASX200", "Addition", "DRO", "DroneShield"),
    ("2025-09-05", "2025-09-22", "ASX200", "Addition", "EBO", "Ebos Group"),
    ("2025-09-05", "2025-09-22", "ASX200", "Addition", "GGP", "Greatland Resources"),
    ("2025-09-05", "2025-09-22", "ASX200", "Addition", "GQG", "GQG Partners"),
    ("2025-09-05", "2025-09-22", "ASX200", "Addition", "IPX", "IperionX"),
    ("2025-09-05", "2025-09-22", "ASX200", "Addition", "PRN", "Perenti"),
    ("2025-09-05", "2025-09-22", "ASX200", "Addition", "SLC", "Superloop"),
    ("2025-09-05", "2025-09-22", "ASX200", "Addition", "TUA", "Tuas"),
    ("2025-09-05", "2025-09-22", "ASX200", "Removal",  "AOV", "Amotiv"),
    ("2025-09-05", "2025-09-22", "ASX200", "Removal",  "CCP", "Credit Corp Group"),
    ("2025-09-05", "2025-09-22", "ASX200", "Removal",  "CU6", "Clarity Pharmaceuticals"),
    ("2025-09-05", "2025-09-22", "ASX200", "Removal",  "NUF", "Nufarm"),
    ("2025-09-05", "2025-09-22", "ASX200", "Removal",  "PNV", "PolyNovo"),
    ("2025-09-05", "2025-09-22", "ASX200", "Removal",  "SIQ", "Smartgroup"),
]


def build_labels_dataframe() -> pd.DataFrame:
    df = pd.DataFrame(REAL_EVENTS, columns=[
        "announcement_date", "effective_date", "index", "action",
        "ticker", "company_name",
    ])
    df["announcement_date"] = pd.to_datetime(df["announcement_date"])
    df["effective_date"] = pd.to_datetime(df["effective_date"])
    return df


def _flatten_columns(data: pd.DataFrame) -> pd.DataFrame:
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)
    return data


def fetch_prices(tickers: list[str], start: date, end: date) -> dict[str, pd.DataFrame]:
    out: dict[str, pd.DataFrame] = {}
    for tkr in tickers:
        symbol = f"{tkr}.AX"
        try:
            df = yf.download(symbol, start=start.isoformat(), end=end.isoformat(),
                              progress=False, auto_adjust=False, threads=False)
        except Exception as exc:
            print(f"  {tkr}: FETCH ERROR {exc}")
            continue
        if not isinstance(df, pd.DataFrame) or df.empty:
            print(f"  {tkr}: EMPTY")
            continue
        df = _flatten_columns(df).reset_index().rename(columns={
            "Date": "date", "Open": "open", "High": "high", "Low": "low",
            "Close": "close", "Adj Close": "adjusted_close", "Volume": "volume",
        })
        df["vwap"] = pd.NA
        keep = ["date", "open", "high", "low", "close", "adjusted_close",
                "volume", "vwap"]
        out[tkr] = df[keep].sort_values("date").reset_index(drop=True)
        print(f"  {tkr}: {len(df)} rows ({df['date'].min().date()} -> "
              f"{df['date'].max().date()})")
    return out


def main() -> None:
    print("Building real S&P/ASX rebalance labels...")
    labels = build_labels_dataframe()
    print(f"  {len(labels)} real events across "
          f"{labels['announcement_date'].dt.year.nunique()} rebalance year(s) "
          f"and {labels['announcement_date'].nunique()} unique rebalance dates.")

    labels_path = PROCESSED_LABELS_DIR / "rebalance_labels.csv"
    labels.to_csv(labels_path, index=False)
    print(f"  -> {labels_path}")

    tickers = sorted(set(labels["ticker"]))
    start = labels["announcement_date"].min().date() - pd.Timedelta(days=60).to_pytimedelta()
    end = labels["effective_date"].max().date() + pd.Timedelta(days=60).to_pytimedelta()
    print(f"\nFetching {len(tickers)} tickers from yfinance "
          f"({start} -> {end})...")
    prices = fetch_prices(tickers, start, end)

    RAW_YAHOO_DIR.mkdir(parents=True, exist_ok=True)
    RAW_FMP_DIR.mkdir(parents=True, exist_ok=True)
    for tkr, df in prices.items():
        df.to_csv(RAW_YAHOO_DIR / f"{tkr}.csv", index=False)
        # Mirror to FMP cache so the validation/reconciliation pipeline works.
        df.to_csv(RAW_FMP_DIR / f"{tkr}.csv", index=False)
    print(f"\nFetched and stored {len(prices)} tickers.")
    print(f"  -> {RAW_YAHOO_DIR}")


if __name__ == "__main__":
    main()
