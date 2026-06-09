"""Fetch dividend-inclusive TOTAL-RETURN benchmarks for ASX 50 / 100 / 200.

The price indices (^AXJO, ^AFLI, ^ATLI) exclude dividends. The honest
buy-and-hold comparison must include reinvested dividends. We use ETF adjusted
close, which by construction reinvests distributions:

    ASX 200 total return : STW.AX  (SPDR S&P/ASX 200 ETF) adjusted close
    ASX 50  total return : SFY.AX  (SPDR S&P/ASX 50 ETF)  adjusted close

There is no popular cap-weighted ASX 100 ETF, so ASX 100 total return is
RECONSTRUCTED: take the ASX 100 price index (^ATLI, first bad bar trimmed) and
apply the ASX 200 dividend-reinvestment factor (STW total return / ^AXJO price
return). ASX 100 and ASX 200 have near-identical large-cap dividend yields, so
this is a sound approximation (documented as such).

Outputs: data/processed/benchmark/asx{50,100,200}_tr.csv
"""

from __future__ import annotations

import pandas as pd
import yfinance as yf

from asxrebalance.paths import PROCESSED_BENCHMARK_DIR


def _dl(sym: str) -> pd.DataFrame:
    d = yf.download(sym, start="2012-01-01", end="2026-01-15",
                    progress=False, auto_adjust=False, threads=False)
    if isinstance(d.columns, pd.MultiIndex):
        d.columns = d.columns.get_level_values(0)
    d = d.reset_index().rename(columns={"Date": "date", "Adj Close": "adjusted_close",
                                         "Close": "close", "Volume": "volume"})
    return d[["date", "adjusted_close", "close"]].sort_values("date").reset_index(drop=True)


def _save(df: pd.DataFrame, name: str, ticker: str, note: str):
    df = df.copy()
    df["ticker"] = name
    df["source"] = ticker
    df["note"] = note
    out = PROCESSED_BENCHMARK_DIR / f"{name.lower()}_tr.csv"
    df.to_csv(out, index=False)
    tr = df["adjusted_close"].iloc[-1] / df["adjusted_close"].iloc[0] - 1
    print(f"  {name} TR <- {ticker}: {len(df)} rows {df.date.min().date()}->{df.date.max().date()}  total {tr*100:+.1f}%")


def main() -> None:
    print("Fetching total-return (dividend-inclusive) benchmarks...")
    stw = _dl("STW.AX")    # ASX 200 TR
    sfy = _dl("SFY.AX")    # ASX 50 TR
    _save(stw, "ASX200", "STW.AX (ETF adj close)", "ETF adjusted close = total return incl reinvested dividends")
    _save(sfy, "ASX50", "SFY.AX (ETF adj close)", "ETF adjusted close = total return incl reinvested dividends")

    # Reconstruct ASX 100 TR from ^ATLI price + ASX 200 dividend factor.
    atli = pd.read_csv(PROCESSED_BENCHMARK_DIR / "asx100_benchmark.csv", parse_dates=["date"]).sort_values("date")
    axjo = pd.read_csv(PROCESSED_BENCHMARK_DIR / "asx200_benchmark.csv", parse_dates=["date"]).sort_values("date")
    # Trim ^ATLI's known-bad first bar (4702 spike).
    med0 = atli["adjusted_close"].iloc[1:6].median()
    if atli["adjusted_close"].iloc[0] > med0 * 1.2:
        atli = atli.iloc[1:].reset_index(drop=True)

    m = (stw.rename(columns={"adjusted_close": "stw_tr"})[["date", "stw_tr"]]
         .merge(axjo.rename(columns={"adjusted_close": "axjo_px"})[["date", "axjo_px"]], on="date")
         .merge(atli.rename(columns={"adjusted_close": "atli_px"})[["date", "atli_px"]], on="date"))
    m = m.sort_values("date").reset_index(drop=True)
    # dividend reinvestment factor relative to first common date
    stw_ret = m["stw_tr"] / m["stw_tr"].iloc[0]
    axjo_ret = m["axjo_px"] / m["axjo_px"].iloc[0]
    div_factor = stw_ret / axjo_ret                     # cumulative dividend uplift
    atli_ret = m["atli_px"] / m["atli_px"].iloc[0]
    m["adjusted_close"] = atli_ret * div_factor * 100   # rebased ASX100 TR index
    _save(m[["date", "adjusted_close"]].assign(close=m["adjusted_close"]),
          "ASX100", "^ATLI price x ASX200 dividend factor",
          "RECONSTRUCTED: ASX100 price index x ASX200 dividend-reinvestment factor")


if __name__ == "__main__":
    main()
