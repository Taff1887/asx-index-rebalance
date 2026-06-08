"""Probe yfinance for ASX index symbols."""

import pandas as pd
import yfinance as yf


def main() -> None:
    for tkr in ["^AXJO", "^AXFL", "^AXTO", "^AXTL", "^AFLI", "^ATLI",
                "^AXSO", "^AORD"]:
        try:
            df = yf.download(tkr, start="2018-01-01", end="2025-12-31",
                              progress=False, auto_adjust=False, threads=False)
            if isinstance(df, pd.DataFrame) and not df.empty:
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.get_level_values(0)
                print(f"{tkr}: {len(df)} rows  first={df.index.min().date()} "
                      f"last={df.index.max().date()}  cols={list(df.columns)}")
            else:
                print(f"{tkr}: empty")
        except Exception as exc:
            print(f"{tkr}: ERROR {exc}")


if __name__ == "__main__":
    main()
