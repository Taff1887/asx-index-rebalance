"""Build a verifiable trade ledger.

For every event in the real labels CSV, look up the actual entry close and
exit close from the per-ticker price CSVs and compute the raw holding-period
return. This is the table a reader can cross-check on Yahoo Finance —
plug in the ticker + dates and they should see the same numbers.

Outputs:
    outputs/verifiable_trades.csv   - long-form ledger with prices
"""

from __future__ import annotations

import pandas as pd

from asxrebalance.paths import (
    OUTPUTS_DIR,
    PROCESSED_LABELS_DIR,
    PROCESSED_RECONCILED_DIR,
    RAW_YAHOO_DIR,
)


def _load_price_series(ticker: str) -> pd.DataFrame | None:
    """Load the per-ticker price CSV used by the backtest."""
    prefer = PROCESSED_RECONCILED_DIR / "prices" / f"{ticker}.csv"
    fallback = RAW_YAHOO_DIR / f"{ticker}.csv"
    for p in (prefer, fallback):
        if p.exists():
            df = pd.read_csv(p, parse_dates=["date"])
            return df.sort_values("date").reset_index(drop=True)
    return None


def _price_on(df: pd.DataFrame, target: pd.Timestamp,
               which: str = "next") -> tuple[pd.Timestamp, float] | None:
    """Return (date, adjusted_close) on or after `target`.

    The strategy entry/exit happens at the close of the announcement / effective
    date. If that date isn't a trading day we use the next trading day's close
    (consistent with how the backtest expands positions onto the bdate range).
    """
    on_or_after = df[df["date"] >= target]
    if on_or_after.empty:
        return None
    row = on_or_after.iloc[0]
    return row["date"], float(row["adjusted_close"])


def main() -> None:
    labels = pd.read_csv(PROCESSED_LABELS_DIR / "rebalance_labels.csv",
                          parse_dates=["announcement_date", "effective_date"])

    rows = []
    for _, evt in labels.iterrows():
        df = _load_price_series(evt["ticker"])
        if df is None or df.empty:
            rows.append({
                "announcement_date": evt["announcement_date"],
                "effective_date": evt["effective_date"],
                "ticker": evt["ticker"],
                "company_name": evt["company_name"],
                "action": evt["action"],
                "side": "long" if evt["action"] in ("Addition", "Promotion") else "short",
                "entry_date": pd.NaT,
                "entry_close_aud": float("nan"),
                "exit_date": pd.NaT,
                "exit_close_aud": float("nan"),
                "holding_days": pd.NA,
                "raw_pct": float("nan"),
                "trade_pct": float("nan"),
                "note": "no price data (delisted or unmapped)",
            })
            continue

        entry = _price_on(df, evt["announcement_date"])
        exit_ = _price_on(df, evt["effective_date"])
        if entry is None or exit_ is None:
            rows.append({
                "announcement_date": evt["announcement_date"],
                "effective_date": evt["effective_date"],
                "ticker": evt["ticker"],
                "company_name": evt["company_name"],
                "action": evt["action"],
                "side": "long" if evt["action"] in ("Addition", "Promotion") else "short",
                "entry_date": pd.NaT,
                "entry_close_aud": float("nan"),
                "exit_date": pd.NaT,
                "exit_close_aud": float("nan"),
                "holding_days": pd.NA,
                "raw_pct": float("nan"),
                "trade_pct": float("nan"),
                "note": "missing entry or exit price",
            })
            continue

        entry_date, entry_px = entry
        exit_date, exit_px = exit_
        raw_pct = (exit_px / entry_px - 1) * 100
        # Trade return: long captures the raw move; short captures the inverse.
        side = "long" if evt["action"] in ("Addition", "Promotion") else "short"
        trade_pct = raw_pct if side == "long" else -raw_pct

        rows.append({
            "announcement_date": evt["announcement_date"],
            "effective_date": evt["effective_date"],
            "ticker": evt["ticker"],
            "company_name": evt["company_name"],
            "action": evt["action"],
            "side": side,
            "entry_date": entry_date,
            "entry_close_aud": round(entry_px, 4),
            "exit_date": exit_date,
            "exit_close_aud": round(exit_px, 4),
            "holding_days": int((exit_date - entry_date).days),
            "raw_pct": round(raw_pct, 2),
            "trade_pct": round(trade_pct, 2),
            "note": "",
        })

    df = pd.DataFrame(rows).sort_values(["announcement_date", "ticker"])
    path = OUTPUTS_DIR / "verifiable_trades.csv"
    df.to_csv(path, index=False)
    print(f"Wrote {len(df)} rows to {path}")
    print()
    print("TOP 10 BY TRADE RETURN (excluding NaN):")
    print(df.dropna(subset=["trade_pct"]).sort_values("trade_pct", ascending=False).head(10)
          [["announcement_date","effective_date","ticker","action","side",
            "entry_close_aud","exit_close_aud","raw_pct","trade_pct"]].to_string(index=False))
    print()
    print("BOTTOM 10 BY TRADE RETURN:")
    print(df.dropna(subset=["trade_pct"]).sort_values("trade_pct", ascending=True).head(10)
          [["announcement_date","effective_date","ticker","action","side",
            "entry_close_aud","exit_close_aud","raw_pct","trade_pct"]].to_string(index=False))
    print()
    print("AGGREGATE BY SIDE (mean trade % before costs):")
    agg = (df.dropna(subset=["trade_pct"]).groupby("side")
             .agg(n=("ticker","size"), mean_trade_pct=("trade_pct","mean"))
             .round(2))
    print(agg.to_string())


if __name__ == "__main__":
    main()
