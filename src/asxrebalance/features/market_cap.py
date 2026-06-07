"""Market-cap and float-adjusted market-cap features."""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd

from ..config import load_methodology_config
from ..data.iwf import asof_iwf
from ..data.shares import asof_shares


def daily_market_cap(prices: pd.DataFrame, shares: pd.DataFrame) -> pd.DataFrame:
    """Forward-fill shares onto each price observation and compute daily market cap."""
    if prices.empty:
        return prices.assign(shares_outstanding=pd.NA, market_cap=pd.NA)
    p = prices.sort_values(["ticker", "date"]).copy()
    s = shares.sort_values(["ticker", "date"]).copy()
    out_frames = []
    for ticker, grp in p.groupby("ticker"):
        s_t = s.loc[s["ticker"] == ticker, ["date", "shares_outstanding"]]
        merged = pd.merge_asof(
            grp.sort_values("date"), s_t.sort_values("date"),
            on="date", direction="backward",
        )
        out_frames.append(merged)
    out = pd.concat(out_frames, ignore_index=True) if out_frames else p
    out["market_cap"] = out["close"] * out["shares_outstanding"]
    return out


def float_adjusted_market_cap(panel: pd.DataFrame, iwf: pd.DataFrame) -> pd.DataFrame:
    """Multiply market cap by the most recent IWF available per (ticker, date)."""
    if panel.empty:
        return panel.assign(iwf=pd.NA, free_float_market_cap=pd.NA)
    out_frames = []
    for ticker, grp in panel.groupby("ticker"):
        w = iwf.loc[iwf["ticker"] == ticker, ["date", "iwf"]].sort_values("date")
        if w.empty:
            merged = grp.copy()
            merged["iwf"] = 1.0
        else:
            merged = pd.merge_asof(
                grp.sort_values("date"), w, on="date", direction="backward",
            )
            merged["iwf"] = merged["iwf"].fillna(1.0)
        out_frames.append(merged)
    out = pd.concat(out_frames, ignore_index=True)
    out["free_float_market_cap"] = out["market_cap"] * out["iwf"]
    return out


def average_float_market_cap(panel: pd.DataFrame, asof: date,
                              lookback_business_days: int | None = None) -> pd.DataFrame:
    """Average float-adjusted market cap over the methodology lookback window."""
    method = load_methodology_config()["float_adjusted_market_cap"]
    lookback = lookback_business_days or int(method["lookback_business_days"])

    window_end = pd.Timestamp(asof)
    window_start = window_end - pd.tseries.offsets.BDay(lookback)
    sub = panel[(panel["date"] >= window_start) & (panel["date"] <= window_end)]
    if sub.empty:
        return pd.DataFrame(columns=["ticker", "avg_float_market_cap"])
    agg = (sub.groupby("ticker")["free_float_market_cap"]
              .mean().rename("avg_float_market_cap").reset_index())
    return agg


def asof_float_market_cap(panel: pd.DataFrame, asof: date,
                          shares: pd.DataFrame, iwf: pd.DataFrame) -> pd.DataFrame:
    """Convenience: produce avg FMC + the point-in-time FMC for ranking."""
    panel = daily_market_cap(panel, shares)
    panel = float_adjusted_market_cap(panel, iwf)
    avg = average_float_market_cap(panel, asof)
    # Point-in-time FMC on the asof date (or the last business day before).
    sub = panel[panel["date"] <= pd.Timestamp(asof)].sort_values("date")
    latest = sub.groupby("ticker").tail(1)[["ticker", "free_float_market_cap"]]
    out = avg.merge(latest, on="ticker", how="outer")
    out = out.rename(columns={"free_float_market_cap": "pit_float_market_cap"})
    out["avg_float_market_cap"] = out["avg_float_market_cap"].replace({np.nan: pd.NA})
    return out


__all__ = [
    "daily_market_cap",
    "float_adjusted_market_cap",
    "average_float_market_cap",
    "asof_float_market_cap",
]
