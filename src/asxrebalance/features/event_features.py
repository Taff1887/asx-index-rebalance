"""Per-event features supplied to the ML overlay."""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd

from .liquidity import adv


def trailing_returns(panel: pd.DataFrame, asof: date,
                     windows: tuple[int, ...] = (21, 63)) -> pd.DataFrame:
    """Trailing total returns over `windows` business days."""
    end = pd.Timestamp(asof)
    out_rows = []
    for ticker, grp in panel.groupby("ticker"):
        s = grp.sort_values("date").set_index("date")["adjusted_close"]
        if s.empty:
            continue
        row = {"ticker": ticker}
        for w in windows:
            past = s.loc[:end].iloc[-w - 1:-1]
            if len(past) >= w:
                row[f"return_{w}d"] = float(s.loc[:end].iloc[-1] / past.iloc[0] - 1)
            else:
                row[f"return_{w}d"] = np.nan
        out_rows.append(row)
    return pd.DataFrame(out_rows)


def trailing_volatility(panel: pd.DataFrame, asof: date,
                        window: int = 63) -> pd.DataFrame:
    end = pd.Timestamp(asof)
    rows = []
    for ticker, grp in panel.groupby("ticker"):
        s = grp.sort_values("date").set_index("date")["adjusted_close"].pct_change()
        if s.empty:
            continue
        sub = s.loc[:end].iloc[-window:]
        if len(sub) < window // 2:
            vol = np.nan
        else:
            vol = float(sub.std() * np.sqrt(252))
        rows.append({"ticker": ticker, f"volatility_{window}d": vol})
    return pd.DataFrame(rows)


def market_cap_gap_to_cutoff(rank_panel: pd.DataFrame, target_count: int) -> pd.DataFrame:
    """Distance from the cutoff (target_count) measured in absolute float-adjusted market cap."""
    sorted_panel = rank_panel.sort_values("avg_float_market_cap", ascending=False).reset_index(drop=True)
    if len(sorted_panel) < target_count:
        cutoff = sorted_panel["avg_float_market_cap"].min()
    else:
        cutoff = float(sorted_panel.iloc[target_count - 1]["avg_float_market_cap"])
    sorted_panel["market_cap_gap_to_cutoff"] = sorted_panel["avg_float_market_cap"] - cutoff
    return sorted_panel


def build_event_features(rank_panel: pd.DataFrame, price_panel: pd.DataFrame,
                         asof: date, target_count: int) -> pd.DataFrame:
    """Assemble the standard ML feature row per ticker."""
    feats = rank_panel.copy()
    feats = market_cap_gap_to_cutoff(feats, target_count)
    rets = trailing_returns(price_panel, asof)
    vol = trailing_volatility(price_panel, asof)
    adv20 = adv(price_panel, asof, 20)
    adv60 = adv(price_panel, asof, 60)
    for extra in (rets, vol, adv20, adv60):
        feats = feats.merge(extra, on="ticker", how="left")
    return feats


__all__ = [
    "trailing_returns",
    "trailing_volatility",
    "market_cap_gap_to_cutoff",
    "build_event_features",
]
