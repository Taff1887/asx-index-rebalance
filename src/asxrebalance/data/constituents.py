"""Index constituent management.

Historical constituents are loaded from CSV (one row per ticker per date) and
exposed via helper functions that respect index hierarchy.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd

from ..paths import PROCESSED_RECONCILED_DIR
from .ticker_mapping import normalise_asx_ticker


CONSTITUENT_COLUMNS = ["date", "index", "ticker", "company_name", "iwf"]


def load_constituents(path: Path | None = None) -> pd.DataFrame:
    """Return the long-form constituent panel."""
    p = path or (PROCESSED_RECONCILED_DIR / "constituents.csv")
    if not p.exists():
        return pd.DataFrame(columns=CONSTITUENT_COLUMNS)
    df = pd.read_csv(p, parse_dates=["date"])
    df["ticker"] = df["ticker"].astype(str).map(normalise_asx_ticker)
    df["index"] = df["index"].astype(str).str.upper()
    return df


def save_constituents(df: pd.DataFrame, path: Path | None = None) -> None:
    p = path or (PROCESSED_RECONCILED_DIR / "constituents.csv")
    p.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(p, index=False)


def current_constituents(df: pd.DataFrame, index_name: str, asof: date) -> pd.DataFrame:
    """Return constituents of `index_name` as of `asof`, picking the most recent snapshot."""
    if df.empty:
        return df
    sub = df[(df["index"] == index_name.upper()) & (df["date"] <= pd.Timestamp(asof))]
    if sub.empty:
        return sub
    latest_date = sub["date"].max()
    return sub.loc[sub["date"] == latest_date].copy()


def index_hierarchy_classification(df: pd.DataFrame, asof: date) -> pd.DataFrame:
    """Tag each ticker with the highest-tier index it belongs to as of `asof`."""
    tiers = ["ASX20", "ASX50", "ASX100", "ASX200", "ASX300", "ALLORDINARIES"]
    asof_panel = df[df["date"] <= pd.Timestamp(asof)].copy()
    if asof_panel.empty:
        return pd.DataFrame(columns=["ticker", "highest_index"])

    asof_panel["index"] = asof_panel["index"].str.upper()
    latest = (asof_panel.sort_values("date")
              .groupby(["ticker", "index"])
              .tail(1))
    pivot = latest.pivot_table(index="ticker", columns="index", values="iwf",
                                aggfunc="last").fillna(0).astype(bool)
    out = []
    for ticker, row in pivot.iterrows():
        highest = None
        for tier in tiers:
            if tier in row.index and row[tier]:
                highest = tier
                break
        out.append({"ticker": ticker, "highest_index": highest})
    return pd.DataFrame(out)


__all__ = [
    "CONSTITUENT_COLUMNS",
    "load_constituents",
    "save_constituents",
    "current_constituents",
    "index_hierarchy_classification",
]
