"""Historical rebalance announcement labels.

Labels are stored as a CSV with one row per ticker per rebalance action. The
CSV can be maintained manually from S&P/ASX announcement PDFs or sourced from
a paid data vendor.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from ..paths import PROCESSED_LABELS_DIR
from .ticker_mapping import normalise_asx_ticker


VALID_ACTIONS = {"Addition", "Removal", "Promotion", "Demotion"}

LABEL_COLUMNS = [
    "announcement_date", "effective_date", "index",
    "action", "ticker", "company_name",
]


def load_labels(path: Path | None = None) -> pd.DataFrame:
    p = path or (PROCESSED_LABELS_DIR / "rebalance_labels.csv")
    if not p.exists():
        return pd.DataFrame(columns=LABEL_COLUMNS)
    df = pd.read_csv(p, parse_dates=["announcement_date", "effective_date"])
    df["ticker"] = df["ticker"].astype(str).map(normalise_asx_ticker)
    df["index"] = df["index"].astype(str).str.upper()
    df["action"] = df["action"].astype(str).str.title()
    return df


def save_labels(df: pd.DataFrame, path: Path | None = None) -> None:
    invalid = set(df["action"]) - VALID_ACTIONS
    if invalid:
        raise ValueError(f"Invalid actions in labels: {sorted(invalid)}")
    p = path or (PROCESSED_LABELS_DIR / "rebalance_labels.csv")
    p.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(p, index=False)


def labels_for_index(df: pd.DataFrame, index_name: str) -> pd.DataFrame:
    return df.loc[df["index"] == index_name.upper()].reset_index(drop=True)


__all__ = ["VALID_ACTIONS", "LABEL_COLUMNS", "load_labels", "save_labels", "labels_for_index"]
