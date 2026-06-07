"""Corporate-action flags used as features and as data-quality signals.

The model treats splits, dividends and capital raises as features, not as
adjustments — the reconciled price panel already carries an adjusted close
for total-return computations. The functions here translate raw CA data into
boolean flags that can be merged into the feature panel.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from ..paths import RAW_MANUAL_DIR
from .ticker_mapping import normalise_asx_ticker


CA_COLUMNS = ["date", "ticker", "type", "value", "notes"]


def load_corporate_actions(path: Path | None = None) -> pd.DataFrame:
    p = path or (RAW_MANUAL_DIR / "corporate_actions.csv")
    if not p.exists():
        return pd.DataFrame(columns=CA_COLUMNS)
    df = pd.read_csv(p, parse_dates=["date"])
    df["ticker"] = df["ticker"].astype(str).map(normalise_asx_ticker)
    df["type"] = df["type"].astype(str).str.lower()
    return df


def corporate_action_flag(ca: pd.DataFrame, asof: pd.Timestamp,
                           lookback_days: int = 60) -> pd.DataFrame:
    """Boolean per-ticker flag indicating a corporate action in the lookback window."""
    if ca.empty:
        return pd.DataFrame(columns=["ticker", "corporate_action_flag"])
    window_start = asof - pd.Timedelta(days=lookback_days)
    in_window = ca[(ca["date"] >= window_start) & (ca["date"] <= asof)]
    flagged = set(in_window["ticker"])
    return pd.DataFrame({"ticker": sorted(flagged),
                         "corporate_action_flag": True})


__all__ = ["CA_COLUMNS", "load_corporate_actions", "corporate_action_flag"]
