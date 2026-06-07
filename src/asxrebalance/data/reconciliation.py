"""Reconcile FMP and Yahoo price panels into a single time series.

Raw FMP and Yahoo values are preserved in the source CSVs. The reconciled panel
carries the chosen value, the source that provided it, and a quality flag so
that downstream features and the backtest can audit every observation.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from ..config import load_validation_config
from .validation import compare_price_sources, compare_volume_sources, load_thresholds


_SEVERITY_ORDER = {"none": 0, "low": 1, "medium": 2, "high": 3, "missing": 2}


def _severity_at_least(value: str, threshold: str) -> bool:
    return _SEVERITY_ORDER.get(value, 0) >= _SEVERITY_ORDER.get(threshold, 0)


def choose_price_source(row: pd.Series, prefer: str) -> tuple[float, str, str]:
    """Pick a price using the preferred source, falling back when missing."""
    fmp = row.get("close_fmp")
    yahoo = row.get("close_yahoo")
    if prefer == "yahoo":
        if pd.notna(yahoo):
            return float(yahoo), "yahoo", "preferred=yahoo"
        if pd.notna(fmp):
            return float(fmp), "fmp", "yahoo_missing"
    else:
        if pd.notna(fmp):
            return float(fmp), "fmp", "preferred=fmp"
        if pd.notna(yahoo):
            return float(yahoo), "yahoo", "fmp_missing"
    return float("nan"), "none", "both_missing"


def choose_volume_source(row: pd.Series, prefer: str) -> tuple[float, str, str]:
    fmp = row.get("volume_fmp")
    yahoo = row.get("volume_yahoo")
    if prefer == "yahoo":
        if pd.notna(yahoo):
            return float(yahoo), "yahoo", "preferred=yahoo"
        if pd.notna(fmp):
            return float(fmp), "fmp", "yahoo_missing"
    else:
        if pd.notna(fmp):
            return float(fmp), "fmp", "preferred=fmp"
        if pd.notna(yahoo):
            return float(yahoo), "yahoo", "fmp_missing"
    return float("nan"), "none", "both_missing"


def flag_unreliable_observations(severity_price: str, severity_volume: str,
                                  drop_threshold: str) -> tuple[str, str, bool]:
    """Return price flag, volume flag, and a drop indicator."""
    drop = (_severity_at_least(severity_price, drop_threshold)
            or _severity_at_least(severity_volume, drop_threshold))
    return severity_price, severity_volume, drop


def reconcile_ohlcv(fmp_df: pd.DataFrame, yahoo_df: pd.DataFrame,
                    config: dict[str, Any] | None = None) -> pd.DataFrame:
    """Merge FMP and Yahoo into a single reconciled panel.

    The result contains the chosen OHLCV values, the source that won, and per-row
    severity flags so we can audit every observation.
    """
    cfg = config or load_validation_config()["reconciliation"]
    thresholds = load_thresholds()

    price_diffs = compare_price_sources(fmp_df, yahoo_df, thresholds)
    close_sev = (price_diffs[price_diffs["field"] == "close"]
                 .set_index(["date", "ticker"])["severity"].rename("close_severity"))
    adj_sev = (price_diffs[price_diffs["field"] == "adjusted_close"]
               .set_index(["date", "ticker"])["severity"].rename("adj_severity"))
    vol_diffs = compare_volume_sources(fmp_df, yahoo_df, thresholds)
    vol_sev = vol_diffs.set_index(["date", "ticker"])["severity"].rename("volume_severity")

    cols = ["date", "ticker", "open", "high", "low", "close", "adjusted_close", "volume"]
    a = fmp_df[cols].rename(columns={
        "open": "open_fmp", "high": "high_fmp", "low": "low_fmp",
        "close": "close_fmp", "adjusted_close": "adj_fmp", "volume": "volume_fmp",
    })
    b = yahoo_df[cols].rename(columns={
        "open": "open_yahoo", "high": "high_yahoo", "low": "low_yahoo",
        "close": "close_yahoo", "adjusted_close": "adj_yahoo", "volume": "volume_yahoo",
    })
    merged = a.merge(b, on=["date", "ticker"], how="outer")
    merged = merged.join(close_sev, on=["date", "ticker"])
    merged = merged.join(adj_sev, on=["date", "ticker"])
    merged = merged.join(vol_sev, on=["date", "ticker"])
    merged[["close_severity", "adj_severity", "volume_severity"]] = (
        merged[["close_severity", "adj_severity", "volume_severity"]].fillna("none")
    )

    prefer_close = cfg.get("prefer_close_source", "fmp")
    prefer_adj = cfg.get("prefer_adjusted_close_source", "yahoo")
    prefer_volume = cfg.get("prefer_volume_source", "fmp")
    drop_severity = cfg.get("drop_severity", "high")

    chosen_close, chosen_close_src, chosen_close_reason = [], [], []
    chosen_adj, chosen_adj_src = [], []
    chosen_open, chosen_high, chosen_low = [], [], []
    chosen_vol, chosen_vol_src, chosen_vol_reason = [], [], []
    price_flags, volume_flags, drop_flags, notes = [], [], [], []

    for _, row in merged.iterrows():
        c, c_src, c_reason = choose_price_source(row, prefer_close)
        chosen_close.append(c)
        chosen_close_src.append(c_src)
        chosen_close_reason.append(c_reason)

        adj_row = row.copy()
        adj_row["close_fmp"] = row.get("adj_fmp")
        adj_row["close_yahoo"] = row.get("adj_yahoo")
        adj, adj_src, _ = choose_price_source(adj_row, prefer_adj)
        chosen_adj.append(adj)
        chosen_adj_src.append(adj_src)

        # OHL fall back to the source chosen for close
        prefer_ohl = c_src if c_src in ("fmp", "yahoo") else prefer_close
        if prefer_ohl == "fmp":
            chosen_open.append(row.get("open_fmp"))
            chosen_high.append(row.get("high_fmp"))
            chosen_low.append(row.get("low_fmp"))
        else:
            chosen_open.append(row.get("open_yahoo"))
            chosen_high.append(row.get("high_yahoo"))
            chosen_low.append(row.get("low_yahoo"))

        v, v_src, v_reason = choose_volume_source(row, prefer_volume)
        chosen_vol.append(v)
        chosen_vol_src.append(v_src)
        chosen_vol_reason.append(v_reason)

        worst_price_sev = max(
            row["close_severity"], row["adj_severity"], key=_SEVERITY_ORDER.get
        )
        p_flag, v_flag, drop = flag_unreliable_observations(
            worst_price_sev, row["volume_severity"], drop_severity
        )
        price_flags.append(p_flag)
        volume_flags.append(v_flag)
        drop_flags.append(drop)
        notes.append(f"close:{c_reason}; volume:{v_reason}; price_sev:{p_flag}; volume_sev:{v_flag}")

    out = pd.DataFrame({
        "date": merged["date"],
        "ticker": merged["ticker"],
        "open": chosen_open,
        "high": chosen_high,
        "low": chosen_low,
        "close": chosen_close,
        "adjusted_close": chosen_adj,
        "volume": chosen_vol,
        "chosen_price_source": chosen_close_src,
        "chosen_adjusted_source": chosen_adj_src,
        "chosen_volume_source": chosen_vol_src,
        "price_quality_flag": price_flags,
        "volume_quality_flag": volume_flags,
        "reconciliation_notes": notes,
        "drop": drop_flags,
    })
    return out.sort_values(["ticker", "date"]).reset_index(drop=True)


def create_reconciled_price_panel(fmp_df: pd.DataFrame, yahoo_df: pd.DataFrame,
                                   drop_unreliable: bool = True) -> pd.DataFrame:
    """Convenience wrapper that returns only retained observations."""
    panel = reconcile_ohlcv(fmp_df, yahoo_df)
    if drop_unreliable:
        panel = panel.loc[~panel["drop"]].drop(columns=["drop"])
    return panel.reset_index(drop=True)


def summarise_reconciliation(panel: pd.DataFrame) -> pd.DataFrame:
    grp = panel.groupby(["chosen_price_source", "price_quality_flag"]).size()
    return grp.rename("count").reset_index()


__all__ = [
    "choose_price_source",
    "choose_volume_source",
    "flag_unreliable_observations",
    "reconcile_ohlcv",
    "create_reconciled_price_panel",
    "summarise_reconciliation",
]
