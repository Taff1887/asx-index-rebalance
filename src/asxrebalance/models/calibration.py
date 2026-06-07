"""Probability calibration helpers."""

from __future__ import annotations

import numpy as np
import pandas as pd


def reliability_curve(y_true: np.ndarray, y_proba: np.ndarray, bins: int = 10) -> pd.DataFrame:
    """Empirical reliability curve aggregated into `bins` equal-width buckets."""
    bin_edges = np.linspace(0, 1, bins + 1)
    bucket = np.clip(np.digitize(y_proba, bin_edges) - 1, 0, bins - 1)
    rows = []
    for b in range(bins):
        idx = bucket == b
        n = int(idx.sum())
        if n == 0:
            rows.append({"bucket": b, "mean_pred": np.nan, "empirical": np.nan, "n": 0})
            continue
        rows.append({
            "bucket": b,
            "mean_pred": float(y_proba[idx].mean()),
            "empirical": float(y_true[idx].mean()),
            "n": n,
        })
    return pd.DataFrame(rows)


def brier_score(y_true: np.ndarray, y_proba: np.ndarray) -> float:
    return float(np.mean((y_proba - y_true) ** 2))


__all__ = ["reliability_curve", "brier_score"]
