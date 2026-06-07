"""Rules-engine backtest harness.

Walks the historical rebalance calendar, runs the rules engine using only
information available before the announcement date, compares against historical
labels (when present), and reports accuracy.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import pandas as pd

from ..calendar import iter_rebalance_windows
from ..config import load_indices_config


@dataclass(frozen=True)
class RebalanceCase:
    index: str
    rebalance_month: date
    asof: date
    labels: pd.DataFrame
    predictions: pd.DataFrame


def evaluate_predictions(predictions: pd.DataFrame, labels: pd.DataFrame) -> dict:
    """Compute simple accuracy metrics for a single rebalance."""
    label_set = set(labels[labels["action"].isin(["Addition", "Promotion"])]["ticker"])
    delete_set = set(labels[labels["action"].isin(["Removal", "Demotion"])]["ticker"])
    pred_add = set(predictions.loc[predictions["predicted_action"].isin(
        ["Addition", "Promotion"]), "ticker"])
    pred_del = set(predictions.loc[predictions["predicted_action"].isin(
        ["Removal", "Demotion"]), "ticker"])

    def _stats(pred: set[str], truth: set[str]) -> dict:
        tp = len(pred & truth)
        fp = len(pred - truth)
        fn = len(truth - pred)
        precision = tp / max(1, tp + fp)
        recall = tp / max(1, tp + fn)
        f1 = 2 * precision * recall / max(1e-12, precision + recall)
        return {"tp": tp, "fp": fp, "fn": fn,
                "precision": precision, "recall": recall, "f1": f1}

    return {
        "additions": _stats(pred_add, label_set),
        "removals": _stats(pred_del, delete_set),
    }


def summarise_backtest(cases: list[RebalanceCase]) -> pd.DataFrame:
    rows = []
    for case in cases:
        metrics = evaluate_predictions(case.predictions, case.labels)
        for action_name, stats in metrics.items():
            rows.append({
                "index": case.index,
                "rebalance_month": case.rebalance_month,
                "action_group": action_name,
                **stats,
            })
    return pd.DataFrame(rows)


def collect_rebalance_windows(index_name: str, start: date, end: date) -> list:
    primary = load_indices_config()["primary_indices"]
    if index_name not in primary:
        raise ValueError(f"{index_name} is not a primary index ({primary})")
    return iter_rebalance_windows(index_name, start, end)


__all__ = ["RebalanceCase", "evaluate_predictions", "summarise_backtest", "collect_rebalance_windows"]
