"""Probabilistic ML overlay for the rules engine.

Trains a classifier per (index, action) target on historical rebalance
announcements. Models are intentionally simple: logistic regression for the
calibrated baseline, random forest for non-linear interactions, optional XGBoost
when installed. The validation strategy is strictly time-series.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

log = logging.getLogger(__name__)


DEFAULT_FEATURES = [
    "fmc_rank",
    "avg_float_market_cap",
    "market_cap_gap_to_cutoff",
    "relative_liquidity",
    "median_daily_value_traded",
    "ADV_20d",
    "ADV_60d",
    "return_21d",
    "return_63d",
    "volatility_63d",
    "current_member",
    "rank_change_1m",
    "rank_change_3m",
    "corporate_action_flag",
    "data_quality_score",
]


@dataclass
class TrainedModel:
    name: str
    pipeline: Pipeline
    features: list[str]
    metrics: dict[str, float]


def _build_features(df: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    out = df.copy()
    for col in features:
        if col not in out.columns:
            out[col] = 0.0
    out[features] = out[features].fillna(0.0).astype(float)
    return out


def train_classifier(panel: pd.DataFrame, label_col: str,
                     features: Iterable[str] = DEFAULT_FEATURES,
                     model: str = "logistic",
                     n_splits: int = 5) -> TrainedModel:
    feats = list(features)
    X = _build_features(panel, feats)[feats].values
    y = panel[label_col].astype(int).values

    if model == "random_forest":
        base = RandomForestClassifier(
            n_estimators=200, min_samples_leaf=5, random_state=0, n_jobs=-1,
            class_weight="balanced",
        )
    elif model == "xgboost":
        try:
            from xgboost import XGBClassifier  # type: ignore
        except ImportError as exc:  # pragma: no cover - optional dep
            raise RuntimeError("xgboost not installed; pip install xgboost") from exc
        base = XGBClassifier(
            n_estimators=300, max_depth=4, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, eval_metric="logloss",
            random_state=0, n_jobs=-1,
        )
    else:
        base = LogisticRegression(max_iter=2000, class_weight="balanced")

    pipeline = Pipeline([("scaler", StandardScaler()), ("clf", base)])

    # Cross-validated metrics, time-series only.
    metrics: dict[str, float] = {}
    if len(np.unique(y)) > 1:
        splits = TimeSeriesSplit(n_splits=n_splits)
        precisions, recalls, f1s, prauc = [], [], [], []
        for train_idx, test_idx in splits.split(X):
            if len(np.unique(y[train_idx])) < 2 or len(np.unique(y[test_idx])) < 2:
                continue
            pipeline.fit(X[train_idx], y[train_idx])
            proba = pipeline.predict_proba(X[test_idx])[:, 1]
            preds = (proba >= 0.5).astype(int)
            precisions.append(precision_score(y[test_idx], preds, zero_division=0))
            recalls.append(recall_score(y[test_idx], preds, zero_division=0))
            f1s.append(f1_score(y[test_idx], preds, zero_division=0))
            prauc.append(average_precision_score(y[test_idx], proba))
        metrics = {
            "precision_mean": float(np.mean(precisions)) if precisions else float("nan"),
            "recall_mean": float(np.mean(recalls)) if recalls else float("nan"),
            "f1_mean": float(np.mean(f1s)) if f1s else float("nan"),
            "pr_auc_mean": float(np.mean(prauc)) if prauc else float("nan"),
        }

    # Fit on the full panel for the trained artefact.
    pipeline.fit(X, y)
    proba_full = pipeline.predict_proba(X)[:, 1]
    preds_full = (proba_full >= 0.5).astype(int)
    cm = confusion_matrix(y, preds_full, labels=[0, 1]).tolist()
    metrics["confusion_matrix_train"] = cm
    metrics["top10_hit_rate_train"] = float(_top_k_hit_rate(y, proba_full, k=10))

    # Wrap in CalibratedClassifierCV for downstream probability use.
    if len(np.unique(y)) > 1:
        calibrated = CalibratedClassifierCV(pipeline, method="sigmoid", cv=3)
        calibrated.fit(X, y)
        return TrainedModel(name=model, pipeline=calibrated, features=feats, metrics=metrics)
    return TrainedModel(name=model, pipeline=pipeline, features=feats, metrics=metrics)


def _top_k_hit_rate(y: np.ndarray, proba: np.ndarray, k: int = 10) -> float:
    if len(proba) == 0:
        return float("nan")
    order = np.argsort(-proba)
    top = order[:k]
    return float(y[top].sum() / max(1, int(y.sum())))


def predict_proba(model: TrainedModel, panel: pd.DataFrame) -> np.ndarray:
    X = _build_features(panel, model.features)[model.features].values
    return model.pipeline.predict_proba(X)[:, 1]


def save_model(model: TrainedModel, path: Path) -> None:
    import pickle
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as fh:
        pickle.dump(model, fh)


def load_model(path: Path) -> TrainedModel:
    import pickle
    with path.open("rb") as fh:
        return pickle.load(fh)


def write_metrics_json(metrics: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(metrics, fh, indent=2, default=str)


def feature_importance(model: TrainedModel) -> pd.DataFrame:
    """Return a DataFrame of feature importances when supported."""
    pipe = model.pipeline
    final = getattr(pipe, "calibrated_classifiers_", None)
    estimator = pipe
    if final and len(final) > 0:
        estimator = final[0].estimator
    if isinstance(estimator, Pipeline):
        clf = estimator.named_steps["clf"]
    else:
        clf = estimator

    if hasattr(clf, "feature_importances_"):
        imp = clf.feature_importances_
    elif hasattr(clf, "coef_"):
        imp = np.abs(clf.coef_).ravel()
    else:
        imp = np.zeros(len(model.features))
    return (pd.DataFrame({"feature": model.features, "importance": imp})
              .sort_values("importance", ascending=False).reset_index(drop=True))


__all__ = [
    "DEFAULT_FEATURES",
    "TrainedModel",
    "train_classifier",
    "predict_proba",
    "save_model",
    "load_model",
    "write_metrics_json",
    "feature_importance",
]
