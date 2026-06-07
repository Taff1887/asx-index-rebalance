"""Centralised filesystem paths.

Resolves to the repository root via the location of this file, which keeps tests
and the CLI insensitive to the current working directory.
"""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

CONFIG_DIR = REPO_ROOT / "config"

DATA_DIR = REPO_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
RAW_FMP_DIR = RAW_DIR / "fmp"
RAW_YAHOO_DIR = RAW_DIR / "yahoo"
RAW_MANUAL_DIR = RAW_DIR / "manual"

INTERIM_DIR = DATA_DIR / "interim"
INTERIM_FMP_CLEAN_DIR = INTERIM_DIR / "fmp_clean"
INTERIM_YAHOO_CLEAN_DIR = INTERIM_DIR / "yahoo_clean"
INTERIM_VALIDATION_DIR = INTERIM_DIR / "validation"

PROCESSED_DIR = DATA_DIR / "processed"
PROCESSED_RECONCILED_DIR = PROCESSED_DIR / "reconciled"
PROCESSED_LABELS_DIR = PROCESSED_DIR / "labels"
PROCESSED_FEATURES_DIR = PROCESSED_DIR / "features"
PROCESSED_BENCHMARK_DIR = PROCESSED_DIR / "benchmark"

OUTPUTS_DIR = REPO_ROOT / "outputs"
FIGURES_DIR = OUTPUTS_DIR / "figures"


def ensure_dirs() -> None:
    """Create the working directory tree if it does not already exist."""
    for d in (
        RAW_FMP_DIR,
        RAW_YAHOO_DIR,
        RAW_MANUAL_DIR,
        INTERIM_FMP_CLEAN_DIR,
        INTERIM_YAHOO_CLEAN_DIR,
        INTERIM_VALIDATION_DIR,
        PROCESSED_RECONCILED_DIR,
        PROCESSED_LABELS_DIR,
        PROCESSED_FEATURES_DIR,
        PROCESSED_BENCHMARK_DIR,
        OUTPUTS_DIR,
        FIGURES_DIR,
    ):
        d.mkdir(parents=True, exist_ok=True)


__all__ = [
    "REPO_ROOT",
    "CONFIG_DIR",
    "DATA_DIR",
    "RAW_DIR",
    "RAW_FMP_DIR",
    "RAW_YAHOO_DIR",
    "RAW_MANUAL_DIR",
    "INTERIM_DIR",
    "INTERIM_FMP_CLEAN_DIR",
    "INTERIM_YAHOO_CLEAN_DIR",
    "INTERIM_VALIDATION_DIR",
    "PROCESSED_DIR",
    "PROCESSED_RECONCILED_DIR",
    "PROCESSED_LABELS_DIR",
    "PROCESSED_FEATURES_DIR",
    "PROCESSED_BENCHMARK_DIR",
    "OUTPUTS_DIR",
    "FIGURES_DIR",
    "ensure_dirs",
]
