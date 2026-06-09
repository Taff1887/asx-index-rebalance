"""Guard test: the real-data analysis must contain ONLY real data.

The repo once shipped a synthetic-data pipeline. This test ensures none of that
(or any other fabricated/substituted data) leaks into the directories the live
analysis actually reads. It scans every price/benchmark CSV for a 'synthetic'
source tag and fails if one is found. It skips quietly when the data dirs are
absent (e.g. a fresh checkout in CI before any fetch).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from asxrebalance.paths import (
    PROCESSED_BENCHMARK_DIR, PROCESSED_RECONCILED_DIR, RAW_FMP_DIR, RAW_YAHOO_DIR,
)

READ_DIRS = [
    RAW_YAHOO_DIR,
    RAW_FMP_DIR,
    PROCESSED_RECONCILED_DIR / "prices",
    PROCESSED_BENCHMARK_DIR,
]


def _all_csvs():
    for d in READ_DIRS:
        if d.exists():
            yield from d.glob("*.csv")


def test_no_synthetic_sourced_data():
    """No CSV the analysis reads may carry a 'synthetic' source tag."""
    offenders = []
    n_scanned = 0
    for p in _all_csvs():
        n_scanned += 1
        # the source tag lives in the header + first rows; reading a small head
        # is enough to catch any synthetic-sourced file.
        head = p.read_text(errors="ignore")[:2000].lower()
        if "synthetic" in head or "simulate" in head:
            offenders.append(p.name)
    if n_scanned == 0:
        pytest.skip("no data fetched yet — nothing to audit")
    assert not offenders, (
        f"Synthetic/substituted data found in the analysis dirs: {offenders[:20]} "
        f"(scanned {n_scanned} files). The study must use REAL data only."
    )


def test_no_synthetic_generator_script():
    """The fake-data generator must not be present (it wrote into the real dirs)."""
    repo = Path(__file__).resolve().parents[1]
    banned = ["scripts/generate_synthetic_data.py", "scripts/generate_report_assets.py"]
    present = [b for b in banned if (repo / b).exists()]
    assert not present, f"Fake-data generator scripts are back in the repo: {present}"
