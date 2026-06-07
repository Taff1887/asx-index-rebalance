from __future__ import annotations

import pandas as pd

from asxrebalance.strategy.signals import apply_filters, build_event_signals, top_k_signals


def _events() -> pd.DataFrame:
    return pd.DataFrame({
        "ticker": ["A", "B", "C", "D"],
        "index": ["ASX200"] * 4,
        "predicted_action": ["Addition", "Removal", "Promotion", "Demotion"],
        "hybrid_probability": [0.9, 0.4, 0.8, 0.7],
        "passive_flow_to_ADV_20d": [1.0, 0.3, 0.6, 0.2],
        "confidence_bucket": ["high conviction", "borderline",
                              "medium conviction", "high conviction"],
        "announcement_date": pd.to_datetime(["2024-03-01"] * 4),
        "effective_date": pd.to_datetime(["2024-03-15"] * 4),
    })


def test_build_event_signals_long_short():
    s = build_event_signals(_events(), variant="announcement_long_short")
    assert {"long", "short"} <= set(s["side"].unique())


def test_apply_filters_min_prob():
    s = build_event_signals(_events())
    out = apply_filters(s, min_prob=0.5, min_flow_to_adv=0.0,
                        high_conviction_only=False)
    assert (out["hybrid_probability"] >= 0.5).all()


def test_top_k_keeps_top_two():
    s = build_event_signals(_events())
    out = top_k_signals(s, k=1)
    assert len(out) <= 2  # at most one per side
