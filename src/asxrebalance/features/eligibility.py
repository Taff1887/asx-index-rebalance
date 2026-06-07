"""Eligibility filters."""

from __future__ import annotations

import pandas as pd

from ..config import load_methodology_config


def apply_eligibility(panel: pd.DataFrame) -> pd.DataFrame:
    """Filter the universe by methodology eligibility rules.

    The input panel must include `is_etf`, `is_lic`, `is_stapled`, `is_foreign_dual`
    boolean columns; missing columns are treated as ``False``.
    """
    rules = load_methodology_config()["eligibility"]
    out = panel.copy()
    for col in ("is_etf", "is_lic", "is_stapled", "is_foreign_dual"):
        if col not in out.columns:
            out[col] = False
    keep = pd.Series(True, index=out.index)
    if rules.get("exclude_etf", True):
        keep &= ~out["is_etf"].astype(bool)
    if rules.get("exclude_lic_lit", True):
        keep &= ~out["is_lic"].astype(bool)
    if rules.get("exclude_stapled_securities", False):
        keep &= ~out["is_stapled"].astype(bool)
    if rules.get("exclude_foreign_dual_listed", False):
        keep &= ~out["is_foreign_dual"].astype(bool)
    out["eligible"] = keep
    return out


__all__ = ["apply_eligibility"]
