"""Portfolio construction — turn signals into per-period weights."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import load_strategy_config


def size_positions(signals: pd.DataFrame) -> pd.DataFrame:
    """Assign weights to each signal according to `strategy.position_sizing`."""
    cfg = load_strategy_config()["strategy"]["position_sizing"]
    method = cfg.get("method", "equal_weight")
    cap = float(cfg.get("max_position_weight", 0.10))

    s = signals.copy()
    if s.empty:
        s["weight"] = []
        return s

    if method == "equal_weight":
        # Equal weight within each entry_date / side bucket.
        s["weight"] = s.groupby(["entry_date", "side"])["ticker"].transform(
            lambda g: 1.0 / max(1, len(g))
        )
    elif method == "confidence_weighted":
        s["weight"] = s.groupby(["entry_date", "side"])["hybrid_probability"].transform(
            lambda g: g / g.sum() if g.sum() else 0
        )
    elif method == "flow_weighted":
        flow = s.get("passive_flow_to_ADV_20d", pd.Series(0, index=s.index)).abs().fillna(0)
        s["weight"] = flow.groupby([s["entry_date"], s["side"]]).transform(
            lambda g: g / g.sum() if g.sum() else 0
        )
    elif method == "vol_scaled":
        vol = s.get("volatility_63d", pd.Series(0.2, index=s.index)).fillna(0.2)
        inv_vol = 1.0 / vol.replace(0, np.nan)
        s["weight"] = inv_vol.groupby([s["entry_date"], s["side"]]).transform(
            lambda g: g / g.sum() if g.sum() else 0
        )
    else:
        raise ValueError(f"Unknown position sizing method: {method}")

    s["weight"] = s["weight"].clip(upper=cap)
    s.loc[s["side"] == "short", "weight"] *= -1.0
    return s


def enforce_exposure(weights: pd.DataFrame) -> pd.DataFrame:
    """Cap gross and net exposure per `strategy.yaml::position_sizing`.

    The net-exposure cap only applies when both long and short sides are
    represented on the day. A one-sided book (long-only or short-only) is
    intentionally directional and would be crushed by a 20% net cap —
    `enforce_exposure` skips the net cap in that case and lets the gross
    cap do the heavy lifting.
    """
    cfg = load_strategy_config()["strategy"]["position_sizing"]
    gross_cap = float(cfg.get("max_gross_exposure", 1.0))
    net_cap = float(cfg.get("max_net_exposure", 0.20))

    out_frames = []
    for entry, grp in weights.groupby("entry_date"):
        g = grp.copy()
        gross = g["weight"].abs().sum()
        if gross > gross_cap and gross > 0:
            g["weight"] *= gross_cap / gross

        sides = set(g["side"].unique()) if "side" in g.columns else set()
        is_two_sided = ("long" in sides) and ("short" in sides)
        net = g["weight"].sum()
        if is_two_sided and abs(net) > net_cap and abs(net) > 0:
            shift = (abs(net) - net_cap) * np.sign(net)
            g["weight"] -= shift / len(g)
        out_frames.append(g)
    return pd.concat(out_frames, ignore_index=True) if out_frames else weights


def expand_to_positions(weights: pd.DataFrame) -> pd.DataFrame:
    """Materialise daily positions between each event's entry_date and exit_date."""
    out_rows = []
    for _, row in weights.iterrows():
        days = pd.bdate_range(row["entry_date"], row["exit_date"])
        for d in days:
            out_rows.append({
                "date": d,
                "ticker": row["ticker"],
                "side": row["side"],
                "weight": row["weight"],
                "rebalance_month": row.get("rebalance_month"),
                "variant": row.get("variant"),
                "index": row.get("index"),
                "announcement_date": row.get("announcement_date"),
                "effective_date": row.get("effective_date"),
            })
    return pd.DataFrame(out_rows)


__all__ = ["size_positions", "enforce_exposure", "expand_to_positions"]
