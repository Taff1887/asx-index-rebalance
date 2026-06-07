"""Rules-based shadow rebalance for the S&P/ASX indices.

For each (index, rebalance date) the engine:

1. Builds the eligible universe.
2. Computes the average float-adjusted market cap and liquidity over the
   methodology lookback.
3. Ranks by average float-adjusted market cap.
4. Applies the configured rank buffers per index.
5. Determines additions, deletions, promotions and demotions, respecting the
   nested ASX 50 / ASX 100 / ASX 200 hierarchy.

The output is a per-ticker decision frame with a confidence bucket.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import pandas as pd

from ..config import load_indices_config, load_methodology_config


CONFIDENCE_BUCKETS = ["high conviction", "medium conviction", "borderline", "no change"]


@dataclass(frozen=True)
class IndexParams:
    name: str
    target_count: int
    addition_buffer: int | None
    deletion_buffer: int | None
    parent_index: str | None


def get_index_params(index_name: str) -> IndexParams:
    cfg = load_indices_config()["indices"][index_name]
    return IndexParams(
        name=index_name,
        target_count=int(cfg["target_count"]),
        addition_buffer=cfg.get("addition_rank_buffer"),
        deletion_buffer=cfg.get("deletion_rank_buffer"),
        parent_index=cfg.get("parent_index"),
    )


def _confidence_bucket(rank: int, buffer_in: int | None, buffer_out: int | None,
                       liquidity_pass: bool) -> tuple[str, str]:
    """Return (predicted_action, confidence_bucket)."""
    if not liquidity_pass:
        return "no change", "no change"
    return "needs_outer_decision", "borderline"  # placeholder, refined below


def predict_rebalance(rank_panel: pd.DataFrame, index_name: str,
                      currently_in_index: set[str],
                      parent_membership: dict[str, str] | None = None) -> pd.DataFrame:
    """Predict additions/removals/promotions/demotions for `index_name`.

    Inputs:
        rank_panel: must contain ticker, avg_float_market_cap, liquidity_pass,
            relative_liquidity, fmc_rank (post-eligibility-filtering).
        currently_in_index: set of current constituent tickers.
        parent_membership: optional ticker -> parent-index-name for hierarchy
            handling (e.g. ASX200 events that also imply ASX100/ASX50 changes).
    """
    params = get_index_params(index_name)
    method = load_methodology_config()["rules_engine"]
    use_buffers = bool(method.get("use_rank_buffers", True))

    target = params.target_count
    add_buf = params.addition_buffer or target
    del_buf = params.deletion_buffer or target

    df = rank_panel.copy()
    df["index"] = index_name
    df["current_member"] = df["ticker"].isin(currently_in_index)
    df["rank_buffer_margin"] = df["fmc_rank"] - target

    actions = []
    confidences = []
    reasons = []

    for _, row in df.iterrows():
        rank = int(row["fmc_rank"])
        is_member = bool(row["current_member"])
        liq_pass = bool(row.get("liquidity_pass", True))

        if not liq_pass:
            if is_member:
                actions.append("Removal")
                confidences.append("high conviction")
                reasons.append("Liquidity fail among current constituents")
            else:
                actions.append("no change")
                confidences.append("no change")
                reasons.append("Liquidity fail among non-constituents")
            continue

        if use_buffers and is_member:
            if rank > del_buf:
                actions.append("Removal")
                confidences.append(
                    "high conviction" if rank > del_buf * 1.05 else "medium conviction"
                )
                reasons.append(f"Rank {rank} > deletion buffer {del_buf}")
            else:
                actions.append("no change")
                confidences.append("no change")
                reasons.append("Inside deletion buffer")
        elif use_buffers and not is_member:
            if rank <= add_buf:
                actions.append("Addition")
                confidences.append(
                    "high conviction" if rank <= add_buf * 0.95 else "medium conviction"
                )
                reasons.append(f"Rank {rank} <= addition buffer {add_buf}")
            else:
                actions.append("no change")
                confidences.append("no change")
                reasons.append("Outside addition buffer")
        else:
            # No buffers: fall back to top-N construction.
            if rank <= target:
                actions.append("no change" if is_member else "Addition")
                confidences.append("medium conviction" if not is_member else "no change")
                reasons.append("Inside top-N target")
            else:
                actions.append("Removal" if is_member else "no change")
                confidences.append("medium conviction" if is_member else "no change")
                reasons.append("Outside top-N target")

    df["predicted_action"] = actions
    df["confidence_bucket"] = confidences
    df["reason"] = reasons

    df = _pair_additions_and_deletions(df, target)
    df = _apply_index_hierarchy(df, parent_membership or {})

    df["predicted_member_after_rebalance"] = df.apply(_predicted_member, axis=1)
    return df.reset_index(drop=True)


def _pair_additions_and_deletions(df: pd.DataFrame, target: int) -> pd.DataFrame:
    """Pair buffer-driven additions and removals; liquidity-driven removals
    are always kept and backfilled with the highest-ranked eligible non-member.

    The pairing only trims when both sides have candidates — when only one
    side has signal, we leave the imbalance for the user to post-process,
    rather than silently zeroing out the engine's recommendation.
    """
    forced_mask = ((df["predicted_action"] == "Removal")
                    & df["reason"].str.startswith("Liquidity fail"))
    forced_removals = df[forced_mask]
    additions = df[df["predicted_action"] == "Addition"].sort_values("fmc_rank")

    # Backfill forced removals with the highest-ranked eligible non-members.
    needed = max(0, len(forced_removals) - len(additions))
    if needed and "current_member" in df.columns:
        candidate_pool = (df[(df["predicted_action"] == "no change")
                              & ~df["current_member"]
                              & df.get("liquidity_pass", True)]
                          .sort_values("fmc_rank").head(needed))
        df.loc[candidate_pool.index, "predicted_action"] = "Addition"
        df.loc[candidate_pool.index, "confidence_bucket"] = "medium conviction"
        df.loc[candidate_pool.index, "reason"] = (
            df.loc[candidate_pool.index, "reason"].astype(str)
            + "; promoted to backfill liquidity-driven removal"
        )

    # Re-read after backfill.
    additions = df[df["predicted_action"] == "Addition"].sort_values("fmc_rank")
    buffer_removals = df[(df["predicted_action"] == "Removal") & ~forced_mask]\
        .sort_values("fmc_rank", ascending=False)
    pair = min(len(additions), len(buffer_removals))

    if pair > 0:
        keep_add = set(additions.head(pair)["ticker"])
        keep_del = set(buffer_removals.head(pair)["ticker"]) | set(forced_removals["ticker"])
        for idx, row in df.iterrows():
            if row["predicted_action"] == "Addition" and row["ticker"] not in keep_add:
                df.at[idx, "predicted_action"] = "no change"
                df.at[idx, "confidence_bucket"] = "borderline"
                df.at[idx, "reason"] += "; trimmed to balance index count"
            if row["predicted_action"] == "Removal" and row["ticker"] not in keep_del:
                df.at[idx, "predicted_action"] = "no change"
                df.at[idx, "confidence_bucket"] = "borderline"
                df.at[idx, "reason"] += "; trimmed to balance index count"
    return df


def _apply_index_hierarchy(df: pd.DataFrame, parent_membership: dict[str, str]) -> pd.DataFrame:
    """Tag promotions/demotions relative to the parent index."""
    df["parent_index"] = df["ticker"].map(parent_membership)
    df["promotion_or_demotion_flag"] = False
    for i, row in df.iterrows():
        parent = row.get("parent_index")
        if parent is None:
            continue
        if row["predicted_action"] == "Addition" and parent == row["index"]:
            df.at[i, "predicted_action"] = "Promotion"
            df.at[i, "promotion_or_demotion_flag"] = True
        if row["predicted_action"] == "Removal" and parent == row["index"]:
            df.at[i, "predicted_action"] = "Demotion"
            df.at[i, "promotion_or_demotion_flag"] = True
    return df


def _predicted_member(row: pd.Series) -> bool:
    if row["predicted_action"] in ("Addition", "Promotion"):
        return True
    if row["predicted_action"] in ("Removal", "Demotion"):
        return False
    return bool(row["current_member"])


def run_rules_engine_for_all_indices(rank_panel: pd.DataFrame,
                                     constituents_by_index: dict[str, set[str]]) -> pd.DataFrame:
    """Run the rules engine across the primary ASX index family."""
    primary = load_indices_config()["primary_indices"]
    out_frames = []
    parent_membership: dict[str, str] = {}
    for index_name in primary:
        df = predict_rebalance(
            rank_panel, index_name,
            currently_in_index=constituents_by_index.get(index_name, set()),
            parent_membership=parent_membership,
        )
        out_frames.append(df)
        parent_membership.update({t: index_name for t in df["ticker"]})
    return pd.concat(out_frames, ignore_index=True)


__all__ = [
    "CONFIDENCE_BUCKETS",
    "IndexParams",
    "get_index_params",
    "predict_rebalance",
    "run_rules_engine_for_all_indices",
]
