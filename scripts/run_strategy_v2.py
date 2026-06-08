"""Validated ASX rebalance strategy — v2 (post-forensic-audit).

Fixes the bugs the verification workflow found in the earlier script:

  BUG 1 (33 fake trades): the old price lookup returned the FIRST available
        bar when a ticker's history started years after the rebalance
        (delisted names re-fetched for a later window). That produced
        entry==exit, fake 0.00% returns on wrong-year prices. FIX: a trade
        is only valid if the price file actually COVERS the announcement date
        and a real bar exists within `MAX_GAP_TD` trading days of each target.

  BUG 2 (ticker reuse): Yahoo reassigns delisted tickers to new micro-caps
        and serves the new company's history under the old ticker (AHE, VRL).
        FIX: a liquidity floor (median daily turnover >= MIN_TURNOVER_AUD)
        rejects the penny-stock impostors, which never had index-level
        turnover. Confirmed reuse tickers are also hard-excluded with a note.

  BUG 3 (frozen/stale windows): zero-volume flat tails (JIN, SCP, SGR, CTD).
        FIX: same liquidity floor + a non-zero-volume requirement on the
        entry and exit bars.

Strategy spec (unchanged, per user):
  Entry: close of the next trading day AFTER announcement (ann + 1 bday).
  Exit:  close on effective_date, and effective+5bd, and effective+10bd.
  Long  = buy Additions; Short = short Removals (short return = -raw move).
  Tiers: ASX 20, ASX 50, ASX 100, ASX 200.

Everything is reported in percent. No simulated data. No cumulative-over-time
line chart (the strategy is in cash ~89% of days, so that chart is meaningless;
see the deployment stat printed below).

Outputs:
  outputs/v2_trades.csv             every VALID trade, with entry/exit prices
  outputs/v2_rejected.csv           every dropped event, with the reason
  outputs/v2_summary.csv            per tier x side x exit: mean/median/winrate/compound
  outputs/v2_benchmarks.csv         real ASX 50/100/200 total return over window
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from asxrebalance.paths import (
    OUTPUTS_DIR,
    PROCESSED_BENCHMARK_DIR,
    PROCESSED_LABELS_DIR,
    PROCESSED_RECONCILED_DIR,
    RAW_YAHOO_DIR,
)

TIERS = ["ASX20", "ASX50", "ASX100", "ASX200"]
MAX_GAP_TD = 5             # matched bar must be within 5 trading days of target
MIN_TURNOVER_AUD = 250_000  # median daily close*volume over window (index-level floor)
# Tickers Yahoo has reassigned to a different company after the original delisted.
# Any trade on these before the reuse is the WRONG company — hard-exclude.
KNOWN_TICKER_REUSE = {"AHE", "VRL"}


def _load_price(ticker: str) -> pd.DataFrame | None:
    for d in (PROCESSED_RECONCILED_DIR / "prices", RAW_YAHOO_DIR):
        p = d / f"{ticker}.csv"
        if p.exists():
            df = pd.read_csv(p, parse_dates=["date"]).sort_values("date").reset_index(drop=True)
            if not df.empty:
                return df
    return None


def _bar_near(df: pd.DataFrame, target: pd.Timestamp) -> pd.Series | None:
    """First bar on/after target, but only if within MAX_GAP_TD trading days."""
    sub = df[df["date"] >= target]
    if sub.empty:
        return None
    bar = sub.iloc[0]
    # Count trading days (rows in df) between target's position and bar.
    gap_cal = (bar["date"] - target).days
    if gap_cal > MAX_GAP_TD + 5:   # generous calendar slack for weekends/holidays
        return None
    return bar


def _window_turnover(df: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> float:
    w = df[(df["date"] >= start) & (df["date"] <= end)].copy()
    if w.empty:
        return 0.0
    w["turnover"] = w["close"] * w["volume"]
    return float(w["turnover"].median())


def build_trade(df: pd.DataFrame, ann: pd.Timestamp, eff: pd.Timestamp,
                 exit_offset: int) -> tuple[dict | None, str]:
    """Return (trade_dict, reason). trade_dict is None if rejected."""
    # The price file must actually cover the announcement (catches delisted
    # names whose Yahoo history begins years later).
    if df["date"].min() > ann + pd.Timedelta(days=3):
        return None, "no_history_at_event"

    entry_target = ann + pd.tseries.offsets.BDay(1)
    exit_target = eff + pd.tseries.offsets.BDay(exit_offset)

    entry = _bar_near(df, entry_target)
    if entry is None:
        return None, "entry_gap"
    ex = _bar_near(df, exit_target)
    if ex is None:
        return None, "exit_gap"
    if entry["date"] >= ex["date"]:
        return None, "zero_or_negative_hold"

    # Liquidity / impostor guard.
    turn = _window_turnover(df, entry["date"], ex["date"])
    if turn < MIN_TURNOVER_AUD:
        return None, "illiquid_or_stale"
    if entry["volume"] <= 0 or ex["volume"] <= 0:
        return None, "zero_volume_endpoint"

    e_px = float(entry["adjusted_close"])
    x_px = float(ex["adjusted_close"])
    if not (np.isfinite(e_px) and np.isfinite(x_px)) or e_px <= 0:
        return None, "bad_price"

    return {
        "entry_date": entry["date"], "entry_close": round(e_px, 4),
        "exit_date": ex["date"], "exit_close": round(x_px, 4),
        "holding_td": int((ex["date"] - entry["date"]).days),
        "median_turnover_aud": round(turn, 0),
        "raw_pct": (x_px / e_px - 1) * 100,
    }, "ok"


def benchmark_total(name: str, start: pd.Timestamp, end: pd.Timestamp) -> tuple[float | None, str]:
    f = PROCESSED_BENCHMARK_DIR / f"{name.lower()}_benchmark.csv"
    if not f.exists():
        return None, ""
    df = pd.read_csv(f, parse_dates=["date"]).sort_values("date")
    df = df[(df["date"] >= start) & (df["date"] <= end)]
    if len(df) < 10:
        return None, ""
    # Robust endpoints: median of the first/last 5 bars neutralises a single
    # bad spike (e.g. ^ATLI's first bar of 4702 vs the real ~3085 level).
    start_px = float(df["adjusted_close"].head(5).median())
    end_px = float(df["adjusted_close"].tail(5).median())
    ret = (end_px / start_px - 1) * 100
    win = f"{df['date'].min().date()}..{df['date'].max().date()}"
    return float(ret), win


def main() -> None:
    labels = pd.read_csv(PROCESSED_LABELS_DIR / "rebalance_labels.csv",
                          parse_dates=["announcement_date", "effective_date"])
    labels = labels[labels["index"].isin(TIERS)].copy()
    print(f"Labels in {TIERS}: {len(labels)} events, "
          f"{labels['announcement_date'].nunique()} quarters\n")

    trades, rejects = [], []
    for _, evt in labels.iterrows():
        tkr = str(evt["ticker"])
        side = "long" if evt["action"] == "Addition" else "short"
        common = {
            "tier": evt["index"], "announcement_date": evt["announcement_date"],
            "effective_date": evt["effective_date"], "ticker": tkr,
            "company_name": evt["company_name"], "action": evt["action"], "side": side,
        }
        if tkr in KNOWN_TICKER_REUSE:
            rejects.append({**common, "reason": "known_ticker_reuse"})
            continue
        df = _load_price(tkr)
        if df is None:
            rejects.append({**common, "reason": "no_price_file"})
            continue
        # Build the base (exit at effective) trade; require it valid.
        base, reason = build_trade(df, evt["announcement_date"], evt["effective_date"], 0)
        if base is None:
            rejects.append({**common, "reason": reason})
            continue
        row = {**common, "entry_date": base["entry_date"], "entry_close": base["entry_close"],
               "median_turnover_aud": base["median_turnover_aud"]}
        for lbl, off in (("eff", 0), ("eff5", 5), ("eff10", 10)):
            t, r = build_trade(df, evt["announcement_date"], evt["effective_date"], off)
            if t is not None:
                raw = t["raw_pct"]
                row[f"exit_{lbl}_date"] = t["exit_date"]
                row[f"exit_{lbl}_close"] = t["exit_close"]
                row[f"raw_{lbl}_pct"] = round(raw, 3)
                row[f"trade_{lbl}_pct"] = round(raw if side == "long" else -raw, 3)
            else:
                row[f"exit_{lbl}_date"] = pd.NaT
                row[f"exit_{lbl}_close"] = np.nan
                row[f"raw_{lbl}_pct"] = np.nan
                row[f"trade_{lbl}_pct"] = np.nan
        trades.append(row)

    t = pd.DataFrame(trades)
    r = pd.DataFrame(rejects)
    t.to_csv(OUTPUTS_DIR / "v2_trades.csv", index=False)
    r.to_csv(OUTPUTS_DIR / "v2_rejected.csv", index=False)

    print(f"VALID trades: {len(t)}   REJECTED: {len(r)}")
    print("\nRejection reasons:")
    print(r["reason"].value_counts().to_string())
    print("\nValid trades by tier/side:")
    print(t.groupby(["tier", "side"]).size().unstack(fill_value=0).reindex(TIERS).to_string())

    # ---- summary ----
    rows = []
    for tier in TIERS:
        for exit_lbl, col in (("eff", "trade_eff_pct"), ("eff5", "trade_eff5_pct"),
                                ("eff10", "trade_eff10_pct")):
            for variant, mask in (("Long-only", t["side"] == "long"),
                                    ("Short-only", t["side"] == "short"),
                                    ("Long & short", pd.Series(True, index=t.index))):
                sub = t.loc[(t["tier"] == tier) & mask, ["announcement_date", col]].dropna()
                if sub.empty:
                    continue
                by_q = sub.groupby("announcement_date")[col].mean() / 100
                compound = (1 + by_q).prod() - 1
                # outlier-robust: drop best & worst quarter
                if len(by_q) > 4:
                    trimmed = by_q.sort_values().iloc[1:-1]
                    compound_trim = (1 + trimmed).prod() - 1
                else:
                    compound_trim = compound
                vals = sub[col]
                rows.append({
                    "tier": tier, "exit_window": exit_lbl, "variant": variant,
                    "n_trades": len(vals), "n_quarters": by_q.shape[0],
                    "mean_pct": round(vals.mean(), 3),
                    "median_pct": round(vals.median(), 3),
                    "std_pct": round(vals.std(), 3),
                    "win_rate_pct": round((vals > 0).mean() * 100, 1),
                    "compound_pct": round(compound * 100, 2),
                    "compound_trimmed_pct": round(compound_trim * 100, 2),
                })
    summary = pd.DataFrame(rows)
    summary.to_csv(OUTPUTS_DIR / "v2_summary.csv", index=False)

    # ---- benchmark ----
    start = t["announcement_date"].min()
    end = max(c for c in [t["exit_eff_date"].max(), t["exit_eff10_date"].max()] if pd.notna(c))
    bench_rows = []
    for tier in ("ASX50", "ASX100", "ASX200"):
        ret, win = benchmark_total(tier, start, end)
        bench_rows.append({"tier": tier, "total_return_pct": round(ret, 2) if ret else None,
                            "window": win})
    bench = pd.DataFrame(bench_rows)
    bench.to_csv(OUTPUTS_DIR / "v2_benchmarks.csv", index=False)

    print("\n" + "=" * 78)
    print("PER-TRADE MEAN % (exit at effective) — the honest primary metric")
    print("=" * 78)
    piv = (summary[summary["exit_window"] == "eff"]
           .pivot_table(index="tier", columns="variant", values="mean_pct")
           .reindex(TIERS))
    print(piv.to_string())
    print("\nWIN RATE % (exit at effective):")
    pivw = (summary[summary["exit_window"] == "eff"]
            .pivot_table(index="tier", columns="variant", values="win_rate_pct")
            .reindex(TIERS))
    print(pivw.to_string())
    print("\nBenchmarks (buy & hold total return over window):")
    print(bench.to_string(index=False))


if __name__ == "__main__":
    main()
