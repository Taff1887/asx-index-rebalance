"""Parse the OpenASX historical ETF-holdings snapshots into a rebalance event log.

Source: https://openasx.tangerineslab.com (5 JSON files in data/raw/openasx/).
The site publishes 28 snapshots of S&P/ASX 200 constituents from 2008-07-18
to 2025-09-26, compiled from iShares (IOZ), SPDR (STW) and BetaShares ETF
holdings disclosures.

Approach
--------
For every pair of consecutive snapshots (t1, t2):
    additions  = (tickers in t2) - (tickers in t1)
    removals   = (tickers in t1) - (tickers in t2)

The strict membership-diff means events are EXACT for snapshot pairs that
bracket a single quarterly rebalance window (typically <30 days apart) and
APPROXIMATE for wider-spaced pairs (months or years between snapshots —
unable to attribute an event to a specific quarterly review without more
data, but the changes are nonetheless confirmed to have occurred between
the two snapshot dates).

Outputs:
    outputs/openasx_event_log.csv  - all events with bracket dates and
                                      confidence flag
    outputs/openasx_summary.txt    - one-line per snapshot pair
"""

from __future__ import annotations

import json
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

from asxrebalance.paths import OUTPUTS_DIR, REPO_ROOT

OPENASX_DIR = REPO_ROOT / "data" / "raw" / "openasx"


def _normalise_ticker(ticker: str) -> str | None:
    """Clean an OpenASX ticker. Early-era snapshots use numeric ISINs, drop those."""
    t = (ticker or "").strip().upper()
    if not t:
        return None
    if t.isdigit():
        return None
    if "." in t:
        t = t.split(".", 1)[0]
    if not t.isalnum() or len(t) > 6:
        return None
    return t


def main() -> None:
    snapshots = json.load((OPENASX_DIR / "snapshots.json").open(encoding="utf-8"))
    dates = sorted(snapshots.keys())
    print(f"Loaded {len(dates)} snapshots {dates[0]} -> {dates[-1]}\n")

    # Build constituent sets per date.
    members: dict[str, dict[str, dict]] = {}
    for d in dates:
        m: dict[str, dict] = {}
        for entry in snapshots[d]:
            tkr = _normalise_ticker(entry.get("ticker", ""))
            if not tkr:
                continue
            m[tkr] = entry
        members[d] = m
        print(f"  {d}: {len(m)} clean tickers from {len(snapshots[d])} raw entries")

    print(f"\n{'-' * 60}")
    print("Snapshot-pair differences:")
    print(f"{'-' * 60}")
    rows = []
    for i in range(1, len(dates)):
        t1, t2 = dates[i - 1], dates[i]
        d1, d2 = date.fromisoformat(t1), date.fromisoformat(t2)
        gap_days = (d2 - d1).days
        confidence = "exact" if gap_days <= 45 else "approximate"
        adds = sorted(set(members[t2]) - set(members[t1]))
        rems = sorted(set(members[t1]) - set(members[t2]))
        print(f"  {t1}  ->  {t2}   ({gap_days} d, {confidence})  "
              f"+{len(adds)} adds, -{len(rems)} removes")
        for tkr in adds:
            info = members[t2].get(tkr, {})
            rows.append({
                "snapshot_before": t1, "snapshot_after": t2,
                "gap_days": gap_days, "confidence": confidence,
                "ticker": tkr, "action": "Addition",
                "name": info.get("name", ""), "sector": info.get("sector", ""),
            })
        for tkr in rems:
            info = members[t1].get(tkr, {})
            rows.append({
                "snapshot_before": t1, "snapshot_after": t2,
                "gap_days": gap_days, "confidence": confidence,
                "ticker": tkr, "action": "Removal",
                "name": info.get("name", ""), "sector": info.get("sector", ""),
            })

    df = pd.DataFrame(rows)
    out_path = OUTPUTS_DIR / "openasx_event_log.csv"
    df.to_csv(out_path, index=False)

    print(f"\n{'-' * 60}")
    print(f"Total events: {len(df)}")
    print(f"  Exact      : {(df['confidence'] == 'exact').sum()}")
    print(f"  Approximate: {(df['confidence'] == 'approximate').sum()}")
    print(f"\nEvents per year (best estimate using snapshot_after date):")
    df["year"] = df["snapshot_after"].str[:4]
    print(df.groupby(["year", "confidence"]).size().unstack(fill_value=0).to_string())
    print(f"\n-> {out_path}")


if __name__ == "__main__":
    main()
