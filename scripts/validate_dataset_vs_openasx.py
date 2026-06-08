"""Cross-validate the press-release-sourced labels.csv against OpenASX membership.

For each event in rebalance_labels.csv, find the OpenASX snapshots immediately
before and after the effective date. The event is confirmed if:

    Addition: ticker is absent from the BEFORE snapshot and present in AFTER.
    Removal:  ticker is present in BEFORE snapshot and absent from AFTER.

If the OpenASX snapshots don't bracket the effective date (e.g. the event
falls outside the snapshot range), the event is "unverifiable".

Outputs:
    outputs/label_validation.csv  - one row per labelled event with the
                                     confirmation status.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from asxrebalance.paths import OUTPUTS_DIR, PROCESSED_LABELS_DIR, REPO_ROOT

OPENASX_DIR = REPO_ROOT / "data" / "raw" / "openasx"


def _normalise(t: str) -> str:
    s = (t or "").strip().upper()
    return s.split(".", 1)[0]


def main() -> None:
    snapshots = json.load((OPENASX_DIR / "snapshots.json").open(encoding="utf-8"))
    dates = sorted(snapshots.keys())
    members_by_date = {d: {_normalise(e["ticker"]) for e in snapshots[d]
                             if not e["ticker"].isdigit()} for d in dates}
    dates_dt = pd.to_datetime(dates)

    labels = pd.read_csv(PROCESSED_LABELS_DIR / "rebalance_labels.csv",
                          parse_dates=["announcement_date", "effective_date"])

    rows = []
    for _, evt in labels.iterrows():
        eff = evt["effective_date"]
        tkr = _normalise(str(evt["ticker"]))
        action = evt["action"]

        before = dates_dt[dates_dt < eff].max()
        after = dates_dt[dates_dt > eff].min()

        result = "unverifiable"
        before_present = None
        after_present = None

        if pd.notna(before) and pd.notna(after):
            before_d = before.strftime("%Y-%m-%d")
            after_d = after.strftime("%Y-%m-%d")
            before_present = tkr in members_by_date[before_d]
            after_present = tkr in members_by_date[after_d]
            if action in ("Addition", "Promotion"):
                if not before_present and after_present:
                    result = "confirmed"
                elif before_present and after_present:
                    result = "already_present"
                elif not before_present and not after_present:
                    result = "still_absent"
                else:
                    result = "absent_after"
            elif action in ("Removal", "Demotion"):
                if before_present and not after_present:
                    result = "confirmed"
                elif before_present and after_present:
                    result = "still_present"
                elif not before_present and not after_present:
                    result = "already_absent"
                else:
                    result = "absent_before"

        rows.append({
            "effective_date": evt["effective_date"],
            "ticker": tkr,
            "action": action,
            "openasx_before": before.strftime("%Y-%m-%d") if pd.notna(before) else "",
            "openasx_after": after.strftime("%Y-%m-%d") if pd.notna(after) else "",
            "in_before": before_present, "in_after": after_present,
            "validation": result,
        })

    out = pd.DataFrame(rows)
    path = OUTPUTS_DIR / "label_validation.csv"
    out.to_csv(path, index=False)

    print(f"Total events tested: {len(out)}\n")
    print("Validation outcomes:")
    print(out["validation"].value_counts().to_string())

    print("\nValidation rate by action:")
    print(out.groupby(["action", "validation"]).size().unstack(fill_value=0).to_string())

    not_confirmed = out[out["validation"] != "confirmed"]
    if not not_confirmed.empty:
        print(f"\n{len(not_confirmed)} events not strictly confirmed. Sample:")
        print(not_confirmed.head(15)[["effective_date", "ticker", "action",
                                       "openasx_before", "openasx_after",
                                       "in_before", "in_after", "validation"]]
              .to_string(index=False))

    print(f"\n-> {path}")


if __name__ == "__main__":
    main()
