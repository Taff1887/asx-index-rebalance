"""Merge ALL ASX 200 index events (multi-index quarterly + dated off-cycle archive)
into one deduped master, then build the definitive price-coverage / missing-stock
list the user asked for.

Inputs:
    data/processed/labels/rebalance_labels.csv      (multi-index quarterly, all tiers)
    data/processed/labels/asx200_dated_events.csv   (dated ASX 200 archive)
Outputs:
    outputs/asx200_events_master.csv   deduped ASX 200 events, scheduled vs off-cycle
    outputs/missing_delisted.csv       every ticker with NO price series (the hunt list)
    outputs/ticker_coverage.csv        every ticker: events, has_price, source
"""

from __future__ import annotations

import pandas as pd

from asxrebalance.paths import (
    OUTPUTS_DIR, PROCESSED_LABELS_DIR, PROCESSED_RECONCILED_DIR, RAW_FMP_DIR, RAW_YAHOO_DIR,
)

OFF_CYCLE_TYPES = {"removal", "removal-replacement", "addition", "demerger",
                   "rebalance-replacement", "update"}


def main() -> None:
    multi = pd.read_csv(PROCESSED_LABELS_DIR / "rebalance_labels.csv",
                        parse_dates=["announcement_date", "effective_date"])
    multi = multi[multi["index"] == "ASX200"].copy()
    multi["source"] = "multi_quarterly"
    multi["source_type"] = "quarterly"
    multi["off_cycle"] = False

    dated = pd.read_csv(PROCESSED_LABELS_DIR / "asx200_dated_events.csv",
                        parse_dates=["announcement_date", "effective_date"])
    dated["source"] = "dated_archive"
    dated["off_cycle"] = dated["source_type"].isin(OFF_CYCLE_TYPES)

    cols = ["announcement_date", "effective_date", "index", "action", "ticker",
            "company_name", "source", "source_type", "off_cycle"]
    for c in cols:
        if c not in multi:
            multi[c] = pd.NA
        if c not in dated:
            dated[c] = pd.NA
    allev = pd.concat([multi[cols], dated[cols]], ignore_index=True)
    allev["ticker"] = allev["ticker"].astype(str).str.upper().str.strip()
    allev["action"] = allev["action"].astype(str).str.title()

    # dedup key: same change = same ticker + action + effective date (fallback announcement)
    allev["eff_key"] = allev["effective_date"].dt.date.astype(str)
    allev["ann_key"] = allev["announcement_date"].dt.date.astype(str)
    allev["key"] = allev["ticker"] + "|" + allev["action"] + "|" + allev["eff_key"]
    # prefer the multi_quarterly row when duplicated (richer company_name), else keep first
    allev["prio"] = (allev["source"] == "multi_quarterly").astype(int)
    allev = (allev.sort_values(["key", "prio"], ascending=[True, False])
                  .drop_duplicates(subset="key", keep="first"))
    # second-pass dedup for off-cycle vs quarterly same week (ticker+action within +/-7d)
    allev = allev.sort_values(["ticker", "action", "announcement_date"]).reset_index(drop=True)
    keep = []
    last = {}
    for i, r in allev.iterrows():
        k = (r["ticker"], r["action"])
        d = r["announcement_date"]
        if k in last and pd.notna(d) and pd.notna(last[k]) and abs((d - last[k]).days) <= 7:
            continue
        keep.append(i)
        last[k] = d
    master = allev.loc[keep].drop(columns=["eff_key", "ann_key", "key", "prio"]).reset_index(drop=True)
    master = master.sort_values("announcement_date").reset_index(drop=True)
    master.to_csv(OUTPUTS_DIR / "asx200_events_master.csv", index=False)

    # ---- price coverage ----
    yahoo = {p.stem for p in RAW_YAHOO_DIR.glob("*.csv")}
    recon = {p.stem for p in (PROCESSED_RECONCILED_DIR / "prices").glob("*.csv")}
    fmp = {p.stem for p in RAW_FMP_DIR.glob("*.csv")}
    have = yahoo | recon | fmp

    tickers = sorted(master["ticker"].unique())
    cov_rows, missing_rows = [], []
    for t in tickers:
        sub = master[master["ticker"] == t]
        has = t in have
        row = {"ticker": t, "n_events": len(sub),
               "actions": ",".join(sorted(sub["action"].unique())),
               "first_ann": sub["announcement_date"].min().date() if sub["announcement_date"].notna().any() else None,
               "last_ann": sub["announcement_date"].max().date() if sub["announcement_date"].notna().any() else None,
               "company_name": sub["company_name"].dropna().astype(str).replace("nan", "").max() if sub["company_name"].notna().any() else "",
               "in_yahoo": t in (yahoo | recon), "in_fmp": t in fmp, "has_price": has}
        cov_rows.append(row)
        if not has:
            missing_rows.append(row)
    cov = pd.DataFrame(cov_rows)
    cov.to_csv(OUTPUTS_DIR / "ticker_coverage.csv", index=False)
    missing = pd.DataFrame(missing_rows).sort_values("n_events", ascending=False)
    missing.to_csv(OUTPUTS_DIR / "missing_delisted.csv", index=False)

    print(f"MASTER ASX 200 events: {len(master)}  (was 337 quarterly-only)")
    print(f"  scheduled (quarterly): {(~master['off_cycle']).sum()}   off-cycle: {master['off_cycle'].sum()}")
    print(f"  additions: {(master.action=='Addition').sum()}   removals: {(master.action=='Removal').sum()}")
    print(f"  date span: {master['announcement_date'].min().date()} -> {master['announcement_date'].max().date()}")
    print(f"\nUnique tickers: {len(tickers)}")
    print(f"  with a price series: {cov['has_price'].sum()}")
    print(f"  MISSING (no price anywhere): {len(missing)}")
    print(f"\n--- MISSING DELISTED TICKERS (the hunt list, top 40 by #events) ---")
    print(missing[["ticker", "n_events", "actions", "first_ann", "last_ann", "company_name"]]
          .head(40).to_string(index=False))


if __name__ == "__main__":
    main()
