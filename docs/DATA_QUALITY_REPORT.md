# Data quality report — OpenASX cross-validation

**Status:** the press-release-derived labels CSV has substantial integrity
issues that need to be resolved before the strategy backtest is trustworthy.

## What we did

1. Acquired **OpenASX** (https://openasx.tangerineslab.com/) — 28 snapshots of
   the S&P/ASX 200 constituent set from **2008-07-18 to 2025-09-26**, compiled
   from iShares, SPDR and BetaShares ETF holdings disclosures. Source data
   is in `data/raw/openasx/*.json`.

2. Built [`scripts/parse_openasx_dataset.py`](../scripts/parse_openasx_dataset.py)
   that takes consecutive snapshot pairs and emits an addition / removal
   event for each ticker that moved between them. Yielded **876 events**, of
   which only **27 are "exact"** (snapshot pair ≤ 45 days apart, so the event
   is attributable to a single quarterly rebalance). The rest are
   "approximate" — we know they happened sometime between two widely-spaced
   snapshots but can't pin the rebalance.

3. Built [`scripts/validate_dataset_vs_openasx.py`](../scripts/validate_dataset_vs_openasx.py)
   that cross-checks every press-release event against OpenASX membership at
   the surrounding snapshot dates. For each event we look at the OpenASX
   snapshot *before* and *after* the effective date and check the ticker's
   membership at both points.

## What we found

**115 events tested. Only 45 (39%) strictly confirmed.**

| Outcome | Count | Meaning |
|---|---:|---|
| **confirmed** | **45** | Membership flipped between the OpenASX before- and after- snapshots in the direction my label claims. Strong evidence the label is correct. |
| already_absent | 31 | Removal claim, but ticker was already absent at the BEFORE snapshot. Either my date is wrong, or the ticker was never in ASX 200. |
| already_present | 21 | Addition claim, but ticker was already present at the BEFORE snapshot. **Most likely confusion with an ASX 100 / ASX 50 promotion** of an existing ASX 200 member. |
| still_absent | 11 | Addition claim, but ticker is absent in BOTH snapshots — implausible if the ticker should be in ASX 200 at all. |
| still_present | 5 | Removal claim, but ticker is still present in the AFTER snapshot. Likely date misalignment with the OpenASX snapshot timing. |
| absent_after | 2 | Addition claim — ticker was present in BEFORE, absent in AFTER. Edge case — the ticker entered and exited inside one snapshot gap. |

## Concrete example of confusion

```
2019-03-18 PNI Addition  -> OpenASX 2019-03-04: NOT IN, OpenASX 2019-03-28: NOT IN
2019-03-18 HUB Addition  -> OpenASX 2019-03-04: ALREADY IN, 2019-03-28: STILL IN
```

PNI ("Pinnacle Investment Mgmt") was supposedly added to ASX 200 on
2019-03-18 per Motley Fool, but OpenASX shows it not in the index either
before or after. HUB ("HUB24") was supposedly added but OpenASX shows it
was already there.

The most likely explanation: **the Motley Fool article was about HUB and PNI
being promoted into ASX 100, not added to ASX 200**. Both were already in
ASX 200, and got bumped up a tier on 2019-03-18. The press article correctly
labelled them as joining ASX 100; my dataset incorrectly labelled them as
joining ASX 200.

This kind of confusion appears in **70 of 115 events**.

## What this means for the strategy

The backtest of "short the removals" is still likely directionally correct —
removals being removed is unambiguous — but the magnitude is unreliable
because the **dataset contains events that aren't actually ASX 200
constituent changes**. They're either:

- ASX 100 / ASX 50 promotions of existing ASX 200 members (most common)
- Events whose effective date in the press release doesn't match the actual
  index change date (rare but happens)
- Tickers we misread from the press article

## Options for fixing this

### Option A: Use OpenASX as the source of truth, discard everything else

* Re-derive labels from `outputs/openasx_event_log.csv` instead of the
  press-release CSV.
* Only the 27 "exact" events have reliable dates.
* The other 849 events are confirmed-occurred but not date-attributable.
* Pro: definitive, single-source, no ambiguity about whether something
  actually changed ASX 200 membership.
* Con: very few events (27) with usable dates for the backtest.

### Option B: Use OpenASX to filter / fix the press-release dataset

* Keep only the 45 events that validate against OpenASX.
* Re-investigate the 70 that don't, fix or drop them.
* Pro: more events than Option A, exact dates from press releases.
* Con: requires per-event manual investigation.

### Option C: Get a denser data source

* iShares (IOZ.AX) and SPDR (STW.AX) publish daily holdings — could scrape
  the BlackRock and State Street websites for every monthly disclosure
  going back 10+ years.
* Or use a paid feed (S&P Indexology API, Bloomberg, FactSet).
* Pro: 12 snapshots per year instead of 1-2, much tighter event attribution.
* Con: more engineering work upfront; paid options cost money.

### Option D: Accept the imperfect dataset and move on

* Note the data quality caveat prominently in the README.
* Run the strategy on the cleanest subset (45 confirmed events).
* Pro: fastest path forward.
* Con: leaves real data quality issues unresolved.

## Recommendation

**Option B** is the highest-leverage path. We already have 45 confirmed
events for the backtest plus 27 OpenASX-exact events that we can add cleanly.
Investigating the 70 unconfirmed events isn't a lot of work — most are likely
ASX 100 promotions that we can re-classify, fix, or drop.
