# Real S&P/ASX 200 rebalance dataset — sources

This file documents the public sources used to compile the real rebalance event
list in `scripts/fetch_real_rebalance_dataset.py`. Every row in
`data/processed/labels/rebalance_labels.csv` was verified against at least one
of these sources.

## Quarterly events

| Rebalance | Announced | Effective | Sources |
|---|---|---|---|
| March 2024 | 2024-03-01 | 2024-03-18 | [Stockhead](https://stockhead.com.au/news/asx-rebalance-tech-in-lithium-out-of-sp-asx-200-as-uranium-stocks-enter-sp-asx-300/), S&P press release |
| September 2024 | 2024-09-06 | 2024-09-23 | [Nasdaq](https://www.nasdaq.com/articles/sp-asx-indices-undergo-major-rebalance), [TipRanks](https://www.tipranks.com/news/company-announcements/sp-asx-indices-rebalance-set-for-september-2024) |
| December 2024 | 2024-12-06 | 2024-12-23 | [NZX announcement](https://www.nzx.com/announcements/443358) (Spark removal) |
| March 2025 | 2025-03-07 | 2025-03-24 | [investorpa](https://investorpa.com/announcement/111956/), S&P press release |
| June 2025 | 2025-06-06 | 2025-06-23 | [Financial Standard](https://www.financialstandard.com.au/news/asx-rebalance-sees-digico-exit-top-200-179811793) |
| September 2025 | 2025-09-05 | 2025-09-22 | [Livewire](https://www.livewiremarkets.com/wires/mark-your-calendar-the-2-3-billion-reshuffle-that-could-move-asx-stocks), [IG](https://www.ig.com/en-ch/news-and-trade-ideas/Macro-Intelligence-September-ASX-200-reshuffle-250911), [TipRanks](https://www.tipranks.com/news/company-announcements/sp-asx-indices-rebalance-announced-for-september-2025) |

## Coverage and limitations

- 40 events total — every confirmed S&P/ASX 200 addition / removal across six
  consecutive quarterly rebalances 2024-Q1 to 2025-Q3 that I could verify from
  publicly cited press releases. June 2024 had no S&P/ASX 200 changes per S&P,
  so it has zero rows.
- 37 of 40 tickers returned Yahoo Finance data. Three (`JLG`, `SPR`, `SYA`) had
  no `.AX` listing on Yahoo at fetch time — likely due to suspension, delisting
  or ticker renaming around the rebalance event. Those three events are
  recorded in the labels CSV but are skipped by the strategy backtest because
  no price series is available.
- Earlier rebalances (2018 - 2023) are not covered. The S&P press releases are
  still public but were not machine-readable at the URLs I tried; including
  them properly would require a paid feed (S&P Global Indexology, FactSet,
  Bloomberg) or manual transcription from S&P PDFs.
- ASX 50 and ASX 100 events: not yet captured in the labels CSV. Several of
  the ASX 200 changes are also implicit ASX 100 / ASX 50 promotions /
  demotions, but the per-tier breakdown is not always stated in the press
  excerpts. Adding ASX 50 / 100 coverage would also need paid data.

## How to add more

If you have a vendor feed or you've transcribed older S&P press releases:

1. Append rows to the `REAL_EVENTS` tuple in
   `scripts/fetch_real_rebalance_dataset.py`.
2. Re-run `python scripts/fetch_real_rebalance_dataset.py` — it pulls yfinance
   data for any new tickers and refreshes the labels CSV.
3. Re-run `python -m asxrebalance reconcile-data` then
   `python -m asxrebalance backtest-strategy ...` for all three variants.
4. Re-run `python scripts/build_six_way_comparison.py` to regenerate the
   bar chart and the six-way metrics table.

No code changes are required.
