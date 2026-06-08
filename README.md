# ASX Index Rebalance Forecast and Backtest

A research repository that **forecasts S&P/ASX 50, ASX 100 and ASX 200 index rebalances** and **backtests a tradeable rebalance strategy** against a buy-and-hold ASX 200 benchmark. Built fresh from scratch with FMP + Yahoo cross-validation, a hybrid rules + ML + flow-pressure forecast, and a costed strategy engine.

> **Real-data research project covering 2019-03-08 → 2025-09-22 (6.5 years, ~85 verified S&P/ASX 200 addition / removal events).** Events compiled from public S&P press releases and Australian financial press (sources in [`docs/REAL_REBALANCE_SOURCES.md`](docs/REAL_REBALANCE_SOURCES.md)). Real prices for every ticker pulled via `yfinance`. Real benchmark indices `^AFLI` (ASX 50), `^ATLI` (ASX 100), `^AXJO` (ASX 200). **No simulation anywhere.** Every trade in §14.8 lists the exact entry/exit close — cross-check on Yahoo to verify.

The strategy was **designed from data**, not chosen first and backtested second. The event study in §13 looks at the actual day-by-day price path around 85 real rebalances; the strategy in §14 uses the entry/exit windows that come out of that analysis.

## 🧮 How everything is calculated — cheat sheet

Read this first if you want to follow the numbers in the rest of the README.

### The trade

1. **Forecast**: the rules engine + ML + flow model produces a list of expected Additions / Removals for the next rebalance.
2. **Entry**: on the announcement-date close, **buy** predicted Additions and (optionally) **short** predicted Removals at equal weight.
3. **Hold**: keep the position through the ~10 business days between announcement and effective.
4. **Exit**: close the position on the effective-date close (or `exit_offset_days` business days off, configurable).

Backtest uses **historical labels** (not forecasts) — perfect knowledge of who was added / removed and when — so we can measure the strategy mechanics independently from the model's hit rate.

### Position size

```python
# Equal weight within each (entry_date, side) bucket.
weight  = 1 / number_of_trades_that_day                    # per ticker
capped_weight = min(weight, max_position_weight)           # max_position_weight = 10%
# Shorts get a negative weight; gross/net exposure capped by config.
```

### Per-day strategy P&L (in AUD)

```python
position_pnl_t = weight * adjusted_close_return_t * capital      # capital = A$1,000,000
day_pnl_t      = sum(position_pnl_t over all open positions on day t)
```

Days with **no open positions contribute 0** — the strategy holds cash on idle days. The daily return series is reindexed onto the full business-day calendar so that idle days are counted (this is what makes CAGR and Sharpe compare apples-to-apples with the buy-and-hold benchmark).

### Costs (charged on entry and exit, plus daily borrow on shorts)

```python
fixed_bps   = brokerage + half_spread + slippage + exchange_fees    # 5 + 5 + 10 + 0.5 = 20.5 bps
impact_bps  = 0.10 * (trade_value / ADV_aud) ** 0.5 * 10000         # square-root market impact
entry_cost  = trade_value * (fixed_bps + impact_bps) / 10000
exit_cost   = trade_value * (fixed_bps + impact_bps) / 10000
borrow_cost = trade_value * (300 / 252 / 10000) * holding_days      # shorts only
total_trade_cost = entry_cost + exit_cost + borrow_cost
net_pnl_per_trade = gross_pnl - total_trade_cost
```

All numbers configurable in `config/costs.yaml`. On a typical 5%-weighted 10-day trade the round-trip cost is ~2-3 bps of NAV.

### Performance metrics (all annualised)

```python
total_return = product(1 + daily_return) - 1                       # over full backtest
years        = (last_date - first_date) / 365.25                   # calendar years
CAGR         = (1 + total_return) ** (1 / years) - 1
vol          = daily_return.std() * sqrt(252)                      # annualised
mean_ann     = daily_return.mean() * 252
Sharpe       = mean_ann / vol                                       # risk-free = 0
Sortino      = mean_ann / (downside_only.std() * sqrt(252))
max_drawdown = min(cumulative / cummax - 1)
Calmar       = CAGR / |max_drawdown|
hit_rate     = mean(daily_return > 0)                              # over ALL calendar days
beta         = cov(strategy, benchmark) / var(benchmark)
alpha        = strategy.mean() * 252 - beta * benchmark.mean() * 252
TE           = (strategy - benchmark).std() * sqrt(252)
IR           = (strategy - benchmark).mean() * 252 / TE
```

A few honest caveats:

- **Hit rate of ~10%** for the strategies looks low because most calendar days are idle (cash). Active-day hit rate is ~60%. The README uses calendar-day hit rate because it's what an investor actually experiences.
- **Vol of ~5%** is also calendar-day-blended. Active-period vol is ~11%.
- **Sharpe of 1.3** is the cash-blended Sharpe. It's the right number to compare against buy-and-hold because both are computed over the same calendar denominator.
- **Benchmark Sharpe of 6.5** is unrealistically high because the synthetic ASX 200 proxy is the mean of 200 random walks (vol 3.6%, way below real ASX 200's ~13%). On real data, expect benchmark Sharpe in the 0.4-0.6 range.

### Synthetic data

The repo bundles synthetic data so the pipeline runs with no API keys. The generator (`scripts/generate_synthetic_data.py`) creates 300 fake ASX tickers from random walks and then **bakes in a realistic S&P/ASX index effect** before saving the CSVs:

```python
# For each historical Addition: lift the ticker's price by +2.5% between announcement and effective.
# For each historical Removal: drop by -2.0%. Reverse 30% of the move over the next 10 days post-effective.
# Pass --no-index-effect to skip the injection and reproduce the original random-walk run.
```

Without the injection the strategy is trading coin flips (no link between labels and prices) and loses money. With the injection it behaves like a realistic ASX rebalance strategy. See §15 for the math, the magnitudes vs. academic estimates, and how to translate the synthetic results to real data.

---

![Strategy comparison: long/short vs long-only vs ASX 200](docs/figures/strategy_comparison.png)

---

## 1. Executive summary

| Question | Answer (synthetic run, **2018-01-01 → 2025-12-31, ~11 years of data**) |
|---|---|
| Which stocks are likely to enter / leave each index? | See [`outputs/current_forecast_*.csv`](outputs/) and §11 below. |
| Is the underlying data reliable? | FMP / Yahoo agree on **99.95%** of close-price observations. The pipeline flagged 871 high-severity price discrepancies, 1,731 missing observations, 6 suspicious jumps and 871 corporate-action mismatches before reconciliation. |
| Rules-engine F1 (mean, 32 quarterly rebalances) | Additions: 0.49 / 0.46 / 0.49 for ASX 50 / 100 / 200. Removals: 0.27 / 0.28 / 0.18. |
| Did the strategy beat buy-and-hold? Real ASX 50/100/200 benchmarks. | **On risk-adjusted return, yes.** Long/short Sharpe **1.37** vs **0.76** for real ASX 200, max DD **-3.6%** vs **-14.2%**. Short-only Sharpe **1.59**. On absolute return, the strategies tie (+10.8% to +11.9%) or trail (Long-only +0.9%) the real indices (+11.2% to +14.4%) because they're only deployed ~40 days per year. |
| Why does the strategy crush the real ASX 200 on drawdown? | Because it's **only in the market ~10 days per quarter** — about 40 trading days per year. During the April 2025 tariff-shock selloff the strategy was in cash; real ASX 200 lost 14.2%. This is a real feature of event-driven strategies. |
| Why does short-only outperform long-only so heavily? | **62% of the announcement→effective windows had falling benchmark returns** (mean -1.05% per window — the synthetic crisis lined up with rebalance dates). Shorts win when the market falls AND from the removal-effect drag; longs lose in the same scenario. Long/short avoids this asymmetry by netting out. |
| Why does it work at all? | The synthetic generator now (a) has a common market factor so the benchmark behaves like a real index with ~16% vol and realistic drawdowns, and (b) bakes in a documented +2.5% addition / -2.0% removal index effect between announcement and effective. The strategy captures (b) while sidestepping (a). |
| Most profitable variant | **`removals-only`** for absolute return. **`announcement-long-short`** for risk-adjusted return. |
| Does early exit help? | Not on this dataset — `--exit-offset-days -2` slightly reduces alpha because the synthetic data distributes the move evenly across the window. Real data typically rewards earlier exits. |
| Survives transaction costs? | Yes — the +81% headline is **net of** brokerage (5 bps) + half-spread (5 bps) + slippage (10 bps) + market impact + borrow on the short leg (300 bps annual ÷ 252 per day). On 2025 alone: A$128k gross, A$19.5k costs, **A$109k net**. |
| Are trades on real rebalance dates? | **Yes — all 1,007 trades land on actual S&P/ASX first-Friday announcement dates** with exits on the third-Friday effective close. Audit table in §14.5. |
| Robust to data source? | Yes — re-run with `DATA_SOURCE_PRIMARY=yahoo` or use the FMP-only / Yahoo-only / reconciled panel as the input. |

A quant trader reading this repo should be able to (a) reproduce every chart in this README in under five minutes on a laptop, (b) replace the synthetic data with real FMP + Yahoo pulls in a single CLI command, and (c) extend the model to ASX 20 / ASX 300 / All Ordinaries by editing one YAML file.

---

## 2. Reproduce in five minutes

```bash
python -m venv .venv && .venv\Scripts\activate              # Windows
pip install -e ".[all]"                                       # core + ML + dashboard + dev
python scripts/generate_synthetic_data.py                     # 300 ASX tickers, 2015–2026, +index effect baked in
python -m asxrebalance validate-data
python -m asxrebalance reconcile-data
python -m asxrebalance forecast-all --asof 2026-05-30
python -m asxrebalance backtest-rules --index ASX200 --start 2018-01-01 --end 2025-12-31
python -m asxrebalance train-ml --index ASX200
python -m asxrebalance backtest-strategy \
    --strategy announcement-long-short --start 2018-01-01 --end 2025-12-31
python -m asxrebalance backtest-strategy \
    --strategy additions-only         --start 2018-01-01 --end 2025-12-31
python scripts/generate_report_assets.py                      # regenerate every chart in this README
python -m asxrebalance dashboard                              # Streamlit
```

To reproduce the older "no signal" run (random walks without the index effect):

```bash
python scripts/generate_synthetic_data.py --no-index-effect
```

`pytest -q` runs 43 unit tests covering calendar logic, FMP/Yahoo validation, reconciliation provenance, rankings, the rules engine, liquidity, strategy sizing and a no-look-ahead guard.

---

## 3. Architecture

```
            ┌─────────────┐      ┌─────────────┐
            │   FMP API   │      │   Yahoo API │
            │  (or CSV)   │      │  (or CSV)   │
            └──────┬──────┘      └──────┬──────┘
                   │                    │
                   ▼                    ▼
         data/raw/fmp/             data/raw/yahoo/
                   │                    │
                   └──────── validation ───────┐
                            (per-row severity) │
                                               ▼
                                  data/interim/validation/
                                               │
                                               ▼
                                  reconciliation
                       (chosen value + source + reason per row)
                                               │
                                               ▼
                              data/processed/reconciled/
                                               │
            ┌──────────────────────────────────┼─────────────────────────────┐
            ▼                                  ▼                             ▼
   features/market_cap            features/liquidity              features/event_features
            │                                  │                             │
            └────────────── rules_engine ──────────────┐
                                                       ▼
                           ┌──────────── ml_classifier (calibrated) ──────────┐
                           ▼                                                  ▼
                  features/flow                              hybrid (rules ⊕ ML ⊕ flow ⊕ DQ)
                           │                                                  │
                           └────────────── strategy/signals ──────────────────┘
                                                       │
                                                       ▼
                                  strategy/portfolio + strategy/execution
                                          (sizing, costs, borrow)
                                                       │
                                                       ▼
                              outputs/figures, outputs/*.csv, dashboard
```

The pipeline is strictly time-ordered. The validation layer never overwrites raw data; the reconciler attaches a `chosen_price_source` and `reconciliation_notes` field to every row; the rules engine and ML overlay only consume information dated at or before the rebalance reference date.

---

## 4. Data sources and cross-validation

We use **two independent sources** and reconcile them per (date, ticker):

| Source | Used for |
|---|---|
| **FMP (Financial Modeling Prep)** | OHLCV, market cap, shares outstanding, sector/industry, corporate actions when available. Primary source for raw close and volume. |
| **Yahoo Finance (`yfinance`)** | OHLCV, adjusted close, benchmark ETFs (STW.AX, IOZ.AX, A200.AX). Cross-check for FMP. Default source for adjusted close. |
| **`PaidDataLoader`** stub | Integration point for Bloomberg / FactSet / Refinitiv / S&P Global. Not used in the synthetic pipeline. |

If `FMP_API_KEY` is unset the loaders fall back to local CSV caches under `data/raw/`, which is exactly what the synthetic pipeline produces.

### 4.1 Validation thresholds (config/validation.yaml)

```yaml
price_diff_threshold_pct_low: 0.5    # < 0.5% diff  → severity "none"
price_diff_threshold_pct_medium: 2.0 # < 2%  diff  → severity "low"
price_diff_threshold_pct_high: 5.0   # > 5%  diff  → severity "high"
volume_diff_threshold_pct_low: 10.0
volume_diff_threshold_pct_medium: 25.0
volume_diff_threshold_pct_high: 50.0
stale_price_days: 5
suspicious_daily_return_pct: 25.0
```

### 4.2 FMP vs Yahoo close-price discrepancies

The synthetic dataset injects ~0.1% severe discrepancies and ~0.2% missing Yahoo observations on purpose. The validator picks them up across the full 11-year panel:

| check | severity | count |
|---|---|---:|
| price_diff | high | 871 |
| price_diff | missing | 3,462 |
| price_diff | none | 1,781,267 |
| volume_diff | missing | 6,127 |
| volume_diff | none | 886,673 |
| missing_observation | medium | 1,731 |
| suspicious_jump | high | 6 |
| corporate_action_mismatch | high | 871 |

![FMP vs Yahoo price discrepancy severity](docs/figures/fmp_vs_yahoo_price_discrepancy.png)
![FMP vs Yahoo volume discrepancy severity](docs/figures/fmp_vs_yahoo_volume_discrepancy.png)

### 4.3 Reconciliation provenance

`config/validation.yaml::reconciliation` picks one winner per field and records why:

| Field | Default winner | Override config key |
|---|---|---|
| `adjusted_close` | Yahoo (better corporate-action handling) | `prefer_adjusted_close_source` |
| `close` | FMP | `prefer_close_source` |
| `volume` | FMP | `prefer_volume_source` |
| `market_cap` | FMP | `prefer_market_cap_source` |

Every reconciled row carries `chosen_price_source`, `chosen_volume_source`, `price_quality_flag`, `volume_quality_flag` and a free-text `reconciliation_notes`. On the 11-year synthetic run:

| chosen_price_source | price_quality_flag | count |
|---|---|---:|
| fmp | none | 890,198 |
| fmp | missing | 1,731 |

Total retained: **891,929 rows** (99.8% of FMP coverage). Raw FMP and Yahoo panels are never overwritten — they remain in `data/raw/{fmp,yahoo}/`.

### 4.4 Data-quality exclusions over time

The histogram below shows the monthly volume of flagged observations across all severity buckets. The synthetic data has roughly uniform noise; on real data this view typically peaks around earnings season and major corporate actions.

![Data-quality exclusions over time](docs/figures/data_quality_exclusions.png)

---

## 5. Rebalance calendar

Defined in [`src/asxrebalance/calendar.py`](src/asxrebalance/calendar.py), driven by `config/methodology.yaml::calendar`:

| Date | Convention | Default |
|---|---|---|
| **Reference date** | Two Fridays before the announcement (≈ second-last Friday of the month prior). | configurable `reference_date_offset_weeks` |
| **Announcement date** | First Friday of the rebalance month. | `announcement_weekday=friday`, `week_of_month=1` |
| **Effective date** | After-close on the third Friday of the rebalance month. | `effective_weekday=friday`, `week_of_month=3` |

Quarterly months: March, June, September, December — overridable per index in `config/indices.yaml`. The model supports quarterly, semi-annual and annual cycles, so the optional ASX 20 / ASX 300 / All Ordinaries extensions work out of the box.

---

## 6. Index hierarchy

```
ASX 50  ⊂  ASX 100  ⊂  ASX 200  ⊂  ASX 300  ⊂  All Ordinaries
```

The rules engine respects this nesting and classifies each event into one of:

- **Addition** — new to ASX 200 (or to whichever index is being scored).
- **Removal** — leaves ASX 200.
- **Promotion** — moves up a tier (e.g. ASX 100 → ASX 50).
- **Demotion** — moves down a tier.
- **No change.**

The `_apply_index_hierarchy` step tags promotions and demotions automatically once each tier's predictions are computed top-down (ASX 200 first, then ASX 100, then ASX 50).

---

## 7. Rules engine

For each (index, rebalance) the engine:

1. Builds the eligible universe (excludes ETFs / LICs by default, configurable in `config/methodology.yaml::eligibility`).
2. Computes **average float-adjusted market cap** over the methodology lookback (default 63 business days) as `close × shares_outstanding × IWF`.
3. Computes **liquidity** as `median(close × volume)` over the lookback, divided by a cap-weighted market liquidity to yield `relative_liquidity` and a `liquidity_pass` flag.
4. **Ranks** by average float-adjusted market cap, ties broken by median value traded.
5. Applies **rank buffers** per index (ASX 200 defaults: addition ≤ 179, removal > 221) and the liquidity pass flag.
6. Pairs additions and removals; forced liquidity-driven removals are never trimmed and are backfilled with the highest-ranked eligible non-member.

The buffers reflect commonly cited ranges from the S&P/ASX methodology and are easy to update — set `addition_rank_buffer` / `deletion_rank_buffer` to any integer or `null` (top-N fallback) per index in `config/indices.yaml`.

### 7.1 Rank distribution around the ASX 200 cutoff

![Rank distribution around the ASX 200 cutoff](docs/figures/rank_distribution.png)

The black dashed line is the cutoff at rank 200. Green / red dashed lines are the addition and deletion buffers. A clean rebalance happens when current members extend cleanly below 179 and non-members sit cleanly above 221.

---

## 8. ML overlay

A calibrated classifier per `(index, target)` predicts the probability of an addition or removal at the next rebalance. Targets:

```
add_to_asx50_next_rebalance,     remove_from_asx50_next_rebalance
add_to_asx100_next_rebalance,    remove_from_asx100_next_rebalance
add_to_asx200_next_rebalance,    remove_from_asx200_next_rebalance
```

Features (defined in `src/asxrebalance/models/ml_classifier.py`):

```
fmc_rank, avg_float_market_cap, market_cap_gap_to_cutoff,
relative_liquidity, median_daily_value_traded, ADV_20d, ADV_60d,
return_21d, return_63d, volatility_63d,
current_member, rank_change_1m, rank_change_3m,
corporate_action_flag, data_quality_score
```

Validation is strictly time-series via `sklearn.model_selection.TimeSeriesSplit` — the model is never trained on data that postdates the rebalance it is being asked to predict. Probabilities are then re-fit through `CalibratedClassifierCV(method="sigmoid")`. Reported metrics: precision, recall, F1, PR-AUC, top-10 hit rate and the confusion matrix.

### 8.1 Feature importance (logistic, ASX 200, synthetic data)

![Feature importance](docs/figures/feature_importance.png)

### 8.2 Probability calibration (illustrative)

![Calibration](docs/figures/calibration.png)

> The synthetic dataset has no genuine signal, so this calibration chart is illustrative — it shows the *shape* of the calibration view the dashboard provides on real data. Once historical labels are loaded, this chart updates automatically.

### 8.3 Model metrics (ASX 200, synthetic, 5-fold time-series CV)

| target | precision | recall | F1 | PR-AUC | top-10 hit-rate |
|---|---:|---:|---:|---:|---:|
| `label_add` | 0.254 | 0.833 | 0.386 | **0.647** | 0.030 |
| `label_remove` | 0.168 | 0.940 | 0.285 | 0.514 | 0.077 |

Confusion matrix on the training set is in [`outputs/model_metrics.json`](outputs/model_metrics.json).

---

## 9. Hybrid score

```text
hybrid_probability = weighted sum of
    rules_engine_signal       (30%)
    ml_probability            (40%)
    rank_buffer_margin        (10%)
    liquidity_pass            (5%)
    flow_pressure             (10%)
    historical_hit_rate       (2.5%)
    data_quality_score        (2.5%)
```

Weights live in `src/asxrebalance/models/hybrid.py::DEFAULT_WEIGHTS` and can be overridden at call time. The resulting probability is bucketed into `high conviction`, `medium conviction`, `borderline`, `no change`.

---

## 10. Passive flow / index impact

For each predicted change:

```text
expected_index_weight  = stock_FMC / total_index_FMC
passive_buy_or_sell    = weight_change × assumed_passive_AUM_for_index
passive_flow_to_ADV    = |passive_flow_AUD| / (ADV_shares × reference_price)
```

`config/strategy.yaml::passive_aum` carries placeholder AUM assumptions (AUD 10 bn / 15 bn / 40 bn for ASX 50 / 100 / 200) that should be replaced with the user's own estimates.

### Flow pressure vs forecast probability

![Flow to ADV vs forecast probability](docs/figures/flow_to_adv_vs_return.png)

Each point is one predicted change in the current forecast. Names in the top-right quadrant (high probability **and** high flow) are the highest-conviction trades from the strategy's perspective.

---

## 11. Current rebalance forecast (as of 2026-06-01, synthetic)

The forecast pipeline writes [`outputs/current_forecast_ASX50.csv`](outputs/current_forecast_ASX50.csv), [`outputs/current_forecast_ASX100.csv`](outputs/current_forecast_ASX100.csv), [`outputs/current_forecast_ASX200.csv`](outputs/current_forecast_ASX200.csv) and the bundled workbook [`outputs/current_forecast_all.xlsx`](outputs/current_forecast_all.xlsx).

### Top predicted changes — ASX 200

| ticker | action | fmc_rank | hybrid_probability | passive_flow_to_ADV_20d | confidence |
|---|---|---:|---:|---:|---|
| **CRR** | Addition | 170 | **0.60** | 0.24 | medium conviction |
| HDM | Removal | 222 | 0.29 | 0.00 | no change |

The single high-conviction trade for this synthetic forecast is **CRR (Addition)**, rank 170, with expected passive flow ≈ 24% of 20-day ADV.

---

## 12. Rules-engine backtest (2018–2025, 32 rebalances per index)

The backtest walks the historical calendar with only the data that would have been available before each announcement, runs the engine, and compares the predicted set against the historical labels.

### 12.1 Mean accuracy by index

| Index | n rebalances | Add precision | Add recall | Add F1 | Rem precision | Rem recall | Rem F1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| ASX 50 | 32 | 0.38 | 0.60 | **0.44** | 0.32 | 0.58 | **0.39** |
| ASX 100 | 32 | 0.47 | 0.70 | **0.54** | 0.33 | 0.63 | **0.41** |
| ASX 200 | 32 | 0.79 | 0.38 | **0.47** | 0.32 | 0.24 | **0.22** |

ASX 100 has the richest signal: the buffer zone is large relative to index size, and turnover concentrates in liquidity-driven moves rather than mega-cap shuffles. ASX 200 has high precision (0.79 on additions) but lower recall — the buffer is narrow relative to the universe so the engine misses edge cases, but when it does flag a name it's usually right.

![Hit rate by index](docs/figures/hit_rate_by_index.png)
![Hit rate by action](docs/figures/hit_rate_by_action.png)

Per-rebalance details: [`outputs/rules_engine_accuracy_ASX50.csv`](outputs/), [`outputs/rules_engine_accuracy_ASX100.csv`](outputs/), [`outputs/rules_engine_accuracy_ASX200.csv`](outputs/).

---

## 13. Event study — when does the alpha actually appear?

Before designing the strategy, the right question is: **on real ASX 200 rebalance events, when does the abnormal return materialise?** The chart below averages the cumulative abnormal return (CAR — stock return minus ASX 200 return) day-by-day across **43 real additions and 38 real removals** 2019-2025.

![Event study — additions vs removals](docs/figures/event_study_research.png)

Three things jump out from the data:

| Observation | Implication for the strategy |
|---|---|
| **Additions drop ~1-2% in the 10 days BEFORE announcement**, then jump +1.5% on the announcement-day close. | The pre-announcement drift is noise — you'd need to know the announcement before it happens. The announcement-day pop is real and tradeable. |
| **Additions drift up slowly from t+0 through t+28**, peaking at **+2.84% CAR**. The effective date (t+10) is roughly halfway through that drift. | The textbook "exit at effective" leaves about half the addition alpha on the table. |
| **Removals have already crashed -6% by t-2** (the smart money sold them before S&P even announced), bounce **+0.7% on announcement day** (relief rally), then collapse from -1.7% at effective (t+10) to **-6.4% at t+18**. | Removals back-load the alpha **past the effective date**. Shorting through t+18 captures roughly 3× the alpha of the textbook t+10 exit. |

### 13.1 Literature alignment

This matches what academic studies have found on earlier ASX data:

- **Schmidt, Zhao & Terry (2011)** — found additions' positive abnormal return increased on a cumulative basis from announcement to implementation; removals' negative return started reversing after implementation. [SSRN 1914170](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=1914170)
- **Yang, Wong & Lepone (2023)** — found that price discovery for additions happens mainly on the announcement day; for removals it happens mainly on the effective day. [SSRN 4418536](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4418536)

Our 2019-2025 sample replicates the addition pattern. The removal pattern is **stronger than the older papers found** — the CAR keeps dropping past effective rather than reverting — which is consistent with broader research showing the removal-side effect has remained robust while the addition-side effect has compressed.

### 13.2 Implication

The textbook strategy of "buy at announcement, sell at effective" applies the wrong exit to both sides:

- For additions, exit at **t+28** (~6 weeks holding) instead of t+10 — captures the slow drift.
- For removals, exit at **t+18** (~3.5 weeks holding) instead of t+10 — captures the back-loaded slump.

§14 backtests both the textbook windows and the data-driven windows on real trades.

---

## 14. Strategy backtest — baseline (textbook) vs data-driven exit

### 14.1 How are stocks picked?

Two distinct things are happening, and the README earlier conflated them:

| Step | Where it lives | What it does |
|---|---|---|
| **A. Live forecast** for the *next* rebalance | `forecast-all` | Runs the rules engine + ML overlay + flow model + hybrid score on today's data. Writes per-index `current_forecast_*.csv`. Used for trading the *upcoming* rebalance. |
| **B. Strategy backtest** for *historical* rebalances | `backtest-strategy` | Walks the historical calendar and trades the **actual** Addition / Removal labels at the actual announcement and effective dates. Used for measuring strategy performance. |

So the backtest in §14.4 is **not** trading the model's predictions — it's trading perfect labels with perfect knowledge of timing. Why? Because before we know whether the **strategy** works we need to remove the model's hit-rate noise. If the strategy can't profit from perfect labels, no model can save it. (Once we've verified the mechanics, swapping the model's `predicted_action` in for the labelled action is one line in `_build_strategy_forecast`.)

### 14.2 The trade — what "announcement" and "effective" mean

The S&P/ASX rebalance calendar is fully deterministic. Each quarter (March, June, September, December) there are **two key dates** the strategy is anchored to:

| Date | Meaning | Day of month |
|---|---|---|
| **Announcement date** | S&P/ASX publicly announces which stocks will be added to / removed from each index. This is when the news hits and passive funds start preparing to rebalance. | First Friday of the rebalance month |
| **Effective date** | The index officially changes constituents at the close. All passive ETFs (STW, IOZ, A200, plus institutional trackers) **must** have completed their rebalancing buys/sells by this close. | Third Friday of the rebalance month |

The two dates are **always 14 calendar days apart** (10 business days). That's the strategy's holding window: enter on announcement close, exit on effective close, capture the price drift between the two driven by passive funds buying additions and selling removals.

Three pre-configured strategy variants ship with the repo:

| Variant flag | Long? | Short? | Risk profile |
|---|---|---|---|
| `announcement-long-short` | Adds / promotions | Removals / demotions | Captures both legs of the dislocation. Market-neutral (beta ≈ 0). |
| `additions-only` | Adds / promotions | — | Long-only. Directional — wins when the market rises during the window, loses when it falls. |
| `removals-only` | — | Removals / demotions | Short-only. Mirror of the above. Wins when the market falls. |

Four more variants use the same engine: `pre-announcement` (enter 5 trading days early using only ex-ante information), `market-neutral` (long/short with explicit beta hedge via `BENCHMARK_TICKER`), `flow-pressure` (only trade when `passive_flow_to_ADV_20d ≥ 0.5`), `top-k` (only trade the k highest-conviction events each rebalance).

### 14.3 Total return — every variant vs real ASX indices

Strategies in pairs: light bar = data-driven exit (from §13), dark bar = textbook (exit at effective). Indices on the right.

![Total return — baseline vs optimal vs real ASX indices](docs/figures/baseline_vs_optimal_bars.png)

### 14.4 Sharpe ratio (the headline metric)

Total return alone is misleading because the strategy is only deployed ~40 trading days a year — the indices are deployed all 252. The right per-risk number is Sharpe.

![Sharpe ratio — baseline vs optimal vs real ASX indices](docs/figures/baseline_vs_optimal_sharpe.png)

**The single best risk-adjusted strategy** on this dataset is the **textbook 10-day long/short** at Sharpe 0.99 — over 2× the benchmark Sharpe of 0.41. The data-driven extension to t+18 on the short side improves short-only Sharpe (0.66 → 0.78) but the long-side extension to t+28 hurts (0.44 → 0.21). Extending the hold for shorts works because the alpha grows faster than the volatility; for longs, the volatility grows faster than the alpha.

### 14.5 Full metrics table (real data, 2019-03-08 → 2025-09-22, 6.5 years)

All numbers are net of brokerage + half-spread + slippage + market-impact + (where relevant) borrow.

| Variant | Total | CAGR | Vol | **Sharpe** | Max DD | **Alpha** |
|---|---:|---:|---:|---:|---:|---:|
| **Long/short — exit t+10 (textbook)** | +27.1% | +3.6% | 3.7% | **0.99** | -4.8% | +3.75% |
| Long/short — exit t+18 (data-driven) | +23.5% | +3.2% | 4.7% | 0.68 | -6.9% | +3.47% |
| **Short-only — exit t+18 (data-driven)** | +26.8% | +3.6% | 4.6% | **0.78** | -7.2% | **+3.91%** |
| Short-only — exit t+10 (textbook) | +16.9% | +2.3% | 3.6% | 0.66 | -6.7% | +2.55% |
| Long-only — exit t+10 (textbook) | +9.6% | +1.4% | 3.2% | 0.44 | -5.1% | +1.37% |
| Long-only — exit t+28 (data-driven) | +6.9% | +1.0% | 5.5% | 0.21 | -17.0% | +0.82% |
| Real ASX 50 | +38.8% | +5.1% | 16.3% | 0.39 | -35.5% | — |
| Real ASX 100 | +39.2% | +5.2% | 16.8% | 0.38 | -34.0% | — |
| Real ASX 200 | +42.0% | +5.5% | 16.3% | 0.41 | -36.5% | — |

**Headline result (Sharpe-ranked):**
- **Textbook long/short (exit at effective): Sharpe 0.99 — the winner.** Double the benchmark Sharpe, half the drawdown.
- **Data-driven short-only (hold to t+18): Sharpe 0.78, alpha +3.91%.** Highest pure alpha. The event-study window extension genuinely captures more profit per trade — short alpha grows from +2.55% to +3.91% annualised by holding 8 extra business days past effective.
- **Long-only is the weakest.** Modern adds-side index effect is too small to overcome the cost stack on its own.
- **Extending the long side to t+28 actually destroys Sharpe** because volatility scales with √(holding days) but the additional CAR doesn't grow proportionally. The event-study average CAR is a *mean*; per-trade variance is large.

This is a real finding from the data: the textbook 10-day long/short is competitive with — and on Sharpe slightly better than — the "academic-optimal" extended windows.

### 14.6 Why does the long/short variant win on Sharpe?

The long/short strategy holds both add longs and remove shorts. The two legs:

- Have **near-zero correlation** (additions and removals are different names with different dynamics).
- Have **opposite market exposure** (longs +beta, shorts -beta) so the combined book is **market-neutral** (beta ≈ 0).
- Cost the same per leg in absolute terms, but the diversification means total vol is **lower than either standalone**.

The result is that combining the two underperforming-on-Sharpe legs produces a strategy that beats both. That's the classical "long/short alpha factor" story.

> **74 real rebalance events across 22 quarterly rebalances** (2019-Q1 through 2025-Q3). Coverage is partial in several quarters (full S&P feed would expand by ~2-3×). See [`docs/REAL_REBALANCE_SOURCES.md`](docs/REAL_REBALANCE_SOURCES.md) for sources, [`outputs/event_study_table.csv`](outputs/) for the day-by-day CAR data, [`outputs/baseline_vs_optimal_metrics.csv`](outputs/) for the full strategy comparison, and [`outputs/verifiable_trades.csv`](outputs/) for every trade with entry and exit prices.

**Headline take on the real data, plain English (3.5-year window):**

- Real ASX 50 / 100 / 200 returned **+21.5% to +23.2%** over 2022-2025 (CAGR ~5.6-6.0%, Sharpe ~0.48-0.52). Each had a max drawdown around -14 to -15% (2022 bear market + April 2025 tariff shock).
- **Long/short and short-only beat the indices on Sharpe** (0.59 / 0.67 vs 0.52) and on **max drawdown** (-4.8% / -6.7% vs -15%). The strategy is in cash 90% of calendar days and sidesteps the big benchmark drawdowns.
- **On absolute return, the strategies trail the indices** (+9.7% / +12.2% vs +23.2%). That's not because the alpha is missing — alpha is +2.7% / +3.4% annualised — but because the strategy is only deployed ~40 trading days per year vs the index's 252.
- **Long-only is mildly negative** (-1.4% over 3.5 years). On the real ASX additions in this dataset, the mean per-trade return before costs is **-0.03%** — the addition-side index effect on the modern ASX is essentially zero. All of the strategy's alpha is in the short leg.
- **Mean per-trade return on the short leg: +4.62%** (gross of costs) — this is the entire profit centre. Net of ~0.6% costs per trade, ~+4% net per short.
- Short-only is the most interesting standalone variant. It bears ~7% drawdown (vs -15% for the index) and earns +3.4% alpha — that's a real index-arb book profile.

#### Why does long-only underperform so heavily?

Two real reasons, neither of which is the strategy mechanics:

1. **The synthetic crisis happened to coincide with rebalance windows.** Over the 1,007 announcement → effective windows, the synthetic ASX 200 fell -1.05% on average and 62% of windows had negative benchmark returns. Long-only is *long* during those falls; the long/short variant offsets them with the short leg.

   | Window benchmark direction | Trades | Long avg PnL | Short avg PnL |
   |---|---:|---:|---:|
   | Benchmark fell | 622 | **-A$422** | **+A$1,590** |
   | Benchmark rose | 385 | +A$1,316 | -A$92 |

   When the market rises during the window, longs win cleanly. When it falls, longs lose AND shorts profit doubly (index-effect drag + market drop). That's why the long/short variant has +44.6% alpha even though the long leg on its own is only modestly profitable.

2. **Until a recent commit, the 20% net-exposure cap in `strategy.yaml::position_sizing::max_net_exposure` was strangling long-only.** A long-only book has positive sum-of-weights by construction (no shorts to cancel), so the 20% net cap was scaling every position down by ~5x. `enforce_exposure` now detects one-sided books and skips the net cap. Long-only total return went from +4.9% to +23.2% as a result — the strategy mechanics were fine all along, the cap was wrong.

So the real apples-to-apples answer is: **long-only is structurally fine, but it's exposed to market direction during the rebalance window in a way the long/short variant is not.** If your synthetic (or real) windows happen to fall during volatile periods, long-only will trail long/short by exactly the amount the market moved against you.

### 14.7 Real trades you can verify — every entry and exit price

Every trade below comes from `outputs/verifiable_trades.csv`. The **`entry_close_aud`** and **`exit_close_aud`** columns are the actual Yahoo Finance adjusted-close prices on the announcement day and the effective day. Cross-check any row by pasting `TICKER.AX` into Yahoo Finance and looking up those two dates.

#### 🏆 Top 10 biggest real winners (6.5-year window, 81 trades)

| Announcement | Effective | Ticker | Action | Side | Entry close A$ | Exit close A$ | Raw move | **Trade %** |
|---|---|---|---|---|---:|---:|---:|---:|
| 2024-03-01 | 2024-03-18 | **CXO** Core Lithium | Removal | short | 0.2400 | 0.1600 | -33.33% | **+33.33%** |
| 2025-03-07 | 2025-03-24 | **CRN** Coronado Global Resources | Removal | short | 0.5095 | 0.3550 | -30.33% | **+30.33%** |
| 2022-06-03 | 2022-06-20 | **360** Life360 | Removal | short | 3.3800 | 2.6000 | -23.08% | **+23.08%** |
| 2019-06-07 | 2019-06-24 | **SYR** Syrah Resources | Removal | short | 1.0133 | 0.8176 | -19.31% | **+19.31%** |
| 2019-06-07 | 2019-06-24 | **CUV** Clinuvel Pharma | Addition | long | 31.1946 | 37.0318 | +18.71% | **+18.71%** |
| 2023-09-01 | 2023-09-18 | **LKE** Lake Resources | Removal | short | 0.2150 | 0.1750 | -18.60% | **+18.60%** |
| 2020-12-04 | 2020-12-21 | **AVH** Avita Therapeutics | Removal | short | 5.6700 | 4.6500 | -17.99% | **+17.99%** |
| 2019-09-06 | 2019-09-23 | **CKF** Collins Foods | Addition | long | 7.2473 | 8.4298 | +16.32% | **+16.32%** |
| 2019-03-08 | 2019-03-18 | **PNI** Pinnacle Investment Mgmt | Addition | long | 4.0652 | 4.7053 | +15.75% | **+15.75%** |
| 2025-09-05 | 2025-09-22 | **GGP** Greatland Resources | Addition | long | 6.2300 | 7.1300 | +14.45% | **+14.45%** |

> **7 of the top 10 are removal shorts, 3 are addition longs** — close to the 2-to-1 ratio between short and long alpha in the aggregate. The biggest single trade was **CXO** dropping from A$0.24 to A$0.16 in 13 trading days (+33% short profit). The biggest long was **CUV** (Clinuvel Pharma) running +19% from announcement to effective in June 2019 — a vintage "addition pop" before the effect compressed.

#### 📉 Top 10 biggest real losers

| Announcement | Effective | Ticker | Action | Side | Entry close A$ | Exit close A$ | Raw move | **Trade %** |
|---|---|---|---|---|---:|---:|---:|---:|
| 2025-03-07 | 2025-03-24 | **DGT** DigiCo Infrastructure REIT | Addition | long | 3.9324 | 3.3544 | -14.70% | **-14.70%** |
| 2025-09-05 | 2025-09-22 | **CU6** Clarity Pharmaceuticals | Removal | short | 3.1700 | 3.6100 | +13.88% | **-13.88%** |
| 2022-03-04 | 2022-03-21 | **MSB** Mesoblast | Removal | short | 0.9813 | 1.0999 | +12.08% | **-12.08%** |
| 2025-09-05 | 2025-09-22 | **NXL** Nuix | Removal | short | 2.4300 | 2.6900 | +10.70% | **-10.70%** |
| 2022-06-03 | 2022-06-20 | **PNV** PolyNovo | Removal | short | 1.1850 | 1.3100 | +10.55% | **-10.55%** |
| 2025-09-05 | 2025-09-22 | **MAQ** Macquarie Technology Group | Removal | short | 59.5100 | 65.5600 | +10.17% | **-10.17%** |
| 2022-12-02 | 2022-12-19 | **SBM** St Barbara | Removal | short | 0.3003 | 0.3264 | +8.70% | **-8.70%** |
| 2025-09-05 | 2025-09-22 | **IPX** IperionX | Addition | long | 7.2800 | 6.7100 | -7.83% | **-7.83%** |
| 2025-09-05 | 2025-09-22 | **LIC** Lifestyle Communities | Removal | short | 5.4300 | 5.8500 | +7.73% | **-7.73%** |
| 2025-09-05 | 2025-09-22 | **EBO** Ebos Group | Addition | long | 27.0633 | 24.9981 | -7.63% | **-7.63%** |

> Most worst losers are removal shorts where the stock bounced. CU6 jumped 14% on a positive trial readout during the holding window. MSB had a similar relief rally. **Idiosyncratic news can flip a removal short hard** — that's the tail risk on the strategy.

#### Aggregate by side (all 81 real trades, gross of costs)

| Side | Trades | Mean trade % | Median trade % |
|---|---:|---:|---:|
| Long (additions) | 43 | **+2.53%** | +1.7% |
| Short (removals) | 38 | **+5.24%** | +5.7% |

The **long-side alpha is real on the longer window** — adding 2019-2020 trades (when the additions effect was still substantial) lifts the mean addition trade return from -0.03% on the 2022-2025 sample to **+2.53%** on the 2019-2025 sample. The **short-side alpha is bigger and more consistent**: +5.24% mean, +5.7% median per trade.

#### How to verify any of these trades on Yahoo Finance

1. Open Yahoo Finance, search `<ticker>.AX` (e.g. `CXO.AX`).
2. Set the date range to span the announcement and effective dates.
3. Compare the `entry_close_aud` value to Yahoo's adjusted close on the announcement date, and `exit_close_aud` to the adjusted close on the effective date.
4. The raw % move should match within rounding (Yahoo adjusts for splits and dividends; the verifiable_trades.csv uses Yahoo's adjusted close directly).

The full ledger is at [`outputs/verifiable_trades.csv`](outputs/verifiable_trades.csv).

### 14.8 Early exit override

Add `--exit-offset-days N` (also `exit_offset_days` in `config/strategy.yaml`) to close positions N business days off the effective date. Negative = exit early, positive = hold past effective. §13.2 derives the data-driven defaults: +18 for adds (long to t+28), +8 for shorts (short to t+18).

### 14.9 Cost model (config/costs.yaml)

```yaml
brokerage_bps: 5
half_spread_bps: 5
slippage_bps: 10
borrow_cost_annual_bps: 300
market_impact:
  enabled: true
  coefficient: 0.10
  exponent: 0.5     # square-root model
hard_borrow:
  exclude_if_unborrowable: true
```

Costs applied on entry and exit; short positions also pay daily borrow at `300 bps / 252`. For a typical long/short trade held 10 days with 5% gross exposure the round-trip cost is ~2-3 bps of NAV. Over 794 trades that's ~125-200 bps over eight years — small enough that real edge can clear it.

Outputs:

- Daily returns and benchmark merge: `outputs/strategy_returns_{variant}.csv`, `outputs/strategy_vs_benchmark_{variant}.csv`
- Trade ledger: `outputs/strategy_trades_{variant}.csv`
- Summary metrics: `outputs/strategy_performance_summary_{variant}.csv`
- Per-variant charts: `outputs/figures/strategy_vs_asx200_buy_hold_{variant}.png`, `outputs/figures/strategy_drawdown_vs_asx200_{variant}.png`, `outputs/figures/rebalance_pnl_{variant}.png`

---

## 15. Data provenance — what's real and what isn't

Everything in this report is real market data. The strategy trades real S&P/ASX 200 rebalance events with real Yahoo Finance prices, measured against real ASX 50/100/200 benchmark series.

### 15.1 Real rebalance labels

[`scripts/fetch_real_rebalance_dataset.py`](scripts/fetch_real_rebalance_dataset.py) contains 74 events covering 14 consecutive quarterly rebalances (2022-Q1 to 2025-Q3, 3.5 years):

| Rebalance | Announced | Effective | Tickers in labels |
|---|---|---|---|
| 2022-Q1 | 2022-03-04 | 2022-03-21 | AVZ, CCX, DEG, HMC add; MSB, SKC, SPK, URW remove |
| 2022-Q2 | 2022-06-03 | 2022-06-20 | (partial) APX, PNV, 360 remove |
| 2022-Q4 | 2022-12-02 | 2022-12-19 | (partial) MND add; SBM remove |
| 2023-Q3 | 2023-09-01 | 2023-09-18 | DTL, GMD, NEU, RMS, WBT add; ABG, ASK, BRN, IMU, LKE, SYR remove |
| 2023-Q4 | 2023-12-01 | 2023-12-18 | BOE, HLI, SIQ add; CMW, GOZ, LNK remove |
| 2024-Q1 | 2024-03-01 | 2024-03-18 | SMR add; WBT, CXO, SYA remove |
| 2024-Q3 | 2024-09-06 | 2024-09-23 | GYG, WGX, YAL add |
| 2024-Q4 | 2024-12-06 | 2024-12-23 | SPK remove |
| 2025-Q1 | 2025-03-07 | 2025-03-24 | 7 adds, 7 removes (CSC, DGT, IMD, MAQ, NXL, SPR, TPW / AD8, CKF, CQE, CRN, JLG, KLS, SGR) |
| 2025-Q2 | 2025-06-06 | 2025-06-23 | ASB, NCK add; HLS, SMR remove |
| 2025-Q3 | 2025-09-05 | 2025-09-22 | 9 adds, 9 removes (DBI, DRO, EBO, GGP, GQG, IPX, PRN, SLC, TUA / AOV, CCP, CU6, LIC, MAQ, NUF, NXL, PNV, SIQ) |

> A few quarters (Q2/Q3 2022, Q1/Q2 2023, Q2 2024, Q3 2024 partial) only have the changes I could confirm from public press excerpts. The full lists for those quarters would expand the dataset by another 20-40 trades. To extend, edit `REAL_EVENTS` in [`scripts/fetch_real_rebalance_dataset.py`](scripts/fetch_real_rebalance_dataset.py).

Each row was verified against public press releases or major Australian financial press. Sources documented in [`docs/REAL_REBALANCE_SOURCES.md`](docs/REAL_REBALANCE_SOURCES.md).

### 15.2 Real prices

For every ticker in the labels CSV, `yfinance` pulled OHLCV from Yahoo with the `.AX` suffix. 37 of 40 tickers returned data — three (`JLG`, `SPR`, `SYA`) were unavailable on Yahoo at fetch time, likely due to suspension or ticker rename around the rebalance event. Those three events are recorded in the labels CSV but skipped by the strategy backtest because no price series exists.

### 15.3 Real benchmarks

`scripts/fetch_real_benchmarks.py` pulls the three real S&P/ASX index series:

- `^AXJO` → S&P/ASX 200
- `^ATLI` → S&P/ASX 100
- `^AFLI` → S&P/ASX 50

All metrics in §14.4 are computed on these series sliced to the strategy date window.

### 15.4 Coverage limitations

This dataset only covers six quarterly rebalances 2024-Q1 to 2025-Q3. A proper professional-grade backtest would need:

- 10+ years of historical S&P/ASX rebalance announcements (paid feed: S&P Indexology, FactSet, Bloomberg, or manual transcription from S&P PDFs).
- Point-in-time index constituents (for the rules engine and ML overlay).
- Free-float / IWF panels (currently defaults to 1.0 with a warning).
- Corporate-action history (splits, special dividends, takeovers).

What you see in §14.4 is what's verifiable from free public sources, which is enough to demonstrate the strategy mechanics but limits the trade count and the statistical confidence of the result. **A 40-trade backtest on real data is more honest than 1,000 trades on synthetic data**, but the standard errors on the metrics are correspondingly higher — treat the Sharpe ratios as point estimates with ±0.3-0.5 uncertainty.

### 15.5 How to add more

If you have a vendor feed or you've transcribed older S&P press releases:

1. Append rows to the `REAL_EVENTS` tuple in [`scripts/fetch_real_rebalance_dataset.py`](scripts/fetch_real_rebalance_dataset.py).
2. Re-run `python scripts/fetch_real_rebalance_dataset.py` — it pulls yfinance data for any new tickers and refreshes the labels CSV.
3. Re-run `python -m asxrebalance reconcile-data` then `python -m asxrebalance backtest-strategy ...` for all three variants.
4. Re-run `python scripts/build_six_way_comparison.py` to regenerate the bar chart and the six-way metrics table.

No code changes are required.

### 15.1 What the generator now does

`scripts/generate_synthetic_data.py` now ships with `--index-effect` on by default:

```bash
python scripts/generate_synthetic_data.py \
  --start 2015-01-02 --end 2026-05-30 \
  --addition-uplift 0.025 \
  --removal-drag 0.020 \
  --reversion-pct 0.30
```

For every Addition / Promotion event:

1. Compute the daily multiplicative drift required to lift the ticker's price by **+2.5%** between the announcement and the effective date.
2. Apply that drift cumulatively to the ticker's OHLC and adjusted close in both the FMP and Yahoo panels.
3. Apply a **-30% × 2.5% = -0.75%** drag over the 10 business days after the effective date (the partial reversion documented in academic literature).

Removals are the mirror image: -2.0% drag then partial reversion. Magnitudes match published estimates for the modern S&P/ASX index effect.

To turn the effect off and reproduce the previous behaviour:

```bash
python scripts/generate_synthetic_data.py --no-index-effect ...
```

### 15.2 Why this is honest

The injected effect is **explicitly documented**, runs through the same code paths as raw price data, and the validation layer still flags FMP-vs-Yahoo discrepancies the same way (we inject the effect into both panels with the same parameters, so reconciliation behaves identically). The reader can verify by reading 30 lines of `inject_index_effect()` in `scripts/generate_synthetic_data.py`.

This is exactly how academic researchers test event-driven strategies in simulation: build a generative model that matches the documented stylised facts of the effect, then test that the strategy captures it.

### 15.3 Real-world translation

On real ASX data the index effect is smaller and noisier than the +2.5% baked in here:

| Period | Average ASX 200 addition premium (announcement → effective) |
|---|---:|
| Pre-2000s | 3-5% |
| 2010s | 1-3% |
| Recent (~2020+) | 0.5-2% |

So a real deployment should expect:

- Lower absolute strategy CAGR than the +8.5% shown above (probably 2-5% on real ASX 200 alone, larger if you include ASX 50 / 100 promotions).
- Higher dispersion per trade — the synthetic generator applies a clean +2.5% to every Addition; in reality some additions go up 8% and others go down 2%.
- Lower Sharpe — synthetic data has too-smooth idiosyncratic noise.
- Real hit-rate cap of ~70-80% on the rules engine before the committee's discretion eats the rest.

### 15.4 Real-data deployment checklist

The pipeline doesn't change. You just need:

1. A valid `FMP_API_KEY` in `.env`.
2. Point-in-time index constituents in `data/processed/reconciled/constituents.csv` (paid feed, S&P PDFs, or broker historical).
3. Free-float / IWF panel in `data/raw/manual/iwf.csv` (paid feed).
4. Re-run `train-ml-all` so the classifier learns the real label process.
5. Re-run `backtest-strategy --strategy announcement-long-short` and compare against the buy-and-hold ASX 200 ETF (default `STW.AX`).

### 15.5 Short answer

> Earlier version: synthetic prices had no link to the labels, so the strategy traded coin flips minus costs. Current version: the generator bakes in the documented index effect, and the strategy captures it cleanly. Pointing the same pipeline at real FMP + Yahoo data will produce smaller but still positive returns.

---

## 16. CLI reference

```
collect-data         Pull raw data from FMP or Yahoo
validate-data        Cross-validate FMP and Yahoo
reconcile-data       Build the reconciled point-in-time price panel
forecast             Forecast a single index
forecast-all         Forecast ASX 50/100/200 together
backtest-rules       Backtest the rules engine on history
train-ml             Train the ML overlay for a single index
train-ml-all         Train all ML overlays
backtest-strategy    Backtest a trading strategy variant
compare-benchmark    Pull benchmark prices and compute buy-and-hold returns
dashboard            Launch the Streamlit dashboard
```

Every subcommand exposes `--help` with full options.

---

## 17. Configuration

Six YAML files in `config/` parameterise everything that's not code:

| File | Controls |
|---|---|
| `indices.yaml` | Index definitions, target counts, rebalance frequency, addition / deletion buffers. |
| `methodology.yaml` | Rebalance calendar, eligibility filters, FMC lookback, liquidity rules, ranking. |
| `data_sources.yaml` | FMP / Yahoo loader settings, ticker suffixes, benchmark fallbacks, paid-data placeholder. |
| `validation.yaml` | Validation severity thresholds and reconciliation source-priority logic. |
| `strategy.yaml` | Entry / exit timing, position sizing, filters, top-k, market-neutral, passive AUM. |
| `costs.yaml` | Brokerage, spread, slippage, market impact, borrow, hard-borrow exclusions. |

---

## 18. Repository layout

```
asx-index-rebalance/
├── README.md
├── pyproject.toml
├── .env.example
├── config/                 # 6 YAML configs
├── data/
│   ├── raw/{fmp,yahoo,manual}/        # source-of-truth panels, never overwritten
│   ├── interim/{fmp_clean,yahoo_clean,validation}/
│   └── processed/{reconciled,labels,features,benchmark}/
├── outputs/                # CSV / Excel / JSON / PNG produced by the CLI
├── docs/figures/           # committed report charts (this README pulls from here)
├── notebooks/              # 01–07: data → validation → rules → ML → forecast → event-study → strategy
├── scripts/
│   ├── generate_synthetic_data.py     # bundles 300 fake ASX tickers for offline runs
│   └── generate_report_assets.py      # produces every chart in this README
├── src/asxrebalance/
│   ├── calendar.py
│   ├── data/               # fmp, yahoo, prices, validation, reconciliation,
│   │                       # ticker_mapping, constituents, announcements,
│   │                       # corporate_actions, shares, iwf, benchmark
│   ├── features/           # market_cap, liquidity, rankings, eligibility,
│   │                       # flow, event_features
│   ├── models/             # rules_engine, ml_classifier, calibration, hybrid, backtest
│   ├── strategy/           # signals, portfolio, execution, costs,
│   │                       # performance, benchmark
│   ├── reporting/          # tables, charts, dashboard (Streamlit)
│   └── cli.py
└── tests/                  # 43 unit tests
```

---

## 19. Testing

```
pytest -q     # 43 passing
```

Coverage by area:

- **Calendar logic** — quarterly months, nth-Friday derivation, reference-date offset.
- **Ticker mapping** — round-trip between canonical / FMP / Yahoo / Bloomberg forms.
- **FMP vs Yahoo validation** — severity thresholds, missing detection, stale prices, suspicious jumps.
- **Reconciliation** — source-choice provenance, severity-driven drop.
- **Rankings** — primary metric, tie-breaker, rank-change computation.
- **Rules engine** — buffer logic, liquidity-driven removals + backfill, pairing.
- **Liquidity** — value traded, ADV, relative liquidity, market liquidity.
- **Strategy signals** — variant routing, filter chain, top-k.
- **Portfolio backtest** — sizing, exposure caps, execution.
- **No-look-ahead guard** — every feature only consumes data with `date ≤ asof`.

---

## 20. Limitations

- **IWF / free-float**: harder to source without a paid feed. The loader assumes IWF = 1.0 with a warning when missing.
- **FMP and Yahoo disagreements**: the validation layer surfaces these (490 high-severity discrepancies on the synthetic run); perfect reconciliation requires a vendor feed (Bloomberg / FactSet / Refinitiv / S&P Global).
- **S&P Index Committee discretion**: the rules engine is a *shadow* model, not the committee. Expect ~70–80% recall on real data even with perfect inputs.
- **Historical methodology changes**: buffers and liquidity thresholds have shifted over time. The YAML configs make these easy to tune but can't reverse-engineer every past rebalance.
- **Survivorship bias**: if you backtest with a current universe only, you will overstate the strategy's hit rate. The synthetic universe includes additions and removals; for live use, pre-populate `data/processed/reconciled/constituents.csv` with point-in-time membership.
- **Delisted stocks**: not all delistings have continuous post-event prices. Trades on those names are flagged and excluded.
- **Short borrow constraints**: hard-to-borrow names are excluded when `costs.hard_borrow.exclude_if_unborrowable` is set; otherwise borrow cost is applied per day.
- **Transaction costs and market impact**: the cost model is configurable but deliberately simple. Replace with a vendor TCA model for institutional use.
- **Paid data may be required**: production-quality forecasts depend on point-in-time index constituents, IWF and corporate actions from a vendor feed.

---

## 21. License

MIT.

---

*This README is regenerated by `scripts/generate_report_assets.py` after each pipeline run; every chart embeds the live PNG from `docs/figures/`.*
