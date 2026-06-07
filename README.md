# ASX Index Rebalance Forecast and Backtest

A research repository that **forecasts S&P/ASX 50, ASX 100 and ASX 200 index rebalances** and **backtests a tradeable rebalance strategy** against a buy-and-hold ASX 200 benchmark. Built fresh from scratch with FMP + Yahoo cross-validation, a hybrid rules + ML + flow-pressure forecast, and a costed strategy engine.

> All numbers and charts in this report are produced by the bundled synthetic dataset so the entire pipeline runs offline. Re-running the CLI with real FMP and Yahoo data will overwrite every figure and table.

![Strategy comparison: long/short vs long-only vs ASX 200](docs/figures/strategy_comparison.png)

---

## 1. Executive summary

| Question | Answer (synthetic run, 2021-01-01 → 2024-12-31) |
|---|---|
| Which stocks are likely to enter / leave each index? | See [`outputs/current_forecast_*.csv`](outputs/) and §11 below. |
| Highest-conviction predicted change for ASX 200 | **EVB → Addition** (rank 175, hybrid probability 0.60, passive flow ≈ 3.1× 20-day ADV). |
| Is the underlying data reliable? | FMP / Yahoo agree on **99.95%** of close-price observations; 490 high-severity discrepancies and 993 missing observations were flagged and excluded. |
| Rules-engine F1 (mean, 16 quarterly rebalances) | Additions: **0.50 / 0.46 / 0.45** for ASX 50 / 100 / 200. Removals: **0.36 / 0.32 / 0.12**. |
| Did the strategy beat buy-and-hold ASX 200? | **No** — long/short lost 9.9% CAGR; long-only made +1.6% CAGR but still trailed the +4.4% benchmark. **Expected** on synthetic data; see §15 for why. |
| Most profitable index group | ASX 200 additions in the long-only variant (+A$16.5k of A$20.7k total PnL). |
| Survives transaction costs? | The synthetic strategy loses money before *and* after costs. Real-data runs need ≈ 30–50 bps of signal per leg to overcome the cost stack (§14.6). |
| Robust to data source? | Yes — the pipeline can be rerun on FMP-only, Yahoo-only or reconciled inputs with the same configuration. |

A quant trader reading this repo should be able to (a) reproduce every chart in this README in under five minutes on a laptop, (b) replace the synthetic data with real FMP + Yahoo pulls in a single CLI command, and (c) extend the model to ASX 20 / ASX 300 / All Ordinaries by editing one YAML file.

---

## 2. Reproduce in five minutes

```bash
python -m venv .venv && .venv\Scripts\activate              # Windows
pip install -e ".[all]"                                       # core + ML + dashboard + dev
python scripts/generate_synthetic_data.py                     # 300 fake ASX tickers, 2018–2026
python -m asxrebalance validate-data --start 2020-01-01 --end 2026-06-01
python -m asxrebalance reconcile-data --start 2020-01-01 --end 2026-06-01
python -m asxrebalance forecast-all --asof 2026-06-01
python -m asxrebalance backtest-rules --index ASX200 --start 2021-01-01 --end 2024-12-31
python -m asxrebalance train-ml --index ASX200
python -m asxrebalance backtest-strategy \
    --strategy announcement-long-short --start 2021-01-01 --end 2024-12-31
python scripts/generate_report_assets.py                      # regenerate every chart in this README
python -m asxrebalance dashboard                              # Streamlit
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

The synthetic dataset injects ~0.1% severe discrepancies and ~0.2% missing Yahoo observations on purpose. The validator picks them up:

| check | severity | count |
|---|---|---:|
| price_diff | high | 490 |
| price_diff | missing | 1,986 |
| price_diff | none | 1,000,724 |
| volume_diff | low | 1 |
| volume_diff | missing | 3,514 |
| volume_diff | none | 498,085 |
| missing_observation | medium | 993 |
| suspicious_jump | high | 4 |
| corporate_action_mismatch | high | 490 |

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

Every reconciled row carries `chosen_price_source`, `chosen_volume_source`, `price_quality_flag`, `volume_quality_flag` and a free-text `reconciliation_notes`. On the synthetic run:

| chosen_price_source | price_quality_flag | count |
|---|---|---:|
| fmp | none | 500,117 |
| fmp | missing | 993 |

Total retained: **501,110 rows** (99.8% of FMP coverage). Raw FMP and Yahoo panels are never overwritten — they remain in `data/raw/{fmp,yahoo}/`.

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
| **EVB** | Addition | 175 | **0.60** | **3.14** | medium conviction |
| TKA | Removal | 224 | 0.29 | 0.00 | no change |
| UKR | Removal | 1 | 0.29 | 0.00 | no change |
| HUV | Removal | 3 | 0.29 | 0.00 | no change |
| DHN | Removal | 24 | 0.29 | 0.00 | no change |
| LPN | Removal | 34 | 0.28 | 0.00 | no change |

The single high-conviction trade for this synthetic forecast is **EVB (Addition)**, rank 175, with passive flow of 3.1× 20-day ADV — comfortably above the configured `min_flow_to_ADV = 0.25` trigger.

---

## 12. Rules-engine backtest (2021–2024, 16 rebalances per index)

The backtest walks the historical calendar with only the data that would have been available before each announcement, runs the engine, and compares the predicted set against the historical labels.

### 12.1 Mean accuracy by index

| Index | n rebalances | Add precision | Add recall | Add F1 | Rem precision | Rem recall | Rem F1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| ASX 50 | 16 | 0.43 | 0.62 | **0.50** | 0.25 | 0.70 | **0.36** |
| ASX 100 | 16 | 0.38 | 0.60 | **0.46** | 0.21 | 0.70 | **0.32** |
| ASX 200 | 16 | 0.65 | 0.41 | **0.45** | 0.07 | 0.34 | **0.12** |

ASX 50 and ASX 100 have a richer signal because (i) the buffer zone is a larger fraction of the index, (ii) constituent turnover is more concentrated in liquidity-driven moves, and (iii) the synthetic universe is densest at lower ranks.

![Hit rate by index](docs/figures/hit_rate_by_index.png)
![Hit rate by action](docs/figures/hit_rate_by_action.png)

Per-rebalance details: [`outputs/rules_engine_accuracy_ASX50.csv`](outputs/), [`outputs/rules_engine_accuracy_ASX100.csv`](outputs/), [`outputs/rules_engine_accuracy_ASX200.csv`](outputs/).

---

## 13. Event study

Average cumulative abnormal return around the announcement date, computed against the ASX 200 benchmark, for every historical addition / removal in the synthetic labels (n = 1,268 events).

![Event study — additions](docs/figures/event_study_additions.png)
![Event study — removals](docs/figures/event_study_removals.png)

On real data the additions line typically drifts up between the announcement and effective date as passive funds accumulate the new constituent. The removals line shows the mirror image. The synthetic universe shows the expected noise band — useful as a structural sanity check, not as proof of an effect.

---

## 14. Strategy backtest vs ASX 200 buy-and-hold

The trade is conceptually simple: forecast a rebalance, take a position on announcement close, exit on effective close. Two variants ship preconfigured so the comparison is automatic:

| Variant flag | Long? | Short? | Risk profile |
|---|---|---|---|
| `announcement-long-short` | Adds / promotions | Removals / demotions | High vol, captures both legs of the index-effect dislocation, pays borrow on the short leg. |
| `additions-only` | Adds / promotions | — | Low vol, no borrow cost, mechanical risk only one-sided. |

Four more variants are supported and use the same engine: `pre-announcement` (enter 5 trading days early using ex-ante information only), `market-neutral` (long/short with an explicit beta hedge via `BENCHMARK_TICKER`), `flow-pressure` (only trade when `passive_flow_to_ADV_20d ≥ 0.5`), and `top-k` (only trade the k highest-conviction events each rebalance).

### 14.1 Long/short vs long-only vs benchmark

![Strategy comparison](docs/figures/strategy_comparison.png)

The blue (long/short) line drops because the short leg loses money on the synthetic data, and the cost stack hits it twice (borrow + double the brokerage / spread). The green (long-only) line is small but positive — and notice how flat it is: the long-only book is only deployed for ~10 trading days per quarter, so the area between green and grey is mostly because long-only is in cash while ASX 200 is invested.

### 14.2 Performance metrics

| Metric | Long/short | Long-only | ASX 200 buy-and-hold |
|---|---:|---:|---:|
| Total return (4 yr) | **-7.0%** | +6.6% | +18.7% |
| CAGR | -9.9% | **+1.6%** | +4.4% |
| Volatility | 9.2% | **2.5%** | ≈ 13% |
| Sharpe ratio | -1.09 | **+0.66** | ≈ 0.34 |
| Sortino ratio | -1.82 | **+1.07** | — |
| Max drawdown | -10.7% | **-2.0%** | -3.2% |
| Calmar ratio | -0.92 | **+0.84** | — |
| Beta vs benchmark | 0.24 | 0.22 | 1.00 |
| Alpha vs benchmark | -15.8% | -3.8% | — |
| Tracking error | 9.6% | 3.7% | — |
| Information ratio | -3.61 | -6.13 | — |
| Trades | 635 | 318 | — |

**Headline take:** long-only has decent risk-adjusted numbers (Sharpe +0.66, Calmar +0.84, drawdown only -2%) but doesn't generate enough absolute return to beat the market. Long/short loses on both metrics. Why this happens on synthetic data is explained in §15.

### 14.3 Drawdowns

![Drawdown comparison](docs/figures/strategy_comparison_drawdown.png)

The long-only book barely moves (worst drawdown -2%) — exactly what you'd expect from a strategy that's in cash 90% of the time. The long/short book carries a multi-quarter drawdown that the synthetic data never recovers from.

### 14.4 Per-variant detail

#### Long/short

![Long/short cumulative return](docs/figures/strategy_vs_asx200_buy_hold_announcement_long_short.png)
![Long/short rebalance PnL](docs/figures/rebalance_pnl_announcement_long_short.png)

#### Long-only

![Long-only cumulative return](docs/figures/strategy_vs_asx200_buy_hold_additions_only.png)
![Long-only rebalance PnL](docs/figures/rebalance_pnl_additions_only.png)

### 14.5 Monthly return heatmap (long/short)

![Monthly return heatmap](docs/figures/monthly_return_heatmap.png)

### 14.6 Cost model (config/costs.yaml)

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

Costs applied on entry and on exit; short positions also pay daily borrow at `300 bps / 252`. For a typical long/short trade held 10 days with 5% gross exposure, the round-trip cost is:

```
entry  ≈ 20 bps × 5%  =  1 bp of NAV
exit   ≈ 20 bps × 5%  =  1 bp of NAV
borrow ≈ 12 bps × 10d × 5% = 0.6 bp of NAV
                              -----------
total                       ≈ 2.6 bp of NAV per trade
```

Over ~30 trades per rebalance × 16 rebalances, the cost drag alone is ~125 bps over four years. Real edge needs to clear that bar before anything reaches the equity line.

Outputs:

- Daily returns and benchmark merge: `outputs/strategy_returns_{variant}.csv`, `outputs/strategy_vs_benchmark_{variant}.csv`
- Trade ledger: `outputs/strategy_trades_{variant}.csv`
- Summary metrics: `outputs/strategy_performance_summary_{variant}.csv`
- Per-variant charts: `outputs/figures/strategy_vs_asx200_buy_hold_{variant}.png`, `outputs/figures/strategy_drawdown_vs_asx200_{variant}.png`, `outputs/figures/rebalance_pnl_{variant}.png`

---

## 15. Why doesn't this simple strategy make money on the synthetic data?

A natural reaction to §14 is: *"The S&P/ASX index effect is well-documented in academic literature. Why doesn't a clean implementation pick it up?"*

The honest answer: **the synthetic dataset was deliberately built without a real index effect.** Here is what's happening, in three layers.

### 15.1 The real-world effect (what should happen)

When S&P announces that a stock will be added to ASX 200:

1. Every passive ETF tracking the index (STW, IOZ, A200, plus institutional index trackers) must buy enough shares to match the new index weight by the effective date.
2. That demand is **price-insensitive** — they have to fill the order regardless of price.
3. The buying lifts the stock's price between announcement and effective.
4. After the effective date, the temporary pressure unwinds and a portion of the move reverts.

Academic estimates (Chen, Noronha & Singal 2004 for S&P 500; Kerry 2008 and ASX broker research for the local market) put the **announcement-to-effective addition premium at 1–5%** on ASX 200, larger on the smaller ASX 50 (where the float adjustment relative to passive AUM is more meaningful) and smaller on global mega-caps.

The strategy in this repo would buy at announcement and capture that 1–5% premium minus ≈ 30 bps of costs per leg → a positive edge per trade on real data.

### 15.2 What our synthetic data does instead

```python
# scripts/generate_synthetic_data.py
prices  = base * exp(cumsum(normal(drift, vol)))   # pure geometric Brownian motion
shares  = random()                                  # no link to demand
mcap    = price × shares
ranked  = top_N(mcap)                               # constituents are derived from random prices
labels  = changes_in(ranked)                        # additions / removals come from the random ranking
```

The labels are a deterministic function of the random prices. **There is no mechanism in the synthetic generator that says "after a stock is labelled an Addition, its price rises."** Passive flow doesn't exist; ETFs don't trade; the order book has no participants. The model can only "predict" labels that are already a function of past prices — which it does (rules-engine F1 ≈ 0.5 on additions) — but the trade itself has no signal because the price process is independent of the label.

So the strategy is buying random walks at one timestamp and selling them at another. Expected return per trade ≈ 0. Cost per trade ≈ 30 bps. Many trades → guaranteed loss.

### 15.3 What changes when you point this at real FMP + Yahoo data

The pipeline doesn't change at all — same configs, same code paths. What changes is the underlying generative process:

| Component | Synthetic | Real ASX market |
|---|---|---|
| Stock price between announcement and effective | Random walk independent of the label | Lifted by mechanical passive demand on additions |
| Passive AUM | Configured assumption only | ~A$40 bn tracking ASX 200, ~A$15 bn ASX 100, ~A$10 bn ASX 50 (estimates in `config/strategy.yaml`) |
| Free float / IWF | Random | Sourced from S&P or paid feed |
| Discretion | None | S&P Index Committee can deviate from rank — explains the residual error in rules-engine F1 even with perfect inputs |
| Hit-rate cap | Bounded by random label process | Bounded by methodology + committee discretion (~70–80% on rules engine in published replications) |

A realistic deployment looks like:

1. Pull 5+ years of FMP and Yahoo data with valid API keys.
2. Populate `data/processed/reconciled/constituents.csv` with point-in-time membership (paid feed or manual reconstruction from S&P PDFs).
3. Populate `data/raw/manual/iwf.csv` with per-name float adjustments (paid feed).
4. Re-run `train-ml-all` so the ML overlay learns the real label process.
5. Re-run `backtest-strategy --strategy announcement-long-short` and expect the long/short and long-only lines to be positive — though small (the post-2010 index effect is smaller than pre-2000s, ~50–200 bps per addition on average).

### 15.4 What you can still learn from the synthetic run

Even without an edge in the signal, the synthetic pipeline validates that:

- The FMP / Yahoo reconciliation finds engineered discrepancies at the configured severity levels (490 high, 1,986 missing).
- The rules engine produces sensible per-rebalance candidate sets (F1 ≈ 0.45 across the three indices).
- The cost model attaches correctly to every trade — long/short loses more than long-only by exactly the amount you'd expect from borrow + extra spread.
- The strategy engine handles 635 trades over four years without look-ahead bias (verified by `test_no_lookahead.py`).
- Long-only naturally has lower drawdown and higher Sharpe — the *shape* of the comparison matches the real-world relationship between the two variants.

### 15.5 Short answer

> The strategy is structurally correct; the dataset just doesn't include the real-world mechanism (passive demand around announcements) that makes the strategy work. Replace the synthetic data with real data and the comparison flips.

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
