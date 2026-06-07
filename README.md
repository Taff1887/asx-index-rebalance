# asx-index-rebalance-forecast

Forecast **S&P/ASX 50, S&P/ASX 100 and S&P/ASX 200** index rebalances and
backtest a tradeable rebalance strategy against a buy-and-hold ASX 200 benchmark.

This repository was built fresh from scratch. It does not depend on or reuse any
other project. All data ingestion, validation, modelling and backtesting logic
lives inside this repo, and the included synthetic dataset lets the entire
pipeline run end-to-end with no API keys.

---

## What this project does

1. **Replicates the S&P/ASX Australian Indices rules** for the ASX 50/100/200
   using the published rebalance calendar and ranking methodology.
2. **Cross-validates two independent data sources** (Financial Modeling Prep and
   Yahoo Finance) and produces a reconciled, point-in-time price panel with
   per-observation source attribution.
3. **Forecasts the next rebalance** for each index using:
   - a deterministic **rules engine** (rank buffers, liquidity tests, hierarchy),
   - a probabilistic **ML overlay** trained on historical labels,
   - a **passive-flow / index-impact** estimator,
   - and a **hybrid score** that combines the three.
4. **Backtests a trading strategy** built on the forecasts and **compares it
   against a buy-and-hold ASX 200 ETF** (STW.AX by default, configurable).
5. Exposes a CLI, a Streamlit dashboard, and seven notebooks that walk through
   each step of the pipeline.

## Why rebalances?

Inclusion in (or removal from) a major index drives mechanical buy/sell
activity from passive funds tracking the index. The size of that flow,
normalised by ADV, creates a short-lived dislocation that can be traded between
the **announcement date** and the **effective date**. The strategy in this repo
attempts to capture that effect across all three indices and explicitly costs
brokerage, half-spread, slippage, market impact and borrow.

---

## Index hierarchy

The S&P/ASX family is nested:

```
ASX 50  ⊂  ASX 100  ⊂  ASX 200  ⊂  ASX 300  ⊂  All Ordinaries
```

The model distinguishes between **Addition**, **Removal**, **Promotion**
(e.g. ASX 100 → ASX 50), **Demotion** (e.g. ASX 50 → ASX 100) and **no change**.
The rules engine respects this hierarchy when pairing additions with removals
to keep each index at its target count.

---

## Data sources

We use **two independent sources** and reconcile them:

| Source | Used for                                                                 |
|--------|--------------------------------------------------------------------------|
| **FMP** (Financial Modeling Prep) | Prices, volume, market cap, shares outstanding, company profile, sector / industry, corporate actions (where available). Primary source for OHLCV and reference data. |
| **Yahoo Finance** (yfinance)      | Prices, adjusted close, volume, benchmark ETFs (STW.AX, IOZ.AX, A200.AX). Cross-check for FMP. Default source for adjusted close. |

Set the API key in `.env` (copy `.env.example`):

```bash
FMP_API_KEY=your_key_here
DATA_SOURCE_PRIMARY=fmp
DATA_SOURCE_SECONDARY=yahoo
BENCHMARK_TICKER=STW.AX
```

If the API key is missing, the FMP loader falls back to local CSVs under
`data/raw/fmp/`. The Yahoo loader does the same with `data/raw/yahoo/`.

### Data validation

The validation layer compares the two sources on every (date, ticker) and
classifies each discrepancy with a severity bucket. Thresholds are in
`config/validation.yaml`:

```yaml
validation:
  price_diff_threshold_pct_low: 0.5
  price_diff_threshold_pct_medium: 2.0
  price_diff_threshold_pct_high: 5.0
  volume_diff_threshold_pct_low: 10.0
  volume_diff_threshold_pct_medium: 25.0
  volume_diff_threshold_pct_high: 50.0
  stale_price_days: 5
  suspicious_daily_return_pct: 25.0
  min_required_source_coverage_pct: 95.0
```

Outputs land in `outputs/` and include the **chosen** value, the **source
chosen**, and the **reason**.

### Reconciliation rules

`config/validation.yaml::reconciliation` controls how the reconciler picks a
winner per field:

- Adjusted close defaults to **Yahoo** (better corporate-action handling).
- Close defaults to **FMP**.
- Volume defaults to **FMP**.
- Observations at or above the `drop_severity` threshold are dropped; raw
  values stay in `data/raw/`.

Source data is **never overwritten** — `data/raw/{fmp,yahoo}/` are append-only,
and reconciled outputs go to `data/processed/reconciled/`.

---

## Repository layout

```
asx-index-rebalance-forecast/
├── README.md
├── pyproject.toml
├── .env.example
├── config/
│   ├── indices.yaml          # ASX index definitions and buffers
│   ├── methodology.yaml      # Calendar, eligibility, ranking, liquidity
│   ├── data_sources.yaml     # FMP / Yahoo loader settings
│   ├── validation.yaml       # Validation thresholds and reconciliation rules
│   ├── strategy.yaml         # Strategy entry/exit, sizing, filters
│   └── costs.yaml            # Brokerage, spread, slippage, market impact, borrow
├── data/
│   ├── raw/{fmp,yahoo,manual}/
│   ├── interim/{fmp_clean,yahoo_clean,validation}/
│   └── processed/{reconciled,labels,features,benchmark}/
├── outputs/
│   └── figures/
├── notebooks/                # 01–07: data → validation → rules → ML → forecast → event-study → strategy
├── scripts/
│   └── generate_synthetic_data.py
├── src/asxrebalance/
│   ├── calendar.py
│   ├── data/                 # fmp, yahoo, prices, validation, reconciliation,
│   │                         # ticker_mapping, constituents, announcements,
│   │                         # corporate_actions, shares, iwf, benchmark
│   ├── features/             # market_cap, liquidity, rankings, eligibility,
│   │                         # flow, event_features
│   ├── models/               # rules_engine, ml_classifier, calibration, hybrid, backtest
│   ├── strategy/             # signals, portfolio, execution, costs,
│   │                         # performance, benchmark
│   ├── reporting/            # tables, charts, dashboard (Streamlit)
│   └── cli.py
└── tests/
```

---

## Quickstart

```bash
# 1. Create a virtual env and install (the [all] extra includes Streamlit, XGBoost, dev tools)
python -m venv .venv
.venv\Scripts\activate              # on Windows
pip install -e ".[all]"

# 2. Seed the repo with synthetic data so everything runs offline
python scripts/generate_synthetic_data.py

# 3. Cross-validate FMP vs Yahoo
python -m asxrebalance validate-data --start 2018-01-01 --end 2026-06-01

# 4. Reconcile into a single point-in-time price panel
python -m asxrebalance reconcile-data --start 2018-01-01 --end 2026-06-01

# 5. Forecast the next rebalance for all three indices
python -m asxrebalance forecast-all --asof 2026-06-01

# 6. Backtest the rules engine on history (ASX 200)
python -m asxrebalance backtest-rules --index ASX200 --start 2020-01-01 --end 2025-12-31

# 7. Train the ML overlay (or pass --model random_forest / xgboost)
python -m asxrebalance train-ml --index ASX200

# 8. Run the strategy backtest and compare against ASX 200 buy-and-hold
python -m asxrebalance backtest-strategy \
    --strategy announcement-long-short --start 2020-01-01 --end 2025-12-31

# 9. Launch the dashboard
python -m asxrebalance dashboard
```

### All CLI commands

```text
collect-data         Pull raw data from FMP or Yahoo
validate-data        Cross-validate FMP and Yahoo
reconcile-data       Build the reconciled point-in-time panel
forecast             Forecast a single index
forecast-all         Forecast ASX 50/100/200 together
backtest-rules       Backtest the rules engine on history
train-ml             Train the ML overlay for a single index
train-ml-all         Train all ML overlays
backtest-strategy    Backtest a trading strategy
compare-benchmark    Pull benchmark prices and compute buy-and-hold returns
dashboard            Launch the Streamlit dashboard
```

Strategy variants supported by `backtest-strategy`:

| Variant                     | Description                                                    |
|-----------------------------|----------------------------------------------------------------|
| `announcement-long-short`   | Long predicted additions/promotions, short removals/demotions, hold announcement → effective. |
| `pre-announcement`          | Enter 5 trading days before announcement using only ex-ante information. |
| `additions-only`            | Long-only predicted additions/promotions. |
| `market-neutral`            | Long/short with explicit beta hedge using benchmark ETF. |
| `flow-pressure`             | Trade only when expected `passive_flow_to_ADV` exceeds the threshold. |
| `top-k`                     | Each rebalance, trade only the top-k highest-conviction events. |

---

## How the rules engine works

For each (index, rebalance date):

1. Build the eligible universe (active ASX-listed ordinaries, not ETF/LIC).
2. Compute the **average float-adjusted market cap** over the methodology
   lookback (default 63 business days), using
   `close × shares_outstanding × IWF`.
3. Compute **liquidity** — median daily value traded over the same window,
   relative to a cap-weighted market liquidity.
4. **Rank** by average float-adjusted market cap.
5. Apply **rank buffers** per index (ASX 200 defaults: addition ≤ 179,
   removal > 221) and the **liquidity pass** flag.
6. Pair additions and removals so the index stays at its target count.
7. Promote / demote between adjacent tiers when the hierarchy implies it.

Output columns include: `fmc_rank`, `avg_fmc_3m`, `relative_liquidity`,
`liquidity_pass`, `rank_buffer_margin`, `predicted_action`,
`confidence_bucket`, `reason`.

## The ML overlay

A calibrated classifier per `(index, target)` predicts the probability of an
addition or removal at the next rebalance. Targets:

- `add_to_asx50_next_rebalance`, `remove_from_asx50_next_rebalance`
- `add_to_asx100_next_rebalance`, `remove_from_asx100_next_rebalance`
- `add_to_asx200_next_rebalance`, `remove_from_asx200_next_rebalance`

Features include `fmc_rank`, `rank_change_1m`, `rank_change_3m`, `avg_fmc_3m`,
`market_cap_gap_to_cutoff`, current membership flags, `relative_liquidity`,
`median_daily_value_traded`, `ADV_20d`, `ADV_60d`, trailing returns and
volatility, corporate-action flags and a `data_quality_score`.

**Validation is strictly time-series** via `TimeSeriesSplit`. We never train on
data that postdates the rebalance we are predicting. Probabilities are
calibrated with `CalibratedClassifierCV`. We report precision, recall, F1,
PR-AUC, a top-k hit rate, the confusion matrix and feature importance.

## The hybrid score

```text
hybrid_probability = weighted average of:
    rules_engine_signal, ML probability, rank-buffer margin,
    liquidity pass, flow pressure, historical hit rate, data-quality score
```

Weights live in `src/asxrebalance/models/hybrid.py::DEFAULT_WEIGHTS` and can be
overridden via the API.

## Passive flow model

For each predicted change, the model estimates:

```text
expected_index_weight = stock_FMC / total_index_FMC
passive_buy_or_sell   = weight_change × assumed_passive_AUM_for_index
passive_flow_to_ADV   = |passive_flow_AUD| / (ADV_shares × reference_price)
```

`config/strategy.yaml::passive_aum` carries placeholder assumptions for each
index (AUD): users should update these with their own AUM estimates.

## Backtest assumptions

Transaction costs (`config/costs.yaml`):

```yaml
costs:
  brokerage_bps: 5
  half_spread_bps: 5
  slippage_bps: 10
  borrow_cost_annual_bps: 300
  market_impact:
    enabled: true
    coefficient: 0.10
    exponent: 0.5
  hard_borrow:
    exclude_if_unborrowable: true
```

The backtest:

- is strictly time-series — no look-ahead bias;
- uses **announcement → effective** windows for the announcement-driven
  strategy and **t-5 → effective** for the pre-announcement strategy;
- includes brokerage, half-spread, slippage, market impact and borrow;
- flags data-quality exclusions; and
- can be re-run on **FMP-only**, **Yahoo-only**, or **reconciled** data to
  test sensitivity to the input source.

---

## Outputs

### Data validation
```
outputs/data_quality_summary.csv
outputs/data_quality_issues.csv
outputs/fmp_vs_yahoo_price_differences.csv
outputs/fmp_vs_yahoo_volume_differences.csv
outputs/missing_data_report.csv
outputs/reconciliation_summary.csv
```

### Forecasts
```
outputs/current_forecast_ASX50.csv
outputs/current_forecast_ASX100.csv
outputs/current_forecast_ASX200.csv
outputs/current_forecast_all.xlsx
```

### Backtests
```
outputs/rules_engine_accuracy.csv
outputs/model_metrics.json
outputs/strategy_returns.csv
outputs/strategy_trades.csv
outputs/strategy_performance_summary.csv
outputs/strategy_vs_benchmark.csv
```

### Charts
```
outputs/figures/strategy_vs_asx200_buy_hold.png  # the key chart
outputs/figures/strategy_drawdown_vs_asx200.png
outputs/figures/feature_importance.png
outputs/figures/rebalance_pnl.png
outputs/figures/monthly_return_heatmap.png
...etc
```

---

## Example forecast table (synthetic data)

| ticker | index   | predicted_action | fmc_rank | hybrid_probability | passive_flow_to_ADV_20d | confidence_bucket  |
|--------|---------|------------------|----------|--------------------|--------------------------|--------------------|
| XYZ    | ASX200  | Addition         | 178      | 0.83               | 1.4                      | high conviction    |
| ABC    | ASX200  | Removal          | 230      | 0.71               | 0.8                      | medium conviction  |
| DEF    | ASX100  | Promotion        | 92       | 0.74               | 0.6                      | medium conviction  |
| GHI    | ASX50   | Demotion         | 47       | 0.66               | 0.5                      | medium conviction  |

## Example strategy performance table

| metric            | strategy | ASX 200 buy-and-hold | excess |
|-------------------|---------:|---------------------:|-------:|
| CAGR              | 0.085    | 0.062                | 0.023  |
| Vol               | 0.12     | 0.15                 | -0.03  |
| Sharpe            | 0.71     | 0.41                 | 0.30   |
| Max drawdown      | -0.07    | -0.21                | 0.14   |
| Hit rate          | 0.55     | 0.53                 | 0.02   |
| Information ratio | 0.65     | —                    | —      |

> These numbers depend on the input data and synthetic seed; rerun
> `python -m asxrebalance backtest-strategy ...` to see live numbers on your data.

---

## Known limitations

- **IWF / free-float availability**: free-float is hard to source without a
  paid feed. The synthetic dataset assumes plausible IWFs; in real use the
  loader falls back to `1.0` with a warning.
- **Data quality**: FMP and Yahoo can disagree, especially around corporate
  actions and on illiquid names. The validation layer surfaces this, but
  perfect reconciliation requires a vendor feed (Bloomberg / FactSet /
  Refinitiv / S&P Global). The `PaidDataLoader` placeholder is the integration
  point for that.
- **Committee discretion**: S&P/ASX retains final discretion on inclusion
  and exclusion. The rules engine is a *shadow* model, not the committee.
- **Historical methodology changes**: rank buffers and liquidity thresholds
  have shifted historically; the YAML configs make these easy to tune but
  cannot reverse-engineer the methodology used at every past rebalance.
- **Survivorship bias**: if you back-test using a current universe only, you
  will overstate the strategy's hit rate. The synthetic universe generated by
  this repo includes additions and removals; for live use, pre-populate
  `data/processed/reconciled/constituents.csv` with point-in-time membership.
- **Delisted stocks**: not all delistings have continuous post-event prices.
  Trades on those names are flagged and excluded.
- **Short borrow constraints**: hard-to-borrow names are excluded when
  `costs.hard_borrow.exclude_if_unborrowable` is set; otherwise borrow cost is
  applied per day.
- **Transaction costs and market impact**: the cost model is configurable but
  is a deliberate simplification. Institutional users should replace it with
  their own TCA model.
- **Paid data may be required**: for production-quality forecasts you should
  source point-in-time index constituents, IWF and corporate actions from a
  paid feed.

---

## Testing

```bash
pytest -q
```

Key test files:

- `tests/test_calendar.py` — rebalance date logic
- `tests/test_ticker_mapping.py` — ticker normalisation
- `tests/test_data_validation.py` — FMP vs Yahoo checks
- `tests/test_reconciliation.py` — source-choice and provenance
- `tests/test_rankings.py` — ranking and rank-change logic
- `tests/test_rules_engine.py` — buffers, hierarchy, liquidity
- `tests/test_liquidity.py` — value-traded / ADV / relative liquidity
- `tests/test_strategy_signals.py` — signal building and filters
- `tests/test_portfolio_backtest.py` — sizing and execution
- `tests/test_no_lookahead.py` — guard against look-ahead bias

---

## License

MIT.
