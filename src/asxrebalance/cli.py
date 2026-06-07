"""Command-line interface for asxrebalance.

Examples
--------
::

    python -m asxrebalance collect-data --source fmp --start 2018-01-01 --end 2026-06-01
    python -m asxrebalance validate-data --start 2018-01-01 --end 2026-06-01
    python -m asxrebalance reconcile-data --start 2018-01-01 --end 2026-06-01
    python -m asxrebalance forecast --index ASX200 --asof 2026-06-01
    python -m asxrebalance forecast-all --asof 2026-06-01
    python -m asxrebalance backtest-rules --index ASX200 --start 2018-01-01 --end 2026-06-01
    python -m asxrebalance train-ml --index ASX200
    python -m asxrebalance backtest-strategy --strategy announcement-long-short \
        --start 2018-01-01 --end 2026-06-01
    python -m asxrebalance compare-benchmark --benchmark STW.AX
    python -m asxrebalance dashboard
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from .calendar import iter_rebalance_windows
from .config import load_indices_config, load_strategy_config, load_validation_config
from .data.benchmark import buy_and_hold_returns, load_benchmark
from .data.constituents import current_constituents, load_constituents
from .data.announcements import load_labels
from .data.fmp import FMPPriceLoader
from .data.iwf import asof_iwf, load_iwf
from .data.reconciliation import (
    create_reconciled_price_panel,
    summarise_reconciliation,
)
from .data.shares import asof_shares, load_shares_outstanding
from .data.validation import generate_data_quality_report
from .data.yahoo import YahooFinancePriceLoader
from .features.event_features import build_event_features
from .features.flow import expected_index_weights, flow_to_adv, passive_flow
from .features.liquidity import (
    adv,
    median_daily_value_traded,
    relative_liquidity,
    stock_liquidity_ratio,
)
from .features.market_cap import asof_float_market_cap
from .features.rankings import rank_universe
from .models.hybrid import hybrid_confidence_bucket, hybrid_score
from .models.ml_classifier import (
    DEFAULT_FEATURES,
    feature_importance,
    predict_proba,
    train_classifier,
    write_metrics_json,
)
from .models.rules_engine import run_rules_engine_for_all_indices
from .paths import (
    FIGURES_DIR,
    INTERIM_FMP_CLEAN_DIR,
    INTERIM_YAHOO_CLEAN_DIR,
    OUTPUTS_DIR,
    PROCESSED_BENCHMARK_DIR,
    PROCESSED_RECONCILED_DIR,
    RAW_FMP_DIR,
    RAW_YAHOO_DIR,
    ensure_dirs,
)
from .reporting.charts import (
    bar_chart,
    calibration_chart,
    cumulative_return_chart,
    drawdown_chart,
    rebalance_pnl_chart,
)
from .reporting.tables import write_csv, write_excel
from .strategy.benchmark import benchmark_daily_returns
from .strategy.execution import backtest_positions
from .strategy.performance import summary_metrics
from .strategy.portfolio import enforce_exposure, expand_to_positions, size_positions
from .strategy.signals import apply_filters, build_event_signals, top_k_signals


log = logging.getLogger("asxrebalance.cli")


# -----------------------------------------------------------------------------
# Data loading helpers
# -----------------------------------------------------------------------------
def _load_from_dir(dir_path: Path) -> pd.DataFrame:
    frames = []
    for f in sorted(dir_path.glob("*.csv")):
        if f.name in ("shares.csv", "iwf.csv"):
            continue
        df = pd.read_csv(f, parse_dates=["date"])
        df["ticker"] = f.stem
        df["source"] = dir_path.name.replace("_clean", "").replace("raw", "")
        frames.append(df)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _load_raw_panel(source: str) -> pd.DataFrame:
    if source == "fmp":
        return _load_from_dir(RAW_FMP_DIR)
    if source == "yahoo":
        return _load_from_dir(RAW_YAHOO_DIR)
    raise ValueError(f"Unknown source: {source}")


def _load_clean_panel(source: str) -> pd.DataFrame:
    if source == "fmp":
        return _load_from_dir(INTERIM_FMP_CLEAN_DIR)
    if source == "yahoo":
        return _load_from_dir(INTERIM_YAHOO_CLEAN_DIR)
    raise ValueError(f"Unknown source: {source}")


def _to_per_ticker(panel: pd.DataFrame, target_dir: Path) -> None:
    target_dir.mkdir(parents=True, exist_ok=True)
    cols = [c for c in ("date", "open", "high", "low", "close",
                        "adjusted_close", "volume", "vwap") if c in panel.columns]
    for ticker, grp in panel.groupby("ticker"):
        grp[cols].sort_values("date").to_csv(target_dir / f"{ticker}.csv", index=False)


# -----------------------------------------------------------------------------
# Sub-command handlers
# -----------------------------------------------------------------------------
def cmd_collect_data(args: argparse.Namespace) -> None:
    ensure_dirs()
    start = date.fromisoformat(args.start)
    end = date.fromisoformat(args.end)
    loader = FMPPriceLoader() if args.source == "fmp" else YahooFinancePriceLoader()
    raw_dir = RAW_FMP_DIR if args.source == "fmp" else RAW_YAHOO_DIR
    tickers = args.tickers or [f.stem for f in raw_dir.glob("*.csv")
                                if f.stem not in ("shares", "iwf")]
    if not tickers:
        log.warning("No tickers provided and no cache present; nothing to collect.")
        return
    written = 0
    for t in tickers:
        df = loader.get_prices(t, start, end)
        if df.empty:
            continue
        out = INTERIM_FMP_CLEAN_DIR if args.source == "fmp" else INTERIM_YAHOO_CLEAN_DIR
        out.mkdir(parents=True, exist_ok=True)
        df.to_csv(out / f"{t}.csv", index=False)
        written += 1
    log.info("Collected %s for %d tickers", args.source, written)


def cmd_validate_data(args: argparse.Namespace) -> None:
    ensure_dirs()
    fmp = _load_raw_panel("fmp")
    yahoo = _load_raw_panel("yahoo")
    if fmp.empty and yahoo.empty:
        log.error("No raw data found. Run scripts/generate_synthetic_data.py first.")
        sys.exit(1)
    report = generate_data_quality_report(fmp, yahoo)
    cfg_reports = load_validation_config()["reports"]
    write_csv(report["summary"], Path(cfg_reports["data_quality_summary"]))
    write_csv(report["issues"], Path(cfg_reports["data_quality_issues"]))
    write_csv(report["price_differences"],
              Path(cfg_reports["fmp_vs_yahoo_price_differences"]))
    write_csv(report["volume_differences"],
              Path(cfg_reports["fmp_vs_yahoo_volume_differences"]))
    write_csv(report["missing"], Path(cfg_reports["missing_data_report"]))
    print(report["summary"].to_string(index=False))


def cmd_reconcile_data(args: argparse.Namespace) -> None:
    ensure_dirs()
    fmp = _load_raw_panel("fmp")
    yahoo = _load_raw_panel("yahoo")
    panel = create_reconciled_price_panel(fmp, yahoo, drop_unreliable=True)
    _to_per_ticker(panel, PROCESSED_RECONCILED_DIR / "prices")
    summary = summarise_reconciliation(panel)
    write_csv(summary, Path(load_validation_config()["reports"]["reconciliation_summary"]))
    write_csv(panel.head(50_000),
              OUTPUTS_DIR / "reconciled_panel_sample.csv")
    log.info("Reconciled panel: %d rows", len(panel))


def _load_reconciled_panel() -> pd.DataFrame:
    prices_dir = PROCESSED_RECONCILED_DIR / "prices"
    if not prices_dir.exists() or not any(prices_dir.glob("*.csv")):
        # Fallback to raw FMP if reconciliation hasn't been run.
        return _load_raw_panel("fmp")
    return _load_from_dir(prices_dir)


def _build_rank_panel(asof: date) -> pd.DataFrame:
    panel = _load_reconciled_panel()
    shares = load_shares_outstanding()
    iwf = load_iwf()
    fmc = asof_float_market_cap(panel, asof, shares, iwf)
    mdvt = median_daily_value_traded(panel, asof)
    adv20 = adv(panel, asof, 20)
    adv60 = adv(panel, asof, 60)
    ratio = stock_liquidity_ratio(mdvt, fmc[["ticker", "avg_float_market_cap"]])
    liq = relative_liquidity(ratio, fmc[["ticker", "avg_float_market_cap"]])
    # `liq` already carries avg_float_market_cap via the ratio merge; drop it
    # before the final merge so we don't end up with avg_float_market_cap_x/_y.
    liq_keep = liq.drop(columns=[c for c in ("avg_float_market_cap",) if c in liq.columns])
    panel_rank = fmc.merge(liq_keep, on="ticker", how="left")
    panel_rank = rank_universe(panel_rank)
    panel_rank = panel_rank.merge(adv20, on="ticker", how="left")
    panel_rank = panel_rank.merge(adv60, on="ticker", how="left")
    return panel_rank


def _current_constituents_map(asof: date) -> dict[str, set[str]]:
    df = load_constituents()
    out: dict[str, set[str]] = {}
    for idx in load_indices_config()["primary_indices"]:
        sub = current_constituents(df, idx, asof)
        out[idx] = set(sub["ticker"]) if not sub.empty else set()
    return out


def _attach_flow(forecast: pd.DataFrame, rank_panel: pd.DataFrame,
                  asof: date) -> pd.DataFrame:
    cfg = load_strategy_config()["passive_aum"]
    panel = _load_reconciled_panel()
    # Reference price for flow-to-ADV.
    ref = (panel.sort_values("date").groupby("ticker").tail(1)
                .rename(columns={"adjusted_close": "ref_price"})
                [["ticker", "ref_price"]])

    out_frames = []
    for index_name in load_indices_config()["primary_indices"]:
        target = load_indices_config()["indices"][index_name]["target_count"]
        ew = expected_index_weights(rank_panel, index_name, target)
        idx_df = forecast[forecast["index"] == index_name].merge(
            ew, on=["ticker", "index"], how="left"
        )
        idx_df["prior_weight"] = idx_df["current_member"].astype(float) * idx_df["expected_weight"].fillna(0)
        flow = passive_flow(idx_df.assign(expected_weight=idx_df["expected_weight"]),
                            passive_aum_by_index=cfg)
        flow = flow.merge(ref, on="ticker", how="left")
        adv_panel = rank_panel[["ticker", "ADV_20d", "ADV_60d"]]
        flow = flow_to_adv(flow, adv_panel, price_col="ref_price")
        out_frames.append(flow)
    return pd.concat(out_frames, ignore_index=True)


def _add_history_for_forecast(df: pd.DataFrame, asof: date) -> pd.DataFrame:
    """Attach announcement/effective dates for the upcoming rebalance."""
    out_frames = []
    for index_name, grp in df.groupby("index"):
        windows = [w for w in iter_rebalance_windows(
            index_name, asof, asof + timedelta(days=120))]
        if not windows:
            continue
        win = windows[0]
        sub = grp.copy()
        sub["announcement_date"] = pd.Timestamp(win.announcement_date)
        sub["effective_date"] = pd.Timestamp(win.effective_date)
        sub["rebalance_month"] = pd.Timestamp(win.rebalance_month)
        out_frames.append(sub)
    return pd.concat(out_frames, ignore_index=True) if out_frames else df


def cmd_forecast(args: argparse.Namespace) -> None:
    ensure_dirs()
    asof = date.fromisoformat(args.asof)
    indices = ([args.index] if args.index
               else load_indices_config()["primary_indices"])
    rank_panel = _build_rank_panel(asof)
    constituents = _current_constituents_map(asof)
    rules = run_rules_engine_for_all_indices(rank_panel, constituents)
    rules = rules[rules["index"].isin(indices)].copy()
    rules["ml_probability"] = 0.5  # no model attached at the CLI layer
    rules["data_quality_score"] = 1.0
    rules["historical_hit_rate"] = 0.65
    rules = hybrid_score(rules)
    rules["confidence_bucket"] = rules["hybrid_probability"].apply(hybrid_confidence_bucket)
    rules = _add_history_for_forecast(rules, asof)
    rules = _attach_flow(rules, rank_panel, asof)
    for idx in indices:
        out = OUTPUTS_DIR / f"current_forecast_{idx}.csv"
        sub = rules[rules["index"] == idx].sort_values("hybrid_probability", ascending=False)
        write_csv(sub, out)
        print(f"Wrote {out}")
    # Excel workbook with all indices.
    write_excel({idx: rules[rules["index"] == idx] for idx in indices},
                OUTPUTS_DIR / "current_forecast_all.xlsx")


def cmd_forecast_all(args: argparse.Namespace) -> None:
    args.index = None
    cmd_forecast(args)


def cmd_backtest_rules(args: argparse.Namespace) -> None:
    ensure_dirs()
    start = date.fromisoformat(args.start)
    end = date.fromisoformat(args.end)
    labels = load_labels()
    rows = []
    for window in iter_rebalance_windows(args.index, start, end):
        asof = window.reference_date
        rank_panel = _build_rank_panel(asof)
        constituents = _current_constituents_map(asof)
        rules = run_rules_engine_for_all_indices(rank_panel, constituents)
        rules = rules[rules["index"] == args.index]
        true_actions = labels[(labels["index"] == args.index)
                              & (labels["effective_date"] == pd.Timestamp(window.effective_date))]
        truth_add = set(true_actions[true_actions["action"].isin(["Addition", "Promotion"])]["ticker"])
        truth_del = set(true_actions[true_actions["action"].isin(["Removal", "Demotion"])]["ticker"])
        pred_add = set(rules[rules["predicted_action"].isin(["Addition", "Promotion"])]["ticker"])
        pred_del = set(rules[rules["predicted_action"].isin(["Removal", "Demotion"])]["ticker"])

        def _stats(pred: set, truth: set) -> tuple[float, float, float]:
            tp = len(pred & truth)
            fp = len(pred - truth)
            fn = len(truth - pred)
            precision = tp / max(1, tp + fp)
            recall = tp / max(1, tp + fn)
            f1 = 2 * precision * recall / max(1e-9, precision + recall)
            return precision, recall, f1

        ap, ar, af1 = _stats(pred_add, truth_add)
        dp, dr, df1 = _stats(pred_del, truth_del)
        rows.append({
            "index": args.index,
            "announcement_date": window.announcement_date,
            "effective_date": window.effective_date,
            "additions_precision": ap, "additions_recall": ar, "additions_f1": af1,
            "removals_precision": dp, "removals_recall": dr, "removals_f1": df1,
            "additions_tp": len(pred_add & truth_add),
            "removals_tp": len(pred_del & truth_del),
        })
    out = pd.DataFrame(rows)
    write_csv(out, OUTPUTS_DIR / "rules_engine_accuracy.csv")
    print(out.to_string(index=False))


def _build_training_panel(index_name: str) -> tuple[pd.DataFrame, list[str]]:
    labels = load_labels()
    panel = _load_reconciled_panel()
    cfg = load_indices_config()["indices"][index_name]
    target = int(cfg["target_count"])
    windows = iter_rebalance_windows(index_name, date(2018, 1, 1),
                                     date.today() - timedelta(days=30))
    rows = []
    for window in windows:
        asof = window.reference_date
        rank_panel = _build_rank_panel(asof)
        ev = build_event_features(rank_panel, panel, asof, target)
        # Labels for this window.
        actions = labels[(labels["index"] == index_name)
                          & (labels["effective_date"] == pd.Timestamp(window.effective_date))]
        add_set = set(actions[actions["action"].isin(["Addition", "Promotion"])]["ticker"])
        del_set = set(actions[actions["action"].isin(["Removal", "Demotion"])]["ticker"])
        ev["label_add"] = ev["ticker"].isin(add_set).astype(int)
        ev["label_remove"] = ev["ticker"].isin(del_set).astype(int)
        ev["rebalance_month"] = pd.Timestamp(window.rebalance_month)
        ev["current_member"] = ev["ticker"].isin(
            current_constituents(load_constituents(), index_name, asof)["ticker"]
        ).astype(int)
        rows.append(ev)
    if not rows:
        return pd.DataFrame(), DEFAULT_FEATURES
    return pd.concat(rows, ignore_index=True), DEFAULT_FEATURES


def cmd_train_ml(args: argparse.Namespace) -> None:
    ensure_dirs()
    indices = ([args.index] if args.index
               else load_indices_config()["primary_indices"])
    metrics_all: dict[str, dict] = {}
    importances = []
    for idx in indices:
        panel, feats = _build_training_panel(idx)
        if panel.empty:
            print(f"No data to train for {idx}")
            continue
        for target in ("label_add", "label_remove"):
            if panel[target].sum() == 0:
                continue
            model = train_classifier(panel, target, feats, model=args.model)
            metrics_all[f"{idx}:{target}"] = model.metrics
            fi = feature_importance(model)
            fi["index"] = idx
            fi["target"] = target
            importances.append(fi)
    write_metrics_json(metrics_all, OUTPUTS_DIR / "model_metrics.json")
    if importances:
        imp_df = pd.concat(importances, ignore_index=True)
        write_csv(imp_df, OUTPUTS_DIR / "feature_importance.csv")
        bar_chart(imp_df.head(15), "feature", "importance",
                  FIGURES_DIR / "feature_importance.png",
                  title="Feature importance (top 15)")
    print(json.dumps(metrics_all, indent=2, default=str))


def cmd_train_ml_all(args: argparse.Namespace) -> None:
    args.index = None
    cmd_train_ml(args)


def _build_strategy_forecast() -> pd.DataFrame:
    """Use historical labels as the forecast input for the strategy backtest."""
    labels = load_labels()
    panel = _load_reconciled_panel()
    rows = []
    for index_name in load_indices_config()["primary_indices"]:
        idx_labels = labels[labels["index"] == index_name]
        for _, row in idx_labels.iterrows():
            rows.append({
                "ticker": row["ticker"],
                "index": index_name,
                "predicted_action": row["action"],
                "hybrid_probability": 0.8,
                "passive_flow_to_ADV_20d": 0.6,
                "confidence_bucket": "high conviction",
                "announcement_date": row["announcement_date"],
                "effective_date": row["effective_date"],
                "rebalance_month": pd.Timestamp(row["announcement_date"]).to_period("M").to_timestamp(),
                "volatility_63d": 0.3,
                "data_quality_flag": "none",
            })
    return pd.DataFrame(rows)


def cmd_backtest_strategy(args: argparse.Namespace) -> None:
    ensure_dirs()
    forecast = _build_strategy_forecast()
    if forecast.empty:
        log.error("No labels available for strategy backtest.")
        sys.exit(1)
    start = date.fromisoformat(args.start) if args.start else forecast["announcement_date"].min().date()
    end = date.fromisoformat(args.end) if args.end else forecast["effective_date"].max().date()
    forecast = forecast[(forecast["announcement_date"] >= pd.Timestamp(start))
                        & (forecast["effective_date"] <= pd.Timestamp(end))]

    cfg = load_strategy_config()["strategy"]
    variant = args.strategy.replace("-", "_")
    signals = build_event_signals(forecast, variant=variant)
    signals = apply_filters(signals,
                            min_prob=float(cfg["filters"]["min_hybrid_probability"]),
                            min_flow_to_adv=float(cfg["filters"]["min_flow_to_ADV"]),
                            high_conviction_only=bool(cfg["filters"]["trade_only_high_conviction"]))
    if cfg["top_k"]["enabled"]:
        signals = top_k_signals(signals, int(cfg["top_k"]["k"]))
    sized = enforce_exposure(size_positions(signals))
    positions = expand_to_positions(sized)
    prices = _load_reconciled_panel()
    result = backtest_positions(positions, prices)

    # Performance.
    daily = result["daily_returns"]
    trades = result["trades"]
    bench = load_benchmark(start, end, args.benchmark)
    if bench.empty:
        bench_returns = pd.DataFrame(columns=["date", "benchmark_return"])
    else:
        bench_returns = benchmark_daily_returns(bench)
    merged = daily.merge(bench_returns, on="date", how="left")
    metrics = summary_metrics(
        daily["return"].set_axis(daily["date"]),
        bench_returns.set_index("date")["benchmark_return"] if not bench_returns.empty else None,
    )
    write_csv(daily, OUTPUTS_DIR / "strategy_returns.csv")
    write_csv(trades, OUTPUTS_DIR / "strategy_trades.csv")
    write_csv(merged, OUTPUTS_DIR / "strategy_vs_benchmark.csv")
    write_csv(pd.DataFrame([metrics]), OUTPUTS_DIR / "strategy_performance_summary.csv")
    cumulative_return_chart(daily, bench_returns, FIGURES_DIR / "strategy_vs_asx200_buy_hold.png")
    drawdown_chart(daily, bench_returns, FIGURES_DIR / "strategy_drawdown_vs_asx200.png")
    rebalance_pnl_chart(trades, FIGURES_DIR / "rebalance_pnl.png")
    print(json.dumps(metrics, indent=2, default=str))


def cmd_compare_benchmark(args: argparse.Namespace) -> None:
    ensure_dirs()
    start = date.fromisoformat(args.start) if args.start else date(2018, 1, 1)
    end = date.fromisoformat(args.end) if args.end else date.today()
    panel = load_benchmark(start, end, args.benchmark)
    if panel.empty:
        log.error("No benchmark data available.")
        sys.exit(1)
    out = buy_and_hold_returns(panel)
    write_csv(out, OUTPUTS_DIR / "buy_and_hold_returns.csv")
    print(out.tail().to_string(index=False))


def cmd_dashboard(args: argparse.Namespace) -> None:
    try:
        import subprocess
        subprocess.run([sys.executable, "-m", "streamlit", "run",
                        str(Path(__file__).parent / "reporting" / "dashboard.py")],
                       check=False)
    except FileNotFoundError:
        log.error("Streamlit not installed. pip install asx-index-rebalance-forecast[dashboard]")


# -----------------------------------------------------------------------------
# Argument parser
# -----------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="asxrebalance",
                                description="S&P/ASX index rebalance forecast and backtest.")
    sub = p.add_subparsers(dest="command", required=True)

    c = sub.add_parser("collect-data")
    c.add_argument("--source", choices=("fmp", "yahoo"), required=True)
    c.add_argument("--start", required=True)
    c.add_argument("--end", required=True)
    c.add_argument("--tickers", nargs="*")
    c.set_defaults(func=cmd_collect_data)

    c = sub.add_parser("validate-data")
    c.add_argument("--start")
    c.add_argument("--end")
    c.set_defaults(func=cmd_validate_data)

    c = sub.add_parser("reconcile-data")
    c.add_argument("--start")
    c.add_argument("--end")
    c.set_defaults(func=cmd_reconcile_data)

    c = sub.add_parser("forecast")
    c.add_argument("--index", choices=("ASX50", "ASX100", "ASX200"), required=True)
    c.add_argument("--asof", required=True)
    c.set_defaults(func=cmd_forecast)

    c = sub.add_parser("forecast-all")
    c.add_argument("--asof", required=True)
    c.set_defaults(func=cmd_forecast_all, index=None)

    c = sub.add_parser("backtest-rules")
    c.add_argument("--index", choices=("ASX50", "ASX100", "ASX200"), required=True)
    c.add_argument("--start", required=True)
    c.add_argument("--end", required=True)
    c.set_defaults(func=cmd_backtest_rules)

    c = sub.add_parser("train-ml")
    c.add_argument("--index", choices=("ASX50", "ASX100", "ASX200"), required=True)
    c.add_argument("--model", choices=("logistic", "random_forest", "xgboost"),
                   default="logistic")
    c.set_defaults(func=cmd_train_ml)

    c = sub.add_parser("train-ml-all")
    c.add_argument("--model", choices=("logistic", "random_forest", "xgboost"),
                   default="logistic")
    c.set_defaults(func=cmd_train_ml_all, index=None)

    c = sub.add_parser("backtest-strategy")
    c.add_argument("--strategy", default="announcement-long-short",
                   choices=("announcement-long-short", "pre-announcement",
                            "additions-only", "market-neutral",
                            "flow-pressure", "top-k"))
    c.add_argument("--start")
    c.add_argument("--end")
    c.add_argument("--benchmark", default="STW.AX")
    c.set_defaults(func=cmd_backtest_strategy)

    c = sub.add_parser("compare-benchmark")
    c.add_argument("--benchmark", default="STW.AX")
    c.add_argument("--start")
    c.add_argument("--end")
    c.set_defaults(func=cmd_compare_benchmark)

    c = sub.add_parser("dashboard")
    c.set_defaults(func=cmd_dashboard)

    return p


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
