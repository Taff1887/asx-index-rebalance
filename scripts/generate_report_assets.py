"""Generate every chart referenced by the README report.

Reads the artefacts produced by the CLI pipeline (validation, reconciliation,
forecast, ML training, strategy backtest) and writes a complete chart set to
``docs/figures/`` so the GitHub README can embed them directly.

Run after the pipeline:

    python scripts/generate_synthetic_data.py
    python -m asxrebalance validate-data --start 2020-01-01 --end 2026-06-01
    python -m asxrebalance reconcile-data --start 2020-01-01 --end 2026-06-01
    python -m asxrebalance forecast-all --asof 2026-06-01
    python -m asxrebalance backtest-rules --index ASX200 --start 2021-01-01 --end 2024-12-31
    python -m asxrebalance backtest-rules --index ASX100 --start 2021-01-01 --end 2024-12-31
    python -m asxrebalance backtest-rules --index ASX50 --start 2021-01-01 --end 2024-12-31
    python -m asxrebalance train-ml --index ASX200
    python -m asxrebalance backtest-strategy --strategy announcement-long-short \
        --start 2021-01-01 --end 2024-12-31
    python scripts/generate_report_assets.py
"""

from __future__ import annotations

import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from asxrebalance.calendar import iter_rebalance_windows  # noqa: E402
from asxrebalance.data.announcements import load_labels  # noqa: E402
from asxrebalance.data.benchmark import buy_and_hold_returns, load_benchmark  # noqa: E402
from asxrebalance.paths import (  # noqa: E402
    FIGURES_DIR,
    OUTPUTS_DIR,
    PROCESSED_RECONCILED_DIR,
    REPO_ROOT,
)

DOCS_FIGURES = REPO_ROOT / "docs" / "figures"
DOCS_FIGURES.mkdir(parents=True, exist_ok=True)

PALETTE = {
    "strategy": "#1f4e79",
    "benchmark": "#a6a6a6",
    "addition": "#2e7d32",
    "removal": "#c62828",
    "high": "#c62828",
    "medium": "#ef6c00",
    "low": "#fbc02d",
    "none": "#9e9e9e",
}
FIGSIZE = (10, 6)


def _save(fig, name: str) -> Path:
    p = DOCS_FIGURES / name
    fig.tight_layout()
    fig.savefig(p, dpi=140)
    plt.close(fig)
    print(f"  -> {p.relative_to(REPO_ROOT)}")
    return p


# ---------------------------------------------------------------------------
# Step 0: copy charts the CLI already produced
# ---------------------------------------------------------------------------
def copy_existing_charts() -> None:
    for src in FIGURES_DIR.glob("*.png"):
        shutil.copy2(src, DOCS_FIGURES / src.name)
        print(f"  copied {src.name}")


# ---------------------------------------------------------------------------
# Step 1: FMP vs Yahoo discrepancy charts
# ---------------------------------------------------------------------------
def fmp_vs_yahoo_charts() -> None:
    price = pd.read_csv(OUTPUTS_DIR / "fmp_vs_yahoo_price_differences.csv")
    close = price[price["field"] == "close"]
    fig, ax = plt.subplots(figsize=FIGSIZE)
    sev_counts = close["severity"].value_counts().reindex(
        ["none", "low", "medium", "high", "missing"], fill_value=0
    )
    colors = [PALETTE.get(s, "#777") for s in sev_counts.index]
    ax.bar(sev_counts.index, sev_counts.values, color=colors)
    ax.set_title("FMP vs Yahoo close-price discrepancy severity")
    ax.set_ylabel("Observations")
    ax.set_yscale("log")
    for i, v in enumerate(sev_counts.values):
        ax.text(i, v, f"{int(v):,}", ha="center", va="bottom")
    _save(fig, "fmp_vs_yahoo_price_discrepancy.png")

    vol = pd.read_csv(OUTPUTS_DIR / "fmp_vs_yahoo_volume_differences.csv")
    fig, ax = plt.subplots(figsize=FIGSIZE)
    sev_counts = vol["severity"].value_counts().reindex(
        ["none", "low", "medium", "high", "missing"], fill_value=0
    )
    colors = [PALETTE.get(s, "#777") for s in sev_counts.index]
    ax.bar(sev_counts.index, sev_counts.values, color=colors)
    ax.set_title("FMP vs Yahoo volume discrepancy severity")
    ax.set_ylabel("Observations")
    ax.set_yscale("log")
    for i, v in enumerate(sev_counts.values):
        ax.text(i, v, f"{int(v):,}", ha="center", va="bottom")
    _save(fig, "fmp_vs_yahoo_volume_discrepancy.png")


# ---------------------------------------------------------------------------
# Step 2: Data quality exclusions over time
# ---------------------------------------------------------------------------
def data_quality_over_time() -> None:
    issues = pd.read_csv(OUTPUTS_DIR / "data_quality_issues.csv", parse_dates=["date"])
    if "date" not in issues.columns:
        return
    issues = issues.dropna(subset=["date"])
    issues["month"] = issues["date"].dt.to_period("M").dt.to_timestamp()
    pivot = (issues.groupby(["month", "severity"]).size().unstack(fill_value=0))
    if pivot.empty:
        return
    fig, ax = plt.subplots(figsize=FIGSIZE)
    bottoms = np.zeros(len(pivot))
    for sev in ["none", "low", "medium", "high", "missing"]:
        if sev not in pivot.columns:
            continue
        ax.bar(pivot.index, pivot[sev], bottom=bottoms,
               label=sev, color=PALETTE.get(sev, "#777"), width=20)
        bottoms = bottoms + pivot[sev].values
    ax.set_title("Data-quality exclusions by month")
    ax.set_ylabel("Observations flagged")
    ax.legend(title="Severity")
    fig.autofmt_xdate()
    _save(fig, "data_quality_exclusions.png")


# ---------------------------------------------------------------------------
# Step 3: Rules-engine accuracy charts
# ---------------------------------------------------------------------------
def hit_rate_charts() -> None:
    rows = []
    for idx in ("ASX50", "ASX100", "ASX200"):
        f = OUTPUTS_DIR / f"rules_engine_accuracy_{idx}.csv"
        # Backwards compat — also accept the single combined file.
        if not f.exists():
            f = OUTPUTS_DIR / "rules_engine_accuracy.csv"
        if not f.exists():
            continue
        df = pd.read_csv(f)
        if "index" in df.columns:
            df = df[df["index"] == idx]
        if df.empty:
            continue
        rows.append({
            "index": idx,
            "additions_precision": df["additions_precision"].mean(),
            "additions_recall": df["additions_recall"].mean(),
            "additions_f1": df["additions_f1"].mean(),
            "removals_precision": df["removals_precision"].mean(),
            "removals_recall": df["removals_recall"].mean(),
            "removals_f1": df["removals_f1"].mean(),
        })
    if not rows:
        return
    summary = pd.DataFrame(rows)

    fig, ax = plt.subplots(figsize=FIGSIZE)
    x = np.arange(len(summary))
    w = 0.4
    ax.bar(x - w / 2, summary["additions_f1"], w, label="Additions F1", color=PALETTE["addition"])
    ax.bar(x + w / 2, summary["removals_f1"], w, label="Removals F1", color=PALETTE["removal"])
    ax.set_xticks(x)
    ax.set_xticklabels(summary["index"])
    ax.set_ylabel("Mean F1 across rebalances")
    ax.set_title("Rules-engine F1 by index")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    _save(fig, "hit_rate_by_index.png")

    metrics = ["precision", "recall", "f1"]
    fig, ax = plt.subplots(figsize=FIGSIZE)
    add_vals = [summary[f"additions_{m}"].mean() for m in metrics]
    del_vals = [summary[f"removals_{m}"].mean() for m in metrics]
    x = np.arange(len(metrics))
    ax.bar(x - 0.2, add_vals, 0.4, label="Additions", color=PALETTE["addition"])
    ax.bar(x + 0.2, del_vals, 0.4, label="Removals", color=PALETTE["removal"])
    ax.set_xticks(x)
    ax.set_xticklabels([m.title() for m in metrics])
    ax.set_title("Rules-engine accuracy by action type (average across indices)")
    ax.set_ylabel("Metric value")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    _save(fig, "hit_rate_by_action.png")


# ---------------------------------------------------------------------------
# Step 4: Rank distribution chart
# ---------------------------------------------------------------------------
def rank_distribution_chart() -> None:
    fc = OUTPUTS_DIR / "current_forecast_ASX200.csv"
    if not fc.exists():
        return
    df = pd.read_csv(fc)
    fig, ax = plt.subplots(figsize=FIGSIZE)
    members = df[df["current_member"] == True]  # noqa: E712
    non_members = df[df["current_member"] == False]  # noqa: E712
    bins = np.linspace(df["fmc_rank"].min(), df["fmc_rank"].max(), 30)
    ax.hist(members["fmc_rank"].dropna(), bins=bins, alpha=0.6,
            label="Current members", color=PALETTE["strategy"])
    ax.hist(non_members["fmc_rank"].dropna(), bins=bins, alpha=0.6,
            label="Non-members", color=PALETTE["benchmark"])
    ax.axvline(200, color="black", linestyle="--", linewidth=1, label="ASX 200 cutoff")
    ax.axvline(179, color=PALETTE["addition"], linestyle="--", linewidth=1,
               label="Addition buffer 179")
    ax.axvline(221, color=PALETTE["removal"], linestyle="--", linewidth=1,
               label="Deletion buffer 221")
    ax.set_xlabel("Float-adjusted market-cap rank")
    ax.set_ylabel("Number of stocks")
    ax.set_title("Rank distribution around the ASX 200 cutoff")
    ax.legend()
    _save(fig, "rank_distribution.png")


# ---------------------------------------------------------------------------
# Step 5: Event study around announcement and effective dates
# ---------------------------------------------------------------------------
def _build_event_study(action_group: str, window: tuple[int, int] = (-10, 10)) -> pd.DataFrame:
    labels = load_labels()
    if labels.empty:
        return pd.DataFrame()
    prices_dir = PROCESSED_RECONCILED_DIR / "prices"
    if not prices_dir.exists():
        return pd.DataFrame()
    bench = load_benchmark(pd.Timestamp("2020-01-01").date(),
                            pd.Timestamp("2026-06-01").date())
    if bench.empty:
        return pd.DataFrame()
    bench_ret = buy_and_hold_returns(bench).set_index("date")["return"]

    if action_group == "additions":
        sub = labels[labels["action"].isin(["Addition", "Promotion"])]
    else:
        sub = labels[labels["action"].isin(["Removal", "Demotion"])]
    rows = []
    for _, evt in sub.iterrows():
        f = prices_dir / f"{evt['ticker']}.csv"
        if not f.exists():
            continue
        px = pd.read_csv(f, parse_dates=["date"])
        if px.empty or "adjusted_close" not in px.columns:
            continue
        px = px.sort_values("date")
        px["ret"] = px["adjusted_close"].pct_change()
        ann = pd.Timestamp(evt["announcement_date"])
        idx_offset = px["date"].sub(ann).dt.days
        for k in range(window[0], window[1] + 1):
            day = px.loc[idx_offset == k]
            if day.empty:
                continue
            d = day["date"].iloc[0]
            stock_ret = day["ret"].iloc[0]
            bench_d = bench_ret.get(d, np.nan)
            if pd.notna(bench_d) and pd.notna(stock_ret):
                rows.append({"k": k, "abnormal": stock_ret - bench_d,
                             "stock": stock_ret, "bench": bench_d})
    return pd.DataFrame(rows)


def event_study_charts() -> None:
    for grp, color in (("additions", PALETTE["addition"]), ("removals", PALETTE["removal"])):
        df = _build_event_study(grp)
        if df.empty:
            continue
        agg = df.groupby("k").agg(
            mean_abn=("abnormal", "mean"),
            n=("abnormal", "size"),
        ).reset_index()
        agg["cum_abn"] = agg["mean_abn"].cumsum()

        fig, ax = plt.subplots(figsize=FIGSIZE)
        ax.plot(agg["k"], agg["cum_abn"] * 100, color=color, linewidth=2,
                label=f"Mean cumulative abnormal return ({grp})")
        ax.axvline(0, color="black", linestyle="--", linewidth=1, label="Announcement (t=0)")
        ax.set_title(f"Event study — {grp.title()} (n={int(agg['n'].mean())} events/day avg)")
        ax.set_xlabel("Trading days from announcement")
        ax.set_ylabel("Cumulative abnormal return (%)")
        ax.grid(True, alpha=0.3)
        ax.legend()
        _save(fig, f"event_study_{grp}.png")


# ---------------------------------------------------------------------------
# Step 6: Flow-to-ADV vs return scatter
# ---------------------------------------------------------------------------
def flow_to_adv_chart() -> None:
    fc_files = list(OUTPUTS_DIR.glob("current_forecast_ASX*.csv"))
    if not fc_files:
        return
    frames = [pd.read_csv(f) for f in fc_files]
    df = pd.concat(frames, ignore_index=True)
    if "passive_flow_to_ADV_20d" not in df.columns:
        return
    df = df[df["predicted_action"].isin(["Addition", "Promotion", "Removal", "Demotion"])]
    if df.empty:
        return
    fig, ax = plt.subplots(figsize=FIGSIZE)
    is_long = df["predicted_action"].isin(["Addition", "Promotion"])
    ax.scatter(
        df.loc[is_long, "passive_flow_to_ADV_20d"].abs(),
        df.loc[is_long, "hybrid_probability"],
        s=50, alpha=0.6, color=PALETTE["addition"], label="Additions / Promotions",
    )
    ax.scatter(
        df.loc[~is_long, "passive_flow_to_ADV_20d"].abs(),
        df.loc[~is_long, "hybrid_probability"],
        s=50, alpha=0.6, color=PALETTE["removal"], label="Removals / Demotions",
    )
    ax.set_xlabel("Passive flow to 20-day ADV (absolute)")
    ax.set_ylabel("Hybrid forecast probability")
    ax.set_title("Flow pressure vs forecast probability")
    ax.legend()
    ax.grid(True, alpha=0.3)
    _save(fig, "flow_to_adv_vs_return.png")


# ---------------------------------------------------------------------------
# Step 7: Monthly return heatmap
# ---------------------------------------------------------------------------
def monthly_heatmap_chart() -> None:
    f = OUTPUTS_DIR / "strategy_returns.csv"
    if not f.exists():
        return
    df = pd.read_csv(f, parse_dates=["date"])
    if df.empty or "return" not in df.columns:
        return
    s = df.set_index("date")["return"].dropna()
    monthly = (1 + s).resample("ME").prod() - 1
    pivot = pd.DataFrame({"y": monthly.index.year, "m": monthly.index.month,
                          "r": monthly.values}).pivot(index="y", columns="m", values="r")
    fig, ax = plt.subplots(figsize=FIGSIZE)
    im = ax.imshow(pivot.values, cmap="RdYlGn", aspect="auto",
                   vmin=-0.05, vmax=0.05)
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    ax.set_title("Monthly strategy returns (%)")
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            v = pivot.values[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v*100:.1f}", ha="center", va="center",
                        fontsize=8, color="black")
    fig.colorbar(im, ax=ax, format="%+.0f%%")
    _save(fig, "monthly_return_heatmap.png")


# ---------------------------------------------------------------------------
# Step 8: Cumulative return chart + drawdown re-rendered into docs
# ---------------------------------------------------------------------------
def strategy_summary_chart() -> None:
    f = OUTPUTS_DIR / "strategy_vs_benchmark.csv"
    if not f.exists():
        return
    df = pd.read_csv(f, parse_dates=["date"]).sort_values("date")
    fig, ax = plt.subplots(figsize=FIGSIZE)
    if "return" in df.columns:
        cum = (1 + df["return"].fillna(0)).cumprod() - 1
        ax.plot(df["date"], cum * 100, color=PALETTE["strategy"], linewidth=2, label="Strategy")
    if "benchmark_return" in df.columns:
        cum_b = (1 + df["benchmark_return"].fillna(0)).cumprod() - 1
        ax.plot(df["date"], cum_b * 100, color=PALETTE["benchmark"], linewidth=2,
                linestyle="--", label="ASX 200 buy-and-hold")
    ax.set_title("Cumulative return: strategy vs ASX 200 buy-and-hold")
    ax.set_ylabel("Cumulative return (%)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    _save(fig, "strategy_vs_asx200_buy_hold.png")

    fig, ax = plt.subplots(figsize=FIGSIZE)
    for col, color, label in (("return", PALETTE["strategy"], "Strategy"),
                               ("benchmark_return", PALETTE["benchmark"], "ASX 200")):
        if col not in df.columns:
            continue
        cum = (1 + df[col].fillna(0)).cumprod()
        dd = (cum / cum.cummax() - 1) * 100
        ax.plot(df["date"], dd, color=color, label=label, linewidth=2)
    ax.set_title("Drawdowns: strategy vs ASX 200")
    ax.set_ylabel("Drawdown (%)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    _save(fig, "strategy_drawdown_vs_asx200.png")


# ---------------------------------------------------------------------------
# Step 9: Calibration chart (skip silently if no ML probabilities available)
# ---------------------------------------------------------------------------
def calibration_placeholder() -> None:
    # Calibration requires probabilities + labels which the synthetic pipeline
    # does not produce by default — render an explanatory placeholder so the
    # README still has a visual.
    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.plot([0, 1], [0, 1], "--", color="grey", label="Perfect calibration")
    rng = np.random.default_rng(0)
    bins = np.linspace(0.05, 0.95, 10)
    empirical = bins + rng.normal(0, 0.04, size=len(bins))
    sizes = rng.integers(low=10, high=200, size=len(bins))
    ax.scatter(bins, empirical, s=sizes, color=PALETTE["strategy"], alpha=0.6,
               label="Bucketed predictions (illustrative)")
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Empirical hit rate")
    ax.set_title("Forecast probability calibration (illustrative)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    _save(fig, "calibration.png")


def main() -> None:
    print("Copying existing CLI charts...")
    copy_existing_charts()
    print("Generating FMP vs Yahoo discrepancy charts...")
    fmp_vs_yahoo_charts()
    print("Generating data quality exclusions chart...")
    data_quality_over_time()
    print("Generating hit-rate charts...")
    hit_rate_charts()
    print("Generating rank distribution chart...")
    rank_distribution_chart()
    print("Generating event study charts...")
    event_study_charts()
    print("Generating flow-to-ADV scatter...")
    flow_to_adv_chart()
    print("Generating monthly heatmap...")
    monthly_heatmap_chart()
    print("Re-rendering strategy charts to docs/...")
    strategy_summary_chart()
    print("Generating calibration placeholder...")
    calibration_placeholder()
    print("Done.")


if __name__ == "__main__":
    main()
