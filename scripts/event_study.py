"""Event study on the 74 real S&P/ASX 200 rebalance events.

For each event, compute the abnormal return (ticker return minus ASX 200
benchmark return) for each business day from t-10 to t+30, where t=0 is the
announcement date. Average across events to see the average price path.

Output:
    outputs/event_study_table.csv   - mean cumulative abnormal return (CAR)
                                        per relative day, for adds and removes
    docs/figures/event_study_research.png - the chart used to design the strategy

The goal is to identify empirically:
    * Where the alpha materializes (which days)
    * Whether adds and removes behave differently (academic literature says yes)
    * Optimal entry / exit days for the strategy
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from asxrebalance.paths import (  # noqa: E402
    OUTPUTS_DIR,
    PROCESSED_BENCHMARK_DIR,
    PROCESSED_LABELS_DIR,
    PROCESSED_RECONCILED_DIR,
    RAW_YAHOO_DIR,
    REPO_ROOT,
)

DOCS_FIGURES = REPO_ROOT / "docs" / "figures"
DOCS_FIGURES.mkdir(parents=True, exist_ok=True)

WINDOW_PRE = 10    # business days before announcement
WINDOW_POST = 30   # business days after announcement


def _load_price_series(ticker: str) -> pd.DataFrame | None:
    for d in (PROCESSED_RECONCILED_DIR / "prices", RAW_YAHOO_DIR):
        p = d / f"{ticker}.csv"
        if p.exists():
            df = pd.read_csv(p, parse_dates=["date"]).sort_values("date").reset_index(drop=True)
            return df
    return None


def _load_benchmark_returns() -> pd.Series:
    f = PROCESSED_BENCHMARK_DIR / "asx200_benchmark.csv"
    df = pd.read_csv(f, parse_dates=["date"]).sort_values("date").set_index("date")
    return df["adjusted_close"].pct_change().rename("bench_ret")


def event_window_returns(df: pd.DataFrame, ann_date: pd.Timestamp,
                          bench_returns: pd.Series) -> pd.Series | None:
    """Return abnormal daily returns indexed by relative day (-10..+30)."""
    if df.empty:
        return None
    df = df.set_index("date")
    if ann_date not in df.index:
        on_or_after = df.index[df.index >= ann_date]
        if len(on_or_after) == 0:
            return None
        ann_date = on_or_after[0]
    anchor_pos = int(df.index.get_loc(ann_date))
    lo = max(0, anchor_pos - WINDOW_PRE)
    hi = min(len(df), anchor_pos + WINDOW_POST + 1)
    window = df.iloc[lo:hi]
    if len(window) < 5:
        return None

    stock_ret = window["adjusted_close"].pct_change()
    bench = bench_returns.reindex(window.index)
    abn = stock_ret - bench
    abn.index = range(lo - anchor_pos, hi - anchor_pos)
    return abn


def main() -> None:
    labels = pd.read_csv(PROCESSED_LABELS_DIR / "rebalance_labels.csv",
                          parse_dates=["announcement_date", "effective_date"])
    bench = _load_benchmark_returns()

    per_event = {"Addition": [], "Removal": []}
    for _, evt in labels.iterrows():
        df = _load_price_series(evt["ticker"])
        if df is None:
            continue
        abn = event_window_returns(df, evt["announcement_date"], bench)
        if abn is None:
            continue
        per_event[evt["action"]].append(abn)

    print(f"Additions with full price coverage : {len(per_event['Addition'])}")
    print(f"Removals  with full price coverage : {len(per_event['Removal'])}")

    rows = []
    for action, series_list in per_event.items():
        if not series_list:
            continue
        df = pd.concat(series_list, axis=1)
        df = df.dropna(axis=0, how="all")
        mean_daily = df.mean(axis=1).rename(f"{action.lower()}_mean_daily")
        car = mean_daily.cumsum().rename(f"{action.lower()}_cum_ar")
        for day, val in mean_daily.items():
            rows.append({"action": action, "rel_day": int(day),
                          "n_events": int(df.loc[day].notna().sum()),
                          "mean_daily_abnormal_pct": float(val) * 100,
                          "cum_abnormal_pct": float(car[day]) * 100})

    table = pd.DataFrame(rows).sort_values(["action", "rel_day"])
    table.to_csv(OUTPUTS_DIR / "event_study_table.csv", index=False)

    # ---- Chart ----------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), sharey=False)

    for ax, action, color, label_long in (
        (axes[0], "Addition", "#2e7d32", "Additions (long the stock)"),
        (axes[1], "Removal",  "#c62828", "Removals (long shows the actual stock move; short profits from the inverse)"),
    ):
        sub = table[table["action"] == action]
        ax.plot(sub["rel_day"], sub["cum_abnormal_pct"], color=color, linewidth=2,
                marker="o", markersize=3, label="Cumulative abnormal return (%)")
        ax.axvline(0, color="black", linestyle="--", linewidth=1, label="Announcement (t=0)")
        # Effective date is 10 business days after announcement.
        ax.axvline(10, color="grey", linestyle=":", linewidth=1.2, label="Effective (t=+10)")
        ax.axhline(0, color="black", linewidth=0.5)
        ax.set_title(f"{action}s — n={int(sub['n_events'].max())}")
        ax.set_xlabel("Business days from announcement")
        ax.set_ylabel("Cumulative abnormal return vs ASX 200 (%)")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best", fontsize=9)

    plt.suptitle("Event study — real S&P/ASX 200 additions vs removals",
                 fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out_png = DOCS_FIGURES / "event_study_research.png"
    fig.savefig(out_png, dpi=140)
    plt.close(fig)
    print(f"\n-> {out_png}")

    # ---- Print the table for the README ---------------------------------------
    print("\nKey cumulative abnormal returns (CAR) for ADDITIONS:")
    add_table = table[table["action"] == "Addition"]
    print(add_table.set_index("rel_day")[["cum_abnormal_pct"]].round(2).T.to_string())

    print("\nKey cumulative abnormal returns (CAR) for REMOVALS:")
    rem_table = table[table["action"] == "Removal"]
    print(rem_table.set_index("rel_day")[["cum_abnormal_pct"]].round(2).T.to_string())

    print("\nRecommendation:")
    add_car = add_table.set_index("rel_day")["cum_abnormal_pct"]
    rem_car = rem_table.set_index("rel_day")["cum_abnormal_pct"]
    print(f"  Add peak CAR  : {add_car.max():+.2f}% on day {int(add_car.idxmax()):+d}")
    print(f"  Add trough    : {add_car.min():+.2f}% on day {int(add_car.idxmin()):+d}")
    print(f"  Rem peak CAR  : {rem_car.max():+.2f}% on day {int(rem_car.idxmax()):+d}")
    print(f"  Rem trough    : {rem_car.min():+.2f}% on day {int(rem_car.idxmin()):+d}")
    print(f"  Optimal long  on additions: buy day {int(add_car.idxmin())},"
          f" sell day {int(add_car.idxmax())},"
          f" gross alpha = {add_car.max() - add_car.min():+.2f}%")
    print(f"  Optimal short on removals : short day {int(rem_car.idxmax())},"
          f" cover day {int(rem_car.idxmin())},"
          f" gross alpha = {rem_car.max() - rem_car.min():+.2f}%")


if __name__ == "__main__":
    main()
