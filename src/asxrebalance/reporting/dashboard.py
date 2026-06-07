"""Streamlit dashboard.

Run with: ``streamlit run src/asxrebalance/reporting/dashboard.py``
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

try:
    import streamlit as st
except ImportError:  # pragma: no cover - optional dep
    st = None  # type: ignore

from ..paths import FIGURES_DIR, OUTPUTS_DIR


def _read(path: Path) -> pd.DataFrame:
    if path.exists():
        return pd.read_csv(path)
    return pd.DataFrame()


def render() -> None:  # pragma: no cover - streamlit
    if st is None:
        raise RuntimeError(
            "Streamlit is not installed. `pip install asx-index-rebalance-forecast[dashboard]`"
        )

    st.set_page_config(page_title="ASX Index Rebalance Forecast", layout="wide")
    st.title("S&P/ASX Index Rebalance Forecast")
    st.caption(
        "Forecasts ASX 50/100/200 rebalances using a rules + ML hybrid, "
        "estimates passive flow pressure, and backtests a tradeable strategy."
    )

    section = st.sidebar.radio("Section", ["Data", "Forecast", "Strategy", "Model"])

    if section == "Data":
        st.header("Data quality")
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Quality summary")
            st.dataframe(_read(OUTPUTS_DIR / "data_quality_summary.csv"))
        with col2:
            st.subheader("Reconciliation summary")
            st.dataframe(_read(OUTPUTS_DIR / "reconciliation_summary.csv"))
        st.subheader("FMP vs Yahoo price differences")
        st.dataframe(_read(OUTPUTS_DIR / "fmp_vs_yahoo_price_differences.csv").head(2000))
        st.subheader("Missing observations")
        st.dataframe(_read(OUTPUTS_DIR / "missing_data_report.csv").head(2000))

    elif section == "Forecast":
        st.header("Current rebalance forecast")
        for idx in ("ASX50", "ASX100", "ASX200"):
            st.subheader(idx)
            st.dataframe(_read(OUTPUTS_DIR / f"current_forecast_{idx}.csv"))

    elif section == "Strategy":
        st.header("Strategy backtest")
        chart = FIGURES_DIR / "strategy_vs_asx200_buy_hold.png"
        if chart.exists():
            st.image(str(chart))
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Performance summary")
            st.dataframe(_read(OUTPUTS_DIR / "strategy_performance_summary.csv"))
        with col2:
            st.subheader("Strategy vs benchmark")
            st.dataframe(_read(OUTPUTS_DIR / "strategy_vs_benchmark.csv"))
        st.subheader("Trades")
        st.dataframe(_read(OUTPUTS_DIR / "strategy_trades.csv"))
        dd = FIGURES_DIR / "strategy_drawdown_vs_asx200.png"
        if dd.exists():
            st.image(str(dd))

    else:
        st.header("Model")
        st.subheader("Rules-engine accuracy")
        st.dataframe(_read(OUTPUTS_DIR / "rules_engine_accuracy.csv"))
        st.subheader("Model metrics")
        st.dataframe(_read(OUTPUTS_DIR / "model_metrics.json").T if (OUTPUTS_DIR / "model_metrics.json").exists() else pd.DataFrame())
        fi = FIGURES_DIR / "feature_importance.png"
        if fi.exists():
            st.image(str(fi))
        cal = FIGURES_DIR / "calibration.png"
        if cal.exists():
            st.image(str(cal))


if __name__ == "__main__":  # pragma: no cover
    render()
