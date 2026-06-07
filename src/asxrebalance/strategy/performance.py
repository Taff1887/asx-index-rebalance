"""Performance metrics for the backtest."""

from __future__ import annotations

import numpy as np
import pandas as pd


def _ann_factor(periods_per_year: int = 252) -> int:
    return periods_per_year


def cagr(daily_returns: pd.Series, periods_per_year: int = 252) -> float:
    if daily_returns.empty:
        return float("nan")
    total = float((1 + daily_returns.fillna(0)).prod())
    years = len(daily_returns) / periods_per_year
    if years <= 0 or total <= 0:
        return float("nan")
    return total ** (1 / years) - 1


def volatility(daily_returns: pd.Series, periods_per_year: int = 252) -> float:
    if daily_returns.empty:
        return float("nan")
    return float(daily_returns.std() * np.sqrt(periods_per_year))


def sharpe(daily_returns: pd.Series, risk_free: float = 0.0,
           periods_per_year: int = 252) -> float:
    vol = volatility(daily_returns, periods_per_year)
    if vol == 0 or np.isnan(vol):
        return float("nan")
    excess = daily_returns - risk_free / periods_per_year
    return float(excess.mean() * periods_per_year / vol)


def sortino(daily_returns: pd.Series, risk_free: float = 0.0,
            periods_per_year: int = 252) -> float:
    downside = daily_returns[daily_returns < 0]
    if downside.empty:
        return float("inf")
    dd_vol = downside.std() * np.sqrt(periods_per_year)
    if dd_vol == 0:
        return float("inf")
    excess = daily_returns - risk_free / periods_per_year
    return float(excess.mean() * periods_per_year / dd_vol)


def max_drawdown(daily_returns: pd.Series) -> float:
    if daily_returns.empty:
        return float("nan")
    cum = (1 + daily_returns.fillna(0)).cumprod()
    peak = cum.cummax()
    dd = cum / peak - 1
    return float(dd.min())


def calmar(daily_returns: pd.Series, periods_per_year: int = 252) -> float:
    mdd = max_drawdown(daily_returns)
    if mdd == 0 or np.isnan(mdd):
        return float("nan")
    return cagr(daily_returns, periods_per_year) / abs(mdd)


def hit_rate(daily_returns: pd.Series) -> float:
    if daily_returns.empty:
        return float("nan")
    return float((daily_returns > 0).mean())


def beta_alpha(strategy: pd.Series, benchmark: pd.Series,
               periods_per_year: int = 252) -> tuple[float, float]:
    df = pd.concat([strategy, benchmark], axis=1, join="inner").dropna()
    if df.empty:
        return float("nan"), float("nan")
    cov = df.cov().iloc[0, 1]
    var = df.iloc[:, 1].var()
    beta = float(cov / var) if var else float("nan")
    alpha_daily = df.iloc[:, 0].mean() - beta * df.iloc[:, 1].mean()
    alpha_ann = float(alpha_daily * periods_per_year)
    return beta, alpha_ann


def tracking_error(strategy: pd.Series, benchmark: pd.Series,
                   periods_per_year: int = 252) -> float:
    diff = (strategy - benchmark).dropna()
    if diff.empty:
        return float("nan")
    return float(diff.std() * np.sqrt(periods_per_year))


def information_ratio(strategy: pd.Series, benchmark: pd.Series,
                      periods_per_year: int = 252) -> float:
    te = tracking_error(strategy, benchmark, periods_per_year)
    if te == 0 or np.isnan(te):
        return float("nan")
    diff = (strategy - benchmark).dropna()
    return float(diff.mean() * periods_per_year / te)


def monthly_returns(daily_returns: pd.Series) -> pd.DataFrame:
    if daily_returns.empty:
        return pd.DataFrame()
    m = (1 + daily_returns).resample("M").prod() - 1
    return m.to_frame("return")


def summary_metrics(daily_returns: pd.Series,
                    benchmark_returns: pd.Series | None = None) -> dict:
    out = {
        "cagr": cagr(daily_returns),
        "vol": volatility(daily_returns),
        "sharpe": sharpe(daily_returns),
        "sortino": sortino(daily_returns),
        "max_drawdown": max_drawdown(daily_returns),
        "calmar": calmar(daily_returns),
        "hit_rate": hit_rate(daily_returns),
    }
    if benchmark_returns is not None:
        b, a = beta_alpha(daily_returns, benchmark_returns)
        out["beta"] = b
        out["alpha"] = a
        out["tracking_error"] = tracking_error(daily_returns, benchmark_returns)
        out["information_ratio"] = information_ratio(daily_returns, benchmark_returns)
    return out


__all__ = [
    "cagr", "volatility", "sharpe", "sortino", "max_drawdown", "calmar",
    "hit_rate", "beta_alpha", "tracking_error", "information_ratio",
    "monthly_returns", "summary_metrics",
]
