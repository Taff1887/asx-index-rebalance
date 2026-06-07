"""Backtest execution engine.

The engine takes a daily positions panel and a price panel, applies transaction
costs at entry/exit, and returns a daily P&L series along with a trade-level
ledger. Deliberately simple and easy to audit.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .costs import borrow_cost_for_day, per_trade_cost


def _prepare_prices(prices: pd.DataFrame) -> pd.DataFrame:
    p = prices[["date", "ticker", "adjusted_close"]].copy()
    p["return"] = p.groupby("ticker")["adjusted_close"].pct_change()
    return p


def backtest_positions(positions: pd.DataFrame, prices: pd.DataFrame,
                       adv_lookup: pd.DataFrame | None = None,
                       capital: float = 1_000_000) -> dict:
    """Run a deterministic backtest.

    Parameters
    ----------
    positions : DataFrame
        date, ticker, weight, side, announcement_date, effective_date, variant, index.
    prices : DataFrame
        date, ticker, adjusted_close.
    adv_lookup : DataFrame
        date, ticker, ADV_aud_20d (optional).
    capital : float
        Notional AUD used as the constant strategy notional.
    """
    if positions.empty:
        return {
            "daily_returns": pd.DataFrame(columns=["date", "return", "cum_return"]),
            "trades": pd.DataFrame(),
            "summary": {"trades": 0, "ret": 0.0},
        }
    px = _prepare_prices(prices)
    pos = positions.copy()
    pos["date"] = pd.to_datetime(pos["date"])
    pos = pos.merge(px[["date", "ticker", "return"]], on=["date", "ticker"], how="left")

    # Per-day P&L on the dollar notional.
    pos["pnl"] = pos["weight"] * pos["return"].fillna(0) * capital

    # Trade entries/exits per (ticker, variant, announcement_date).
    grp_keys = ["ticker", "announcement_date", "variant"]
    trades = (pos.groupby(grp_keys)
                 .agg(side=("side", "first"),
                      index=("index", "first"),
                      effective_date=("effective_date", "first"),
                      entry_date=("date", "min"),
                      exit_date=("date", "max"),
                      avg_weight=("weight", "mean"),
                      cumulative_pnl=("pnl", "sum"))
                 .reset_index())

    # Entry + exit transaction costs use the entry-day price as a proxy.
    entry_prices = pos.sort_values("date").groupby(grp_keys).head(1)[
        ["ticker", "announcement_date", "variant", "date"]
    ].rename(columns={"date": "entry_date_actual"})
    trade_costs = []
    for _, row in trades.iterrows():
        trade_value = abs(row["avg_weight"]) * capital
        adv_aud = np.nan
        if adv_lookup is not None and not adv_lookup.empty:
            match = adv_lookup[(adv_lookup["ticker"] == row["ticker"])
                               & (adv_lookup["date"] <= row["entry_date"])]
            if not match.empty:
                adv_aud = float(match.sort_values("date").iloc[-1].get("ADV_aud_20d", np.nan))
        entry = per_trade_cost(trade_value, adv_aud, row["side"])
        exit_ = per_trade_cost(trade_value, adv_aud, row["side"])
        days = max(1, (pd.Timestamp(row["exit_date"]) - pd.Timestamp(row["entry_date"])).days)
        borrow = borrow_cost_for_day(trade_value) * days if row["side"] == "short" else 0.0
        trade_costs.append(entry + exit_ + borrow)
    trades["transaction_costs_aud"] = trade_costs
    trades["net_pnl_aud"] = trades["cumulative_pnl"] - trades["transaction_costs_aud"]

    # Apply transaction costs on the entry and exit dates.
    cost_series = pd.Series(0.0, index=pos["date"].unique())
    cost_series.index = pd.to_datetime(cost_series.index)
    for _, row in trades.iterrows():
        half = row["transaction_costs_aud"] / 2.0
        cost_series.loc[pd.Timestamp(row["entry_date"])] = cost_series.get(
            pd.Timestamp(row["entry_date"]), 0
        ) - half
        cost_series.loc[pd.Timestamp(row["exit_date"])] = cost_series.get(
            pd.Timestamp(row["exit_date"]), 0
        ) - half

    daily = pos.groupby("date")["pnl"].sum().rename("gross_pnl").to_frame()
    daily = daily.join(cost_series.rename("cost"), how="left")
    daily["cost"] = daily["cost"].fillna(0)
    daily["pnl"] = daily["gross_pnl"] + daily["cost"]

    # Re-index onto the full business-day calendar spanning the backtest so
    # idle days contribute zero return. Without this, len(daily)/252 is much
    # smaller than the calendar span and CAGR / Sharpe are overstated.
    start_d = pd.Timestamp(positions["announcement_date"].min()).normalize()
    end_d = pd.Timestamp(positions["effective_date"].max()).normalize()
    full_index = pd.bdate_range(start_d, end_d)
    daily.index = pd.to_datetime(daily.index)
    daily = daily.reindex(full_index, fill_value=0.0)
    daily.index.name = "date"
    daily["return"] = daily["pnl"] / capital
    daily["cum_return"] = (1 + daily["return"]).cumprod() - 1
    daily = daily.reset_index()

    return {
        "daily_returns": daily,
        "trades": trades,
        "summary": {
            "trades": int(len(trades)),
            "net_pnl_aud": float(trades["net_pnl_aud"].sum()),
            "total_return": float(daily["cum_return"].iloc[-1]) if not daily.empty else 0.0,
        },
    }


__all__ = ["backtest_positions"]
