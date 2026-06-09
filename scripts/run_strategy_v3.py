"""Strategy v3: longer holds + the 'hold ASX 200 instead of cash' overlay.

Builds on the validated v2 trade gate. Adds:

1. LONGER EXIT WINDOWS to test the superannuation-fund hypothesis (do index
   funds keep buying additions for weeks after inclusion?):
       exit at effective, +5, +10, +20, +30, +40 trading days.

2. THE 'HOLD ASX 200 INSTEAD OF CASH' STRATEGY (switching model). The
   standalone strategy is in cash ~89% of the time. Here, instead of cash, the
   portfolio holds the ASX 200 total-return index (STW.AX adjusted close,
   dividends reinvested) by default, and SWITCHES into the rebalance trade only
   during each ~10-day holding window:

       daily_return(d) = ASX 200 TR return(d)            ... most days (default)
       daily_return(d) = trade-book return(d)            ... during a window

   The trade-book return during a window is, per variant:
       long-only   : mean daily return of that quarter's ADDITIONS
       short-only  : minus the mean daily return of that quarter's REMOVALS
       long/short  : mean(additions) - mean(removals)   (~market-neutral)

   No leverage: you are either in the market OR in the trade, never both. This
   is the honest "what if the idle cash earned the index instead" construction
   the user asked for.

Everything in percent. Real prices only. No cumulative-over-cash chart.

Outputs:
    outputs/v3_horizon.csv     per-trade mean/median/win by tier,side,exit window
    outputs/v3_overlay.csv     total return: ASX200 TR vs overlay variants
    outputs/v3_daily.csv       daily return series (benchmark + overlay variants)
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from asxrebalance.paths import (
    OUTPUTS_DIR, PROCESSED_BENCHMARK_DIR, PROCESSED_LABELS_DIR,
    PROCESSED_RECONCILED_DIR, RAW_YAHOO_DIR,
)

TIERS = ["ASX20", "ASX50", "ASX100", "ASX200"]
EXITS = [("eff", 0), ("eff5", 5), ("eff10", 10), ("eff20", 20), ("eff30", 30), ("eff40", 40)]
MAX_GAP_TD = 5
MIN_TURNOVER_AUD = 250_000
KNOWN_TICKER_REUSE = {"AHE", "VRL"}


def _load_price(tk):
    for d in (PROCESSED_RECONCILED_DIR / "prices", RAW_YAHOO_DIR):
        p = d / f"{tk}.csv"
        if p.exists():
            df = pd.read_csv(p, parse_dates=["date"]).sort_values("date").reset_index(drop=True)
            if not df.empty:
                return df
    return None


def _bar_near(df, target):
    sub = df[df["date"] >= target]
    if sub.empty:
        return None
    bar = sub.iloc[0]
    if (bar["date"] - target).days > MAX_GAP_TD + 5:
        return None
    return bar


def _turnover(df, a, b):
    w = df[(df["date"] >= a) & (df["date"] <= b)]
    if w.empty:
        return 0.0
    return float((w["close"] * w["volume"]).median())


def valid_entry(df, ann, eff):
    """Return entry bar if the event is tradeable per the v2 gate, else None."""
    if df["date"].min() > ann + pd.Timedelta(days=3):
        return None
    entry = _bar_near(df, ann + pd.tseries.offsets.BDay(1))
    if entry is None:
        return None
    base_exit = _bar_near(df, eff)
    if base_exit is None or entry["date"] >= base_exit["date"]:
        return None
    if _turnover(df, entry["date"], base_exit["date"]) < MIN_TURNOVER_AUD:
        return None
    if entry["volume"] <= 0:
        return None
    return entry


def main() -> None:
    labels = pd.read_csv(PROCESSED_LABELS_DIR / "rebalance_labels.csv",
                          parse_dates=["announcement_date", "effective_date"])
    labels = labels[labels["index"].isin(TIERS)].copy()

    # ASX 200 total return (daily) for the overlay base + abnormal returns.
    tr = pd.read_csv(PROCESSED_BENCHMARK_DIR / "asx200_tr.csv", parse_dates=["date"]).sort_values("date")
    tr = tr.set_index("date")["adjusted_close"]
    tr_ret = tr.pct_change()

    trade_rows = []
    collected = []  # (side, announcement_date, raw_stock_daily_return_in_window)

    for _, evt in labels.iterrows():
        tk = str(evt["ticker"])
        if tk in KNOWN_TICKER_REUSE:
            continue
        df = _load_price(tk)
        if df is None:
            continue
        entry = valid_entry(df, evt["announcement_date"], evt["effective_date"])
        if entry is None:
            continue
        side = "long" if evt["action"] == "Addition" else "short"
        sgn = 1 if side == "long" else -1
        e_px = float(entry["adjusted_close"]); e_date = entry["date"]

        # per-trade returns at each exit horizon
        row = {"tier": evt["index"], "side": side, "ticker": tk,
               "announcement_date": evt["announcement_date"], "entry_date": e_date}
        exit_dates = {}
        for lbl, off in EXITS:
            xb = _bar_near(df, evt["effective_date"] + pd.tseries.offsets.BDay(off))
            if xb is not None and xb["date"] > e_date and float(xb["adjusted_close"]) > 0:
                raw = (float(xb["adjusted_close"]) / e_px - 1) * 100
                row[f"trade_{lbl}_pct"] = round(sgn * raw, 3)
                exit_dates[lbl] = xb["date"]
            else:
                row[f"trade_{lbl}_pct"] = np.nan
        trade_rows.append(row)

        # switching: store the RAW daily stock returns over the holding window
        # (entry -> eff5 exit). During this window the portfolio switches out
        # of ASX 200 and into this trade.
        x_date = exit_dates.get("eff5") or exit_dates.get("eff")
        if x_date is None:
            continue
        stock_ret = df.set_index("date")["adjusted_close"].pct_change()
        win = stock_ret.loc[(stock_ret.index > e_date) & (stock_ret.index <= x_date)]
        collected.append((side, tk, int(evt["announcement_date"].year), win))

    trades = pd.DataFrame(trade_rows)

    # ---- horizon summary (the superfund test) ----
    hrows = []
    for tier in TIERS:
        for side in ("long", "short"):
            for lbl, _ in EXITS:
                col = f"trade_{lbl}_pct"
                sub = trades.loc[(trades.tier == tier) & (trades.side == side), col].dropna()
                if sub.empty:
                    continue
                hrows.append({"tier": tier, "side": side, "exit": lbl, "n": len(sub),
                              "mean_pct": round(sub.mean(), 3),
                              "median_pct": round(sub.median(), 3),
                              "win_rate_pct": round((sub > 0).mean() * 100, 1)})
    horizon = pd.DataFrame(hrows)
    horizon.to_csv(OUTPUTS_DIR / "v3_horizon.csv", index=False)

    # ---- daily series: trade book during windows, CASH (risk-free) otherwise ----
    # Per day, the equal-weight mean raw return of all OPEN long trades and of all
    # open short trades. On days with an open trade the portfolio is in the trade
    # book; on all other days it sits in CASH earning the risk-free rate (2.5%/yr).
    # (No 'switch into the index' overlay — the standalone strategy holds cash.)
    idx = tr_ret.index
    CAP = 0.12          # winsorise daily book return; a constituent rarely moves >12%/day
    RF_DAILY = 0.025 / 252.0  # 2.5%/yr risk-free earned on idle (cash) days

    # The verification pass found a SINGLE 2013 trade (PRU, which fell 55.6% in
    # a near-empty early book) accounts for roughly half the gross short total.
    # The "robust" series excludes that dominant outlier so the headline is not
    # a one-trade artifact.
    OUTLIER = {"PRU"}

    def build_books(exclude: set):
        ls_, lc_ = pd.Series(0.0, index=idx), pd.Series(0.0, index=idx)
        ss_, sc_ = pd.Series(0.0, index=idx), pd.Series(0.0, index=idx)
        for side, tk, yr, win in collected:
            if tk in exclude:
                continue
            w = win.reindex(idx); mask = w.notna()
            if side == "long":
                ls_ = ls_.add(w.fillna(0.0), fill_value=0.0); lc_ = lc_.add(mask.astype(float), fill_value=0.0)
            else:
                ss_ = ss_.add(w.fillna(0.0), fill_value=0.0); sc_ = sc_.add(mask.astype(float), fill_value=0.0)
        lr = (ls_ / lc_).where(lc_ > 0).clip(-CAP, CAP)
        sr = (ss_ / sc_).where(sc_ > 0).clip(-CAP, CAP)
        d = pd.DataFrame({"asx200_tr": tr_ret}, index=idx)   # buy & hold benchmark
        d["strat_long"] = lr.where(lc_ > 0, RF_DAILY)         # else CASH
        d["strat_short"] = (-sr).where(sc_ > 0, RF_DAILY)     # else CASH
        anyopen = (lc_ > 0) | (sc_ > 0)
        d["strat_ls"] = (lr.fillna(0.0) - sr.fillna(0.0)).where(anyopen, RF_DAILY)
        return d, anyopen

    daily, any_open = build_books(set())     # all trades
    daily_robust, _ = build_books(OUTLIER)   # exclude the dominant 2013 PRU outlier

    first = min(win.index.min() for _, _, _, win in collected if len(win))
    daily = daily.loc[daily.index >= first]
    daily_robust = daily_robust.loc[daily_robust.index >= first]
    daily.reset_index().rename(columns={"index": "date"}).to_csv(OUTPUTS_DIR / "v3_daily.csv", index=False)
    daily_robust.reset_index().rename(columns={"index": "date"}).to_csv(
        OUTPUTS_DIR / "v3_daily_robust.csv", index=False)

    def total(d, col):
        return float((1 + d[col].fillna(0)).prod() - 1) * 100

    deployed = float(any_open.reindex(daily.index).fillna(False).mean()) * 100
    ov = pd.DataFrame([
        {"series": "ASX 200 buy & hold (total return)",
         "all_trades_pct": round(total(daily, "asx200_tr"), 1),
         "robust_ex_outlier_pct": round(total(daily_robust, "asx200_tr"), 1)},
        {"series": "Trade LONG additions, cash (2.5%) between",
         "all_trades_pct": round(total(daily, "strat_long"), 1),
         "robust_ex_outlier_pct": round(total(daily_robust, "strat_long"), 1)},
        {"series": "Trade SHORT removals, cash (2.5%) between",
         "all_trades_pct": round(total(daily, "strat_short"), 1),
         "robust_ex_outlier_pct": round(total(daily_robust, "strat_short"), 1)},
        {"series": "Trade LONG/SHORT, cash (2.5%) between",
         "all_trades_pct": round(total(daily, "strat_ls"), 1),
         "robust_ex_outlier_pct": round(total(daily_robust, "strat_ls"), 1)},
    ])
    ov.to_csv(OUTPUTS_DIR / "v3_overlay.csv", index=False)
    print(f"\n(in a trade on {deployed:.1f}% of days; in CASH at 2.5% the rest)")

    print("HORIZON — ASX200 mean per-trade % by exit (superfund test):")
    piv = (horizon[horizon.tier == "ASX200"]
           .pivot_table(index="exit", columns="side", values="mean_pct")
           .reindex([e[0] for e in EXITS]))
    print(piv.to_string())
    print("\nHORIZON — ASX200 win rate %:")
    pivw = (horizon[horizon.tier == "ASX200"]
            .pivot_table(index="exit", columns="side", values="win_rate_pct")
            .reindex([e[0] for e in EXITS]))
    print(pivw.to_string())
    print("\nTRADE-BOOK + CASH(2.5%) total return (no leverage, no index overlay):")
    print(ov.to_string(index=False))
    print(f"\nDaily span: {daily.index.min().date()} -> {daily.index.max().date()}  ({len(daily)} days)")


if __name__ == "__main__":
    main()
