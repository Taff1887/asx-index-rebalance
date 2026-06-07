"""Generate a fresh synthetic dataset for the full pipeline.

Produces:

* 300 fake ASX tickers with daily OHLCV from both FMP and Yahoo (with
  intentional discrepancies to exercise the validation layer).
* Shares outstanding and IWF panels.
* Constituent snapshots for ASX 50 / 100 / 200 across the backtest window.
* Historical rebalance labels consistent with the constituents.
* Benchmark ASX 200 price series.

Run with: ``python scripts/generate_synthetic_data.py``
"""

from __future__ import annotations

import argparse
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from asxrebalance.calendar import iter_rebalance_windows
from asxrebalance.paths import (
    PROCESSED_BENCHMARK_DIR,
    PROCESSED_LABELS_DIR,
    PROCESSED_RECONCILED_DIR,
    RAW_FMP_DIR,
    RAW_MANUAL_DIR,
    RAW_YAHOO_DIR,
    ensure_dirs,
)


N_TICKERS = 300
SECTORS = [
    "Financials", "Materials", "Energy", "Healthcare", "Industrials",
    "Consumer Discretionary", "Consumer Staples", "Information Technology",
    "Communication Services", "Utilities", "Real Estate",
]


def make_tickers(n: int) -> list[str]:
    rng = np.random.default_rng(42)
    used = set()
    tickers = []
    while len(tickers) < n:
        code = "".join(rng.choice(list("ABCDEFGHIJKLMNOPQRSTUVWXYZ"), 3))
        if code not in used:
            used.add(code)
            tickers.append(code)
    return sorted(tickers)


def make_prices(tickers: list[str], start: date, end: date,
                rng: np.random.Generator) -> pd.DataFrame:
    """Generate prices with a common market factor + per-stock idiosyncratic noise.

    Single-stock daily return = beta_i * market_eps + idio_eps_i. This produces:
      * Realistic single-stock vol (~25-35% annualised).
      * Positive cross-stock correlation (~0.3-0.5) — without it the
        equal-weighted index is too smooth.
      * Equal-weighted index vol ≈ mean(beta) * market_vol when there are
        many names, so the synthetic ASX 200 proxy has ~14% vol and real
        drawdowns instead of the 3.6% vol of an average of independent walks.
    """
    dates = pd.bdate_range(start, end)
    n_days = len(dates)

    # ---- Market factor (drives the ASX 200 proxy) ----------------------------
    market_drift = 0.00028          # ~7% annualised expected drift
    market_vol = 0.0088              # ~14% annualised vol
    market_eps = rng.normal(market_drift, market_vol, size=n_days)
    # Inject a few crisis-style shocks so the benchmark has tail risk.
    n_crises = max(1, n_days // 700)
    for _ in range(n_crises):
        idx = int(rng.integers(low=60, high=n_days - 60))
        shock_days = int(rng.integers(low=10, high=30))
        end_shock = min(n_days, idx + shock_days)
        # Negative drift cluster: -0.5% to -1.5% per day over the cluster.
        cluster_mean = -float(rng.uniform(0.003, 0.012))
        cluster_vol = market_vol * 2.5
        market_eps[idx:end_shock] += rng.normal(cluster_mean, cluster_vol,
                                                 size=end_shock - idx)

    # ---- Per-stock parameters ------------------------------------------------
    base_prices = rng.uniform(0.5, 80.0, size=len(tickers))
    betas = rng.uniform(0.3, 1.8, size=len(tickers))
    idio_drifts = rng.normal(0.0, 0.00008, size=len(tickers))
    idio_vols = rng.uniform(0.010, 0.025, size=len(tickers))

    frames = []
    for i, t in enumerate(tickers):
        idio_eps = rng.normal(idio_drifts[i], idio_vols[i], size=n_days)
        eps = betas[i] * market_eps + idio_eps
        # Single-stock jumps so the validation layer has something to flag.
        if i % 47 == 0:
            jump_day = int(rng.integers(low=50, high=n_days - 50))
            eps[jump_day] += float(rng.choice([-0.3, 0.3]))
        prices = base_prices[i] * np.exp(np.cumsum(eps))
        volume = rng.lognormal(mean=12.5, sigma=0.6, size=n_days)
        # Sprinkle zero-volume days.
        zero_mask = rng.uniform(size=n_days) < 0.005
        volume[zero_mask] = 0.0
        df = pd.DataFrame({
            "date": dates,
            "ticker": t,
            "open": prices * (1 + rng.normal(0, 0.001, size=n_days)),
            "high": prices * (1 + np.abs(rng.normal(0, 0.005, size=n_days))),
            "low":  prices * (1 - np.abs(rng.normal(0, 0.005, size=n_days))),
            "close": prices,
            "adjusted_close": prices,
            "volume": volume,
            "vwap": prices,
        })
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def add_yahoo_discrepancies(fmp: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """Introduce realistic small differences and a few large discrepancies."""
    y = fmp.copy()
    n = len(y)
    # 0.05% baseline noise on close/adjusted_close.
    noise = rng.normal(0, 0.0005, size=n)
    y["close"] = y["close"] * (1 + noise)
    y["adjusted_close"] = y["adjusted_close"] * (1 + noise * 0.5)

    # ~2% noise on volume.
    vol_noise = rng.normal(0, 0.02, size=n)
    y["volume"] = y["volume"] * (1 + vol_noise)
    y["volume"] = y["volume"].clip(lower=0)

    # Strong discrepancies on 0.1% of rows.
    severe_mask = rng.uniform(size=n) < 0.001
    y.loc[severe_mask, "close"] *= 1 + rng.choice([-0.06, 0.06], size=severe_mask.sum())

    # Missing observations on Yahoo for 0.2% of rows.
    miss_mask = rng.uniform(size=n) < 0.002
    return y.loc[~miss_mask].reset_index(drop=True)


def make_shares(tickers: list[str], start: date, end: date,
                rng: np.random.Generator) -> pd.DataFrame:
    """Annual shares outstanding panel."""
    rows = []
    base = rng.uniform(2e7, 5e9, size=len(tickers))
    for i, t in enumerate(tickers):
        for y in range(start.year, end.year + 1):
            growth = (1 + rng.uniform(-0.05, 0.10)) ** (y - start.year)
            rows.append({
                "date": pd.Timestamp(year=y, month=1, day=2),
                "ticker": t,
                "shares_outstanding": base[i] * growth,
                "source": "synthetic",
                "quality_flag": "ok",
            })
    return pd.DataFrame(rows)


def make_iwf(tickers: list[str], start: date, rng: np.random.Generator) -> pd.DataFrame:
    rows = []
    for t in tickers:
        rows.append({
            "date": pd.Timestamp(start),
            "ticker": t,
            "iwf": float(rng.uniform(0.3, 1.0)),
            "source": "synthetic",
            "quality_flag": "ok",
        })
    return pd.DataFrame(rows)


def make_company_profile(tickers: list[str], rng: np.random.Generator) -> pd.DataFrame:
    return pd.DataFrame({
        "canonical_ticker": tickers,
        "yahoo_ticker": [f"{t}.AX" for t in tickers],
        "fmp_ticker": [f"{t}.AX" for t in tickers],
        "company_name": [f"{t} Holdings Ltd" for t in tickers],
        "sector": rng.choice(SECTORS, size=len(tickers)),
        "industry": rng.choice(["Banks", "Mining", "Oil", "Software", "Retail",
                                "Telecom", "REITs", "Utilities"], size=len(tickers)),
        "first_seen_date": pd.Timestamp("2010-01-01"),
        "last_seen_date": pd.Timestamp.today().normalize(),
        "active_flag": True,
        "notes": "synthetic",
    })


def make_constituents_and_labels(prices: pd.DataFrame, shares: pd.DataFrame,
                                  start: date, end: date) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Rank by simulated market cap at every announcement date and tag changes."""
    sw = shares.sort_values(["ticker", "date"])
    rows_const = []
    rows_labels = []
    prev_members: dict[str, set[str]] = {"ASX50": set(), "ASX100": set(), "ASX200": set()}

    for index_name, target in [("ASX200", 200), ("ASX100", 100), ("ASX50", 50)]:
        windows = iter_rebalance_windows(index_name, start, end)
        for window in windows:
            ann = pd.Timestamp(window.announcement_date)
            ref = pd.Timestamp(window.reference_date)
            # Most-recent shares before reference date.
            s_asof = (sw.loc[sw["date"] <= ref]
                        .sort_values("date").groupby("ticker").tail(1))
            # Average close over the 3-month window ending at the reference date.
            window_prices = prices[
                (prices["date"] <= ref)
                & (prices["date"] >= ref - pd.tseries.offsets.BDay(63))
            ]
            avg_price = window_prices.groupby("ticker")["close"].mean()
            mcap = (s_asof.set_index("ticker")["shares_outstanding"] * avg_price).dropna()
            ranked = mcap.sort_values(ascending=False).head(target)
            new_members = set(ranked.index)
            for t in new_members:
                rows_const.append({
                    "date": pd.Timestamp(window.effective_date),
                    "index": index_name,
                    "ticker": t,
                    "company_name": f"{t} Holdings Ltd",
                    "iwf": 1.0,
                })
            prev = prev_members.get(index_name, set())
            added = new_members - prev
            removed = prev - new_members
            for t in added:
                rows_labels.append({
                    "announcement_date": ann,
                    "effective_date": pd.Timestamp(window.effective_date),
                    "index": index_name,
                    "action": "Addition",
                    "ticker": t,
                    "company_name": f"{t} Holdings Ltd",
                })
            for t in removed:
                rows_labels.append({
                    "announcement_date": ann,
                    "effective_date": pd.Timestamp(window.effective_date),
                    "index": index_name,
                    "action": "Removal",
                    "ticker": t,
                    "company_name": f"{t} Holdings Ltd",
                })
            prev_members[index_name] = new_members

    return pd.DataFrame(rows_const), pd.DataFrame(rows_labels)


def make_benchmark(prices: pd.DataFrame, top_n: int = 200) -> pd.DataFrame:
    """Equal-weight average of top-N tickers as a synthetic ASX 200 proxy."""
    daily = prices.pivot_table(index="date", columns="ticker", values="adjusted_close")
    universe = daily.iloc[-1].nlargest(top_n).index
    sub = daily[universe]
    ret = sub.pct_change().mean(axis=1)
    px = (1 + ret.fillna(0)).cumprod() * 100
    out = pd.DataFrame({
        "date": px.index,
        "ticker": "STW",
        "close": px.values,
        "adjusted_close": px.values,
        "volume": 1_000_000,
        "source": "synthetic",
    })
    return out.reset_index(drop=True)


def write_per_ticker_csvs(panel: pd.DataFrame, target_dir: Path) -> None:
    target_dir.mkdir(parents=True, exist_ok=True)
    cols = ["date", "open", "high", "low", "close", "adjusted_close", "volume", "vwap"]
    for ticker, grp in panel.groupby("ticker"):
        grp[cols].to_csv(target_dir / f"{ticker}.csv", index=False)


def inject_index_effect(panel: pd.DataFrame, labels: pd.DataFrame,
                         addition_uplift: float = 0.025,
                         removal_drag: float = 0.020,
                         reversion_pct: float = 0.30) -> pd.DataFrame:
    """Bake a realistic S&P/ASX index effect into the price panel.

    For every labelled event we add a multiplicative drift to the ticker's
    prices from announcement to effective (ramping up to ``addition_uplift`` or
    down by ``removal_drag``), then partially revert ``reversion_pct`` of the
    move over the next ~10 business days. The effect compounds across multiple
    events on the same ticker so the same generator can drive ASX 50 / 100 /
    200 simultaneously.

    Magnitudes default to roughly the literature values for post-2010 ASX
    additions (~2-3%) and removals (~1.5-2.5%).
    """
    if labels.empty:
        return panel

    out = panel.sort_values(["ticker", "date"]).copy().reset_index(drop=True)
    extra = np.zeros(len(out), dtype=float)

    # Build a fast lookup from ticker to the integer range of rows in `out`.
    bounds: dict[str, tuple[int, int]] = {}
    for ticker, grp in out.groupby("ticker", sort=False):
        bounds[ticker] = (grp.index[0], grp.index[-1] + 1)

    for _, evt in labels.iterrows():
        ticker = evt["ticker"]
        if ticker not in bounds:
            continue
        lo, hi = bounds[ticker]
        sub = out.iloc[lo:hi]
        ann = pd.Timestamp(evt["announcement_date"])
        eff = pd.Timestamp(evt["effective_date"])
        is_add = evt["action"] in ("Addition", "Promotion")
        mag = addition_uplift if is_add else -removal_drag

        window_mask = (sub["date"] > ann) & (sub["date"] <= eff)
        n = int(window_mask.sum())
        if n > 0:
            per_day = (1.0 + mag) ** (1.0 / n) - 1.0
            idx = sub.index[window_mask]
            extra[idx] += per_day

        # Partial reversion over the next 10 business days.
        rev_total = -mag * reversion_pct
        rev_end = eff + pd.Timedelta(days=15)
        rev_mask = (sub["date"] > eff) & (sub["date"] <= rev_end)
        rev_n = int(rev_mask.sum())
        if rev_n > 0:
            per_day = (1.0 + rev_total) ** (1.0 / rev_n) - 1.0
            idx = sub.index[rev_mask]
            extra[idx] += per_day

    out["cum_multiplier"] = 1.0
    for ticker, (lo, hi) in bounds.items():
        out.loc[lo:hi - 1, "cum_multiplier"] = (1.0 + extra[lo:hi]).cumprod()

    for col in ("open", "high", "low", "close", "adjusted_close", "vwap"):
        if col in out.columns:
            out[col] = out[col] * out["cum_multiplier"]
    return out.drop(columns="cum_multiplier")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", default="2015-01-02",
                        help="Backtest start date (default: 2015-01-02 — ~11 years of history).")
    parser.add_argument("--end", default="2026-05-30")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--index-effect", action="store_true", default=True,
                        help="Inject a realistic announcement→effective price effect.")
    parser.add_argument("--no-index-effect", dest="index_effect", action="store_false")
    parser.add_argument("--addition-uplift", type=float, default=0.025,
                        help="Total addition uplift between announcement and effective.")
    parser.add_argument("--removal-drag", type=float, default=0.020,
                        help="Total removal drag between announcement and effective.")
    parser.add_argument("--reversion-pct", type=float, default=0.30,
                        help="Fraction of the move that reverses over the 10 days post-effective.")
    args = parser.parse_args()

    ensure_dirs()
    rng = np.random.default_rng(args.seed)
    start = date.fromisoformat(args.start)
    end = date.fromisoformat(args.end)

    print("Generating synthetic tickers...")
    tickers = make_tickers(N_TICKERS)
    print(f"  {len(tickers)} tickers")

    print("Generating FMP price series...")
    fmp = make_prices(tickers, start, end, rng)
    write_per_ticker_csvs(fmp, RAW_FMP_DIR)
    print(f"  {len(fmp)} rows written to {RAW_FMP_DIR}")

    print("Generating Yahoo price series with intentional discrepancies...")
    yahoo = add_yahoo_discrepancies(fmp, rng)
    write_per_ticker_csvs(yahoo, RAW_YAHOO_DIR)
    print(f"  {len(yahoo)} rows written to {RAW_YAHOO_DIR}")

    print("Generating shares outstanding panel...")
    shares = make_shares(tickers, start, end, rng)
    shares.to_csv(RAW_MANUAL_DIR / "shares.csv", index=False)
    print(f"  {len(shares)} rows")

    print("Generating IWF panel...")
    iwf = make_iwf(tickers, start, rng)
    iwf.to_csv(RAW_MANUAL_DIR / "iwf.csv", index=False)

    print("Generating company profile / ticker master...")
    profile = make_company_profile(tickers, rng)
    profile.to_csv(PROCESSED_RECONCILED_DIR / "ticker_master.csv", index=False)

    print("Generating constituents and rebalance labels...")
    constituents, labels = make_constituents_and_labels(fmp, shares, start, end)
    constituents.to_csv(PROCESSED_RECONCILED_DIR / "constituents.csv", index=False)
    labels.to_csv(PROCESSED_LABELS_DIR / "rebalance_labels.csv", index=False)
    print(f"  {len(constituents)} constituent rows, {len(labels)} label rows")

    if args.index_effect:
        print(f"Injecting index effect (add+{args.addition_uplift:.1%}, "
              f"rem-{args.removal_drag:.1%}, reversion {args.reversion_pct:.0%})...")
        fmp = inject_index_effect(fmp, labels,
                                   addition_uplift=args.addition_uplift,
                                   removal_drag=args.removal_drag,
                                   reversion_pct=args.reversion_pct)
        yahoo = inject_index_effect(yahoo, labels,
                                     addition_uplift=args.addition_uplift,
                                     removal_drag=args.removal_drag,
                                     reversion_pct=args.reversion_pct)
        write_per_ticker_csvs(fmp, RAW_FMP_DIR)
        write_per_ticker_csvs(yahoo, RAW_YAHOO_DIR)
        print("  index effect applied to FMP and Yahoo panels")

    print("Generating benchmark ASX 200 proxy...")
    benchmark = make_benchmark(fmp)
    benchmark.to_csv(PROCESSED_BENCHMARK_DIR / "asx200_benchmark.csv", index=False)

    print("Synthetic data generation complete.")


if __name__ == "__main__":
    main()
