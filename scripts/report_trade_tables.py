"""Print the strategy comparison and best/worst trade tables for the README."""

from __future__ import annotations

import pandas as pd

from asxrebalance.paths import OUTPUTS_DIR

CAPITAL = 1_000_000


def add_pct(df: pd.DataFrame) -> pd.DataFrame:
    notional = df["avg_weight"].abs() * CAPITAL
    notional = notional.replace(0, pd.NA)
    df = df.copy()
    df["notional_aud"] = notional
    df["gross_pct"] = df["cumulative_pnl"] / notional * 100
    df["net_pct"] = df["net_pnl_aud"] / notional * 100
    return df


def main() -> None:
    print("=" * 70)
    print("STRATEGY METRICS")
    print("=" * 70)
    for v in ("announcement_long_short", "additions_only", "removals_only"):
        df = pd.read_csv(OUTPUTS_DIR / f"strategy_performance_summary_{v}.csv")
        s = df.iloc[0]
        print(f"\n  {v}")
        print(f"    total return : {s['strategy_total_return']*100:7.1f} %")
        print(f"    CAGR         : {s['cagr']*100:7.2f} %")
        print(f"    vol          : {s['vol']*100:7.2f} %")
        print(f"    Sharpe       : {s['sharpe']:7.2f}")
        print(f"    Sortino      : {s['sortino']:7.2f}")
        print(f"    Max DD       : {s['max_drawdown']*100:7.2f} %")
        print(f"    Calmar       : {s['calmar']:7.2f}")
        print(f"    beta         : {s['beta']:7.4f}")
        print(f"    alpha        : {s['alpha']*100:7.2f} %")
        print(f"    tracking err : {s['tracking_error']*100:7.2f} %")
        print(f"    info ratio   : {s['information_ratio']:7.2f}")
        print(f"    hit rate     : {s['hit_rate']*100:7.1f} %")
    s = df.iloc[0]
    print(f"\n  benchmark")
    print(f"    total return : {s['benchmark_total_return']*100:7.1f} %")
    print(f"    CAGR         : {s['benchmark_cagr']*100:7.2f} %")
    print(f"    vol          : {s['benchmark_vol']*100:7.2f} %")
    print(f"    Sharpe       : {s['benchmark_sharpe']:7.2f}")
    print(f"    Max DD       : {s['benchmark_max_drawdown']*100:7.2f} %")

    print("\n" + "=" * 70)
    print("BEST AND WORST TRADES (long/short, all 8 years)")
    print("=" * 70)
    df = pd.read_csv(OUTPUTS_DIR / "strategy_trades_announcement_long_short.csv",
                      parse_dates=["announcement_date", "effective_date",
                                   "entry_date", "exit_date"])
    df = add_pct(df)
    cols = ["announcement_date", "effective_date", "entry_date", "exit_date",
            "ticker", "index", "side", "notional_aud", "cumulative_pnl",
            "transaction_costs_aud", "net_pnl_aud", "gross_pct", "net_pct"]

    print("\nTop 10 winners by net %:")
    print(df.sort_values("net_pct", ascending=False).head(10)[cols].to_string(index=False))

    print("\nTop 10 losers by net %:")
    print(df.sort_values("net_pct", ascending=True).head(10)[cols].to_string(index=False))

    print("\n2025 SAMPLE (long/short, sorted by net %)")
    sub = df[df["announcement_date"].dt.year >= 2025].sort_values("net_pct", ascending=False)
    print(f"\n  Best 5 in 2025:")
    print(sub.head(5)[cols].to_string(index=False))
    print(f"\n  Worst 5 in 2025:")
    print(sub.tail(5)[cols].to_string(index=False))

    print("\n" + "=" * 70)
    print("2025 AGGREGATE BY INDEX/SIDE (% of notional)")
    print("=" * 70)
    agg = (sub.groupby(["index", "side"])
              .agg(n=("ticker", "size"),
                    gross_pct=("gross_pct", "mean"),
                    cost_pct=("transaction_costs_aud",
                              lambda s: (s / (sub.loc[s.index, "notional_aud"])).mean() * 100),
                    net_pct=("net_pct", "mean")))
    print(agg.round(2).to_string())


if __name__ == "__main__":
    main()
