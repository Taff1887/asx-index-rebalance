"""Why do shorts outperform longs in the long/short variant?"""

from __future__ import annotations

import pandas as pd

from asxrebalance.paths import OUTPUTS_DIR, PROCESSED_BENCHMARK_DIR


def main() -> None:
    b = pd.read_csv(PROCESSED_BENCHMARK_DIR / "asx200_benchmark.csv", parse_dates=["date"])
    b["ret"] = b["adjusted_close"].pct_change()
    tr = pd.read_csv(OUTPUTS_DIR / "strategy_trades_announcement_long_short.csv",
                      parse_dates=["announcement_date", "effective_date"])

    def window_ret(row: pd.Series) -> float:
        win = b[(b["date"] >= row["announcement_date"])
                & (b["date"] <= row["effective_date"])]
        return float((1 + win["ret"].fillna(0)).prod() - 1)

    tr["bench_window_return"] = tr.apply(window_ret, axis=1)
    print("Benchmark return during the announcement-to-effective windows:")
    print(tr["bench_window_return"].describe().round(4))
    print()
    neg = tr["bench_window_return"] < 0
    print(f"Trades where benchmark FELL during the window: {int(neg.sum())} of {len(tr)}"
          f"  ({100 * neg.mean():.0f}%)")
    longs = tr["side"] == "long"
    shorts = tr["side"] == "short"
    print()
    print(f"  Long  PnL when benchmark fell: A${tr.loc[longs & neg, 'net_pnl_aud'].sum():>10,.0f}"
          f"  ({int((longs & neg).sum()):>3} trades, avg A${tr.loc[longs & neg, 'net_pnl_aud'].mean():,.0f})")
    print(f"  Short PnL when benchmark fell: A${tr.loc[shorts & neg, 'net_pnl_aud'].sum():>10,.0f}"
          f"  ({int((shorts & neg).sum()):>3} trades, avg A${tr.loc[shorts & neg, 'net_pnl_aud'].mean():,.0f})")
    print(f"  Long  PnL when benchmark rose: A${tr.loc[longs & ~neg, 'net_pnl_aud'].sum():>10,.0f}"
          f"  ({int((longs & ~neg).sum()):>3} trades, avg A${tr.loc[longs & ~neg, 'net_pnl_aud'].mean():,.0f})")
    print(f"  Short PnL when benchmark rose: A${tr.loc[shorts & ~neg, 'net_pnl_aud'].sum():>10,.0f}"
          f"  ({int((shorts & ~neg).sum()):>3} trades, avg A${tr.loc[shorts & ~neg, 'net_pnl_aud'].mean():,.0f})")


if __name__ == "__main__":
    main()
