"""One-off audit explaining why long-only underperforms."""

from __future__ import annotations

import pandas as pd

from asxrebalance.paths import OUTPUTS_DIR


def main() -> None:
    tr_ls = pd.read_csv(OUTPUTS_DIR / "strategy_trades_announcement_long_short.csv",
                         parse_dates=["announcement_date"])
    tr_lo = pd.read_csv(OUTPUTS_DIR / "strategy_trades_additions_only.csv",
                         parse_dates=["announcement_date"])

    print("==== LONG/SHORT trades ====")
    longs_ls = tr_ls[tr_ls["side"] == "long"]
    shorts_ls = tr_ls[tr_ls["side"] == "short"]
    print(f"  Total: {len(tr_ls)}")
    print(f"  Longs : {len(longs_ls)} trades, avg weight {longs_ls['avg_weight'].mean()*100:.3f}%, "
          f"net PnL A${longs_ls['net_pnl_aud'].sum():,.0f}")
    print(f"  Shorts: {len(shorts_ls)} trades, avg weight {shorts_ls['avg_weight'].mean()*100:.3f}%, "
          f"net PnL A${shorts_ls['net_pnl_aud'].sum():,.0f}")
    print()
    print("==== LONG-ONLY trades ====")
    print(f"  Total: {len(tr_lo)} trades, avg weight {tr_lo['avg_weight'].mean()*100:.3f}%, "
          f"net PnL A${tr_lo['net_pnl_aud'].sum():,.0f}")
    print()
    print("==== Apples-to-apples (per-trade PnL) ====")
    print(f"  Long leg of L/S   : avg A${longs_ls['net_pnl_aud'].mean():,.0f} / trade  "
          f"(weight {longs_ls['avg_weight'].mean()*100:.3f}%)")
    print(f"  Long-only         : avg A${tr_lo['net_pnl_aud'].mean():,.0f} / trade  "
          f"(weight {tr_lo['avg_weight'].mean()*100:.3f}%)")
    if tr_lo["avg_weight"].mean() != 0:
        ratio = longs_ls["avg_weight"].mean() / tr_lo["avg_weight"].mean()
        print(f"  L/S longs have {ratio:.1f}x the position size of long-only longs.")
    print()
    print("Diagnosis:")
    print("  size_positions() weights every trade in its (entry_date, side) bucket.")
    print("  enforce_exposure() then caps gross AND net to limits from strategy.yaml.")
    print()
    print("    max_gross_exposure: 1.00   -> total |weight| <= 100% of NAV")
    print("    max_net_exposure  : 0.20   -> sum(weight)  <= 20% of NAV")
    print()
    print("  For long-only, sum(weight) > 0 by construction (no shorts to cancel).")
    print("  So the 20% net cap clamps the book down hard — by ~3-5x.")
    print("  For long/short, longs and shorts cancel each other so net ~ 0 naturally.")
    print("  Result: same names, same dates, but long-only has way smaller positions.")


if __name__ == "__main__":
    main()
