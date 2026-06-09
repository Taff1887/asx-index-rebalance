"""What if you could trade at the OPEN instead of waiting for the close?

The S&P announcement is released AFTER the market close on day T, so the first
moment you can legitimately trade is the **next session's open** (T+1 open). The
main strategy conservatively waits for that day's **close** (T+1 close). The
event study says the move lands at the open — so how much edge does the close
entry give up?

For every validated trade we compare, holding the SAME exit (eff+5 close):
    entry at T+1 OPEN   (first legitimate fill)   vs
    entry at T+1 CLOSE  (the conservative default)
and the "first-day move" between them = entry_close/entry_open − 1 (signed),
which is exactly what waiting for the close forfeits.

Real prices only (open + close from the same series used everywhere else).

Outputs: outputs/open_vs_close.csv · docs/figures/open_vs_close.png
"""

from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import stats  # noqa: E402

from asxrebalance.paths import (  # noqa: E402
    OUTPUTS_DIR, PROCESSED_RECONCILED_DIR, RAW_YAHOO_DIR, REPO_ROOT,
)

DOCS = REPO_ROOT / "docs" / "figures"
EXIT = "eff5"   # the sweet-spot exit


def _adj_open_on(tk, d):
    """Adjusted OPEN = raw open x (adjusted_close / raw close) on that day, so it is
    on the SAME dividend-adjusted basis as the entry_close / exit prices used
    everywhere else. (Comparing raw open to adjusted close would inject the whole
    dividend-adjustment factor — a bug.)"""
    for base in (PROCESSED_RECONCILED_DIR / "prices", RAW_YAHOO_DIR):
        p = base / f"{tk}.csv"
        if p.exists():
            df = pd.read_csv(p, parse_dates=["date"])
            row = df[df["date"] == pd.Timestamp(d)]
            if row.empty:
                return np.nan
            r = row.iloc[0]
            o, c, ac = r.get("open"), r.get("close"), r.get("adjusted_close")
            if pd.notna(o) and pd.notna(c) and pd.notna(ac) and o > 0 and c > 0:
                return float(o) * float(ac) / float(c)
    return np.nan


def main() -> None:
    t = pd.read_csv(OUTPUTS_DIR / "v2_trades.csv",
                    parse_dates=["entry_date", f"exit_{EXIT}_date"])
    t["entry_open"] = [_adj_open_on(tk, d) for tk, d in zip(t["ticker"], t["entry_date"])]
    t = t[t["entry_open"].notna() & t[f"exit_{EXIT}_close"].notna()].copy()
    sgn = np.where(t["side"] == "long", 1.0, -1.0)

    # returns to the SAME exit, from open vs from close
    t["ret_open"] = sgn * (t[f"exit_{EXIT}_close"] / t["entry_open"] - 1) * 100
    t["ret_close"] = sgn * (t[f"exit_{EXIT}_close"] / t["entry_close"] - 1) * 100
    # what the close entry forfeits = the first-day open->close move (signed by side)
    t["first_day_give_up"] = sgn * (t["entry_close"] / t["entry_open"] - 1) * 100
    t.to_csv(OUTPUTS_DIR / "open_vs_close.csv", index=False)

    groups = [("Addition", "long"), ("Removal", "short")]
    summ = []
    for action, side in groups:
        d = t[t["side"] == side]
        summ.append({
            "side": f"{action}s ({side})", "n": len(d),
            "open_median": d["ret_open"].median(), "close_median": d["ret_close"].median(),
            "open_mean": d["ret_open"].mean(), "close_mean": d["ret_close"].mean(),
            "give_up_median": d["first_day_give_up"].median(),
        })
    s = pd.DataFrame(summ)

    # ---- chart: open vs close entry, median per trade ----
    fig, ax = plt.subplots(figsize=(11, 6.8))
    x = np.arange(len(s)); w = 0.38
    bo = ax.bar(x - w / 2, s["open_median"], w, color="#1565c0", label="enter at OPEN (first fill)")
    bc = ax.bar(x + w / 2, s["close_median"], w, color="#90caf9", label="enter at CLOSE (1-session wait)")
    for bars, col in ((bo, "open_median"), (bc, "close_median")):
        for r, v in zip(bars, s[col]):
            ax.annotate(f"{v:+.2f}%", (r.get_x() + r.get_width() / 2, v), ha="center",
                        va="bottom" if v >= 0 else "top", fontsize=9, fontweight="bold")
    # edge captured by the open entry
    for xi, row in zip(x, s.itertuples()):
        edge = row.open_median - row.close_median
        ax.annotate(f"open captures\n{edge:+.2f}% more", (xi, max(row.open_median, row.close_median) + 0.45),
                    ha="center", fontsize=8.5, color="#0d47a1")
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xticks(x); ax.set_xticklabels([f"{r.side}\n(n={r.n})" for r in s.itertuples()])
    ax.set_ylabel(f"Median per-trade return to eff+5 (%)")
    ax.set_title("Trade at the OPEN vs wait for the CLOSE — same eff+5 exit\n"
                 "The announcement is released after-hours, so the next-day OPEN is the first legal fill")
    ax.legend(); ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout(); fig.savefig(DOCS / "open_vs_close.png", dpi=140); plt.close(fig)
    print("  -> docs/figures/open_vs_close.png")

    pd.set_option("display.width", 200)
    print("\nOPEN vs CLOSE entry (same eff+5 exit):")
    print(s.round(2).to_string(index=False))
    # significance of the difference (paired) for removals (the tradeable side)
    rem = t[t.side == "short"]
    diff = (rem["ret_open"] - rem["ret_close"]).dropna()
    tt, p = stats.ttest_1samp(diff, 0)
    print(f"\nRemovals: entering at the open adds a median {rem['first_day_give_up'].median():+.2f}%/trade "
          f"(mean {diff.mean():+.2f}%, paired t={tt:.2f}, p={p:.4f})")


if __name__ == "__main__":
    main()
