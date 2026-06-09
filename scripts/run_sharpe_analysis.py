"""Sharpe ratios for the switching strategies vs buy-and-hold vs risk-free.

Addresses three questions:
  - Is the much-debated 2013 PRU outlier actually *good* on a Sharpe basis? (A big
    winner lifts return but also volatility, so Sharpe is the fairer test than the
    raw compounded total.) We report every variant WITH and WITHOUT PRU.
  - How does "switching" compare to simply NOT switching (buy & hold ASX 200 TR)?
  - …and to just holding the risk-free rate (assumed 2.5%/yr)?

Sharpe = annualised mean daily excess return / annualised daily vol, with the
excess measured over a 2.5%/yr risk-free rate. Real daily return series only
(outputs/v3_daily.csv = all trades, v3_daily_robust.csv = PRU excluded).

Outputs: outputs/sharpe_table.csv · docs/figures/sharpe_ratios.png
"""

from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from asxrebalance.paths import OUTPUTS_DIR, REPO_ROOT  # noqa: E402

DOCS = REPO_ROOT / "docs" / "figures"
RF_ANNUAL = 0.025
RF_DAILY = RF_ANNUAL / 252.0
TRADING_DAYS = 252

COLS = {
    "asx200_tr": "ASX 200 buy & hold\n(NOT switching)",
    "strat_short": "Switch → SHORT\nremovals",
    "strat_long": "Switch → LONG\nadditions",
    "strat_ls": "Switch → LONG/SHORT",
}


def stats(r: pd.Series):
    r = r.dropna()
    n = len(r)
    cagr = (1 + r).prod() ** (TRADING_DAYS / n) - 1
    vol = r.std() * np.sqrt(TRADING_DAYS)
    sharpe = (r.mean() - RF_DAILY) / r.std() * np.sqrt(TRADING_DAYS) if r.std() > 0 else np.nan
    return cagr * 100, vol * 100, sharpe


def main() -> None:
    alld = pd.read_csv(OUTPUTS_DIR / "v3_daily.csv", parse_dates=["date"])
    rob = pd.read_csv(OUTPUTS_DIR / "v3_daily_robust.csv", parse_dates=["date"])

    rows = []
    for col, label in COLS.items():
        ca, va, sa = stats(alld[col])
        cr, vr, sr = stats(rob[col])
        rows.append({"variant": label.replace("\n", " "), "col": col,
                     "cagr_all_pct": round(ca, 2), "vol_all_pct": round(va, 2), "sharpe_all": round(sa, 3),
                     "cagr_robust_pct": round(cr, 2), "vol_robust_pct": round(vr, 2),
                     "sharpe_robust": round(sr, 3)})
    # risk-free baseline
    rows.append({"variant": "Hold risk-free (2.5%/yr)", "col": "rf",
                 "cagr_all_pct": RF_ANNUAL * 100, "vol_all_pct": 0.0, "sharpe_all": 0.0,
                 "cagr_robust_pct": RF_ANNUAL * 100, "vol_robust_pct": 0.0, "sharpe_robust": 0.0})
    tab = pd.DataFrame(rows)
    tab.to_csv(OUTPUTS_DIR / "sharpe_table.csv", index=False)

    # ---- chart: Sharpe with vs without the PRU outlier ----
    plot = tab[tab.col != "rf"]
    x = np.arange(len(plot)); w = 0.38
    fig, ax = plt.subplots(figsize=(12.5, 7))
    b1 = ax.bar(x - w / 2, plot["sharpe_all"], w, color="#90caf9",
                label="WITH 2013 PRU outlier (all trades)")
    b2 = ax.bar(x + w / 2, plot["sharpe_robust"], w, color="#1565c0",
                label="WITHOUT PRU (robust)")
    for bars, key in ((b1, "sharpe_all"), (b2, "sharpe_robust")):
        for xi, v in zip([r.get_x() + r.get_width() / 2 for r in bars], plot[key]):
            ax.annotate(f"{v:.2f}", (xi, v), ha="center",
                        va="bottom" if v >= 0 else "top", fontsize=8.5, fontweight="bold")
    # reference lines: buy & hold and risk-free
    bh = tab[tab.col == "asx200_tr"]["sharpe_all"].iloc[0]
    ax.axhline(bh, color="#616161", ls="--", lw=1.2, label=f"ASX 200 buy & hold Sharpe = {bh:.2f}")
    ax.axhline(0, color="#c62828", ls=":", lw=1.4, label="risk-free (2.5%) Sharpe = 0")
    ax.set_xticks(x); ax.set_xticklabels(plot["variant"].str.replace(" ", "\n", 1), fontsize=8.5)
    ax.set_ylabel("Annualised Sharpe ratio (rf = 2.5%/yr)")
    ax.set_title("Sharpe ratios — does switching beat buy-and-hold, risk-adjusted?\n"
                 "And is the 2013 PRU outlier actually *good* for the short book's Sharpe?")
    ax.legend(fontsize=9); ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout(); fig.savefig(DOCS / "sharpe_ratios.png", dpi=140); plt.close(fig)
    print("  -> docs/figures/sharpe_ratios.png")

    pd.set_option("display.width", 200)
    print("\nSHARPE / RETURN / VOL (annualised; rf = 2.5%):")
    print(tab[["variant", "cagr_all_pct", "vol_all_pct", "sharpe_all",
               "cagr_robust_pct", "vol_robust_pct", "sharpe_robust"]].to_string(index=False))
    print(f"\nspan: {alld.date.min().date()} -> {alld.date.max().date()}  ({len(alld)} trading days)")


if __name__ == "__main__":
    main()
