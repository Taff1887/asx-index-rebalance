"""Frictionless (gross) then friction-laden (net) per-trade backtest.

The user asked to "build the trading strategy WITHOUT any frictions ... and then
implement with liquidity costs, borrowing costs etc."

GROSS  = the per-trade returns exactly as in run_strategy_v2 (no costs).
NET    = gross minus a realistic, per-trade cost stack built from config/costs.yaml:

    execution (both sides):  brokerage + half-spread + slippage + exchange fee
                             = 2 x (5 + 5 + 10 + 0.5) bps = 41 bps round trip
    liquidity / market impact (both sides, square-root law):
        impact_per_side = coefficient * daily_vol * sqrt(clip / ADV)
        where ADV = the trade's median window $-turnover (REAL, from the data),
              daily_vol = the stock's trailing 60-day realised vol (REAL),
              clip = assumed $ order size (we report 3 sizes).
    short borrow (shorts only):  300 bps/yr  x  holding_days / 365

Liquidity cost is therefore driven by each name's ACTUAL traded turnover, which
is the point: thin removals cost more to short than liquid ones.

Outputs:
    outputs/friction_pertrade.csv   per-trade gross, each cost (bps), net
    outputs/friction_summary.csv    gross vs net mean/median/win by tier,side,exit
    docs/figures/gross_vs_net.png   paired gross/net bars (mean & median)
    docs/figures/cost_breakdown.png where the cost goes, by liquidity bucket
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

# --- cost assumptions (mirrors config/costs.yaml) ---
BROKERAGE_BPS = 5.0
HALF_SPREAD_BPS = 5.0
SLIPPAGE_BPS = 10.0
EXCHANGE_BPS = 0.5
PER_SIDE_FIXED_BPS = BROKERAGE_BPS + HALF_SPREAD_BPS + SLIPPAGE_BPS + EXCHANGE_BPS  # 20.5
FIXED_RT_BPS = 2 * PER_SIDE_FIXED_BPS                                                # 41.0
BORROW_ANNUAL_BPS = 300.0
IMPACT_COEF = 0.10
IMPACT_EXP = 0.5
CLIPS = {"A$250k": 250_000, "A$500k": 500_000, "A$1m": 1_000_000}
BASE_CLIP = "A$500k"
EXITS = ["eff", "eff5", "eff10"]


def _load_price(tk):
    for d in (PROCESSED_RECONCILED_DIR / "prices", RAW_YAHOO_DIR):
        p = d / f"{tk}.csv"
        if p.exists():
            df = pd.read_csv(p, parse_dates=["date"]).sort_values("date").reset_index(drop=True)
            if not df.empty:
                return df
    return None


def trailing_vol(tk, entry_date, window=60):
    """Realised daily vol over the `window` bars ending at entry (fallback 3%)."""
    df = _load_price(tk)
    if df is None:
        return 0.03
    df = df[df["date"] <= pd.Timestamp(entry_date)]
    if len(df) < 20:
        return 0.03
    r = df["adjusted_close"].pct_change().dropna().tail(window)
    v = float(r.std())
    return v if np.isfinite(v) and v > 0 else 0.03


def impact_rt_bps(vol, adv, clip):
    if adv is None or adv <= 0:
        adv = 250_000.0
    part = clip / adv
    per_side = IMPACT_COEF * vol * (part ** IMPACT_EXP)   # fraction
    return 2 * per_side * 10_000                           # round-trip bps


def main() -> None:
    trades = pd.read_csv(
        OUTPUTS_DIR / "v2_trades.csv",
        parse_dates=["entry_date", "exit_eff_date", "exit_eff5_date", "exit_eff10_date"])

    # cache trailing vol per (ticker, entry)
    trades["daily_vol"] = [trailing_vol(t, d) for t, d in
                           zip(trades["ticker"], trades["entry_date"])]

    rows = []
    for _, r in trades.iterrows():
        for ex in EXITS:
            g = r.get(f"trade_{ex}_pct")
            xd = r.get(f"exit_{ex}_date")
            if pd.isna(g) or pd.isna(xd):
                continue
            hold_days = max(1, (pd.Timestamp(xd) - pd.Timestamp(r["entry_date"])).days)
            borrow = (BORROW_ANNUAL_BPS * hold_days / 365.0) if r["side"] == "short" else 0.0
            rec = {"tier": r["tier"], "side": r["side"], "exit": ex, "ticker": r["ticker"],
                   "gross_pct": float(g), "hold_days": hold_days,
                   "adv_aud": float(r["median_turnover_aud"]), "daily_vol": float(r["daily_vol"]),
                   "fixed_rt_bps": FIXED_RT_BPS, "borrow_bps": round(borrow, 2)}
            for cname, clip in CLIPS.items():
                imp = impact_rt_bps(r["daily_vol"], r["median_turnover_aud"], clip)
                cost = FIXED_RT_BPS + borrow + imp
                rec[f"impact_bps_{cname}"] = round(imp, 2)
                rec[f"cost_bps_{cname}"] = round(cost, 2)
                rec[f"net_pct_{cname}"] = round(float(g) - cost / 100.0, 3)
            rows.append(rec)
    pt = pd.DataFrame(rows)
    pt.to_csv(OUTPUTS_DIR / "friction_pertrade.csv", index=False)

    # ---- summary gross vs net (base clip) ----
    def agg(df, col):
        x = df[col].to_numpy(float)
        return len(x), x.mean(), np.median(x), 100 * (x > 0).mean()

    srows = []
    for tier in ["ASX200", "ASX100", "ALL"]:
        for side in ["short", "long"]:
            for ex in EXITS:
                sel = (pt["side"] == side) & (pt["exit"] == ex)
                if tier != "ALL":
                    sel &= (pt["tier"] == tier)
                d = pt[sel]
                if len(d) < 8:
                    continue
                n, gm, gmed, gw = agg(d, "gross_pct")
                _, nm, nmed, nw = agg(d, f"net_pct_{BASE_CLIP}")
                # significance of NET (does the edge survive costs?)
                nx = d[f"net_pct_{BASE_CLIP}"].to_numpy(float)
                try:
                    _, p_w = stats.wilcoxon(nx)
                except ValueError:
                    p_w = np.nan
                p_sign = stats.binomtest(int((nx > 0).sum()), len(nx), 0.5).pvalue
                srows.append({
                    "tier": tier, "side": side, "exit": ex, "n": n,
                    "gross_mean": round(gm, 3), "gross_median": round(gmed, 3), "gross_win": round(gw, 1),
                    "avg_cost_bps": round(d[f"cost_bps_{BASE_CLIP}"].mean(), 1),
                    "net_mean": round(nm, 3), "net_median": round(nmed, 3), "net_win": round(nw, 1),
                    "net_p_wilcoxon": round(float(p_w), 4) if pd.notna(p_w) else np.nan,
                    "net_p_sign": round(float(p_sign), 4),
                })
    summ = pd.DataFrame(srows)
    summ.to_csv(OUTPUTS_DIR / "friction_summary.csv", index=False)

    # ---- chart 1: gross vs net paired bars (median, the robust effect size) ----
    foc = summ[summ.tier.isin(["ASX200", "ALL"])].copy()
    foc["label"] = foc["tier"] + " " + foc["side"] + "\n@" + foc["exit"]
    foc = foc.sort_values(["tier", "side", "exit"]).reset_index(drop=True)
    x = np.arange(len(foc))
    fig, ax = plt.subplots(figsize=(14, 7))
    ax.bar(x - 0.2, foc["gross_median"], 0.38, color="#90caf9", label="GROSS median / trade")
    ax.bar(x + 0.2, foc["net_median"], 0.38, color="#1565c0", label=f"NET median / trade ({BASE_CLIP} clip)")
    for i, r in foc.iterrows():
        ax.annotate(f"{r['gross_median']:+.1f}", (i - 0.2, r["gross_median"]),
                    ha="center", va="bottom" if r["gross_median"] >= 0 else "top", fontsize=7)
        surv = (pd.notna(r["net_p_wilcoxon"]) and r["net_p_wilcoxon"] < 0.05)
        ax.annotate(f"{r['net_median']:+.1f}{'*' if surv else ''}\n−{r['avg_cost_bps']:.0f}bp",
                    (i + 0.2, r["net_median"]),
                    ha="center", va="bottom" if r["net_median"] >= 0 else "top", fontsize=7,
                    color="#0d47a1")
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xticks(x); ax.set_xticklabels(foc["label"], fontsize=8)
    ax.set_ylabel("Median return per trade (%)")
    ax.set_title("Gross vs net of costs — median per-trade return\n"
                 f"NET = gross − (41bp execution + market impact at {BASE_CLIP} + short borrow). "
                 "* = net edge still significant (Wilcoxon p<0.05)")
    ax.legend(); ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout(); fig.savefig(DOCS / "gross_vs_net.png", dpi=140); plt.close(fig)
    print("  -> docs/figures/gross_vs_net.png")

    # ---- chart 2: cost breakdown by liquidity bucket (ASX200 shorts) ----
    sh = pt[(pt.tier == "ASX200") & (pt.side == "short") & (pt.exit == "eff5")].copy()
    sh["adv_bucket"] = pd.qcut(sh["adv_aud"], 3, labels=["thin", "mid", "liquid"])
    fig, ax = plt.subplots(figsize=(10, 6))
    buckets = ["thin", "mid", "liquid"]
    fixed = [FIXED_RT_BPS] * 3
    borrow = [sh[sh.adv_bucket == b]["borrow_bps"].mean() for b in buckets]
    impact = [sh[sh.adv_bucket == b][f"impact_bps_{BASE_CLIP}"].mean() for b in buckets]
    xb = np.arange(3)
    ax.bar(xb, fixed, 0.55, color="#bdbdbd", label="execution (41bp)")
    ax.bar(xb, borrow, 0.55, bottom=fixed, color="#ffb74d", label="short borrow")
    ax.bar(xb, impact, 0.55, bottom=np.array(fixed) + np.array(borrow), color="#c62828",
           label=f"market impact ({BASE_CLIP})")
    for i in range(3):
        tot = fixed[i] + borrow[i] + impact[i]
        ax.annotate(f"{tot:.0f}bp", (i, tot), ha="center", va="bottom", fontweight="bold")
    ax.set_xticks(xb)
    ax.set_xticklabels([f"{b}\n(ADV {sh[sh.adv_bucket==b]['adv_aud'].median()/1e6:.1f}m)" for b in buckets])
    ax.set_ylabel("Round-trip cost (bps)")
    ax.set_title("Where the cost goes — ASX 200 short removals by liquidity bucket\n"
                 "Thin removals are far more expensive to short (impact scales with 1/√turnover)")
    ax.legend(); ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout(); fig.savefig(DOCS / "cost_breakdown.png", dpi=140); plt.close(fig)
    print("  -> docs/figures/cost_breakdown.png")

    # ---- console ----
    pd.set_option("display.width", 200)
    print("\nGROSS vs NET per-trade (base clip = %s):" % BASE_CLIP)
    print(summ.to_string(index=False))

    print("\nASX 200 short @eff5 — does the edge survive each order size?")
    d = pt[(pt.tier == "ASX200") & (pt.side == "short") & (pt.exit == "eff5")]
    print(f"  gross: mean {d['gross_pct'].mean():+.2f}%  median {d['gross_pct'].median():+.2f}%  "
          f"win {100*(d['gross_pct']>0).mean():.1f}%")
    for cname in CLIPS:
        nx = d[f"net_pct_{cname}"]
        _, pw = stats.wilcoxon(nx.to_numpy(float))
        print(f"  net {cname:>7}: mean {nx.mean():+.2f}%  median {nx.median():+.2f}%  "
              f"win {100*(nx>0).mean():.1f}%  avg cost {d[f'cost_bps_{cname}'].mean():.0f}bp  "
              f"Wilcoxon p={pw:.3f}")


if __name__ == "__main__":
    main()
