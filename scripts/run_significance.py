"""Is the rebalance signal a REAL edge, or just noise / luck?

For each tier x side x exit-horizon we run five tests on the per-trade returns:

  1. one-sample t-test vs 0          (parametric: is the MEAN != 0?)
  2. Wilcoxon signed-rank vs 0       (non-parametric: is the MEDIAN != 0?)
  3. bootstrap 95% CI of the mean    (does the CI exclude 0?)
  4. sign / binomial test            (is the WIN RATE != 50%?)
  5. PLACEBO test (the luck test)    (key one — see below)

The placebo test is the one that separates a real index-effect from "these
stocks just drifted." For every real trade we keep the SAME ticker, the SAME
side (long/short) and the SAME holding length in trading days, but we slide the
entry to a RANDOM date in that stock's history. Repeating this B times builds a
null distribution of the strategy's mean return under random timing. If the real
(rebalance-timed) mean sits out in the tail of that null, the edge is about the
EVENT, not about the stocks. If it sits in the middle, it's luck / drift.

Deterministic: fixed RNG seed.  Everything in percent.  Real prices only.

Outputs:
    outputs/signal_stats.csv             full table of every test
    docs/figures/signal_significance.png mean +/- bootstrap CI, flagged by p
    docs/figures/placebo_null.png        ASX 200 short eff5: real vs luck null
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
RNG = np.random.default_rng(20240609)
N_BOOT = 10_000
N_PLACEBO = 5_000
EXITS = ["eff", "eff5", "eff10"]


def _load_price(tk):
    for d in (PROCESSED_RECONCILED_DIR / "prices", RAW_YAHOO_DIR):
        p = d / f"{tk}.csv"
        if p.exists():
            df = pd.read_csv(p, parse_dates=["date"]).sort_values("date").reset_index(drop=True)
            if not df.empty:
                return df
    return None


def bootstrap_ci(x, n=N_BOOT, alpha=0.05):
    x = np.asarray(x, float)
    idx = RNG.integers(0, len(x), size=(n, len(x)))
    means = x[idx].mean(axis=1)
    return np.percentile(means, [100 * alpha / 2, 100 * (1 - alpha / 2)])


def placebo_pvalue(trades_sub, exit_lbl):
    """Null = same stocks/side/holding-length, random entry date.

    Returns (observed_mean, null_means_array, p_one_sided).
    p is the fraction of random-timed books whose mean is at least as extreme
    (in the direction of the observed edge) as the real book.
    """
    # Build per-trade (price-array, side-sign, holding_bars).
    recs = []
    px_cache = {}
    for _, r in trades_sub.iterrows():
        tk = r["ticker"]
        if tk not in px_cache:
            df = _load_price(tk)
            px_cache[tk] = df["adjusted_close"].to_numpy(float) if df is not None else None
        arr = px_cache[tk]
        if arr is None or len(arr) < 30:
            continue
        # holding length in trading bars from the realised entry/exit dates
        ed = pd.Timestamp(r["entry_date"])
        xd = pd.Timestamp(r[f"exit_{exit_lbl}_date"]) if pd.notna(r.get(f"exit_{exit_lbl}_date")) else None
        if xd is None:
            continue
        df = _load_price(tk)
        dts = df["date"].to_numpy()
        try:
            i0 = int(np.where(dts == np.datetime64(ed))[0][0])
            i1 = int(np.where(dts == np.datetime64(xd))[0][0])
        except IndexError:
            continue
        hold = i1 - i0
        sgn = 1.0 if r["side"] == "long" else -1.0
        if hold > 0:
            recs.append((arr, sgn, hold))
    if len(recs) < 5:
        return None
    # observed mean is taken from the real per-trade column for consistency
    obs = float(trades_sub[f"trade_{exit_lbl}_pct"].dropna().mean())

    null_means = np.empty(N_PLACEBO)
    for b in range(N_PLACEBO):
        rets = np.empty(len(recs))
        for j, (arr, sgn, hold) in enumerate(recs):
            hi = len(arr) - hold - 1
            s = RNG.integers(0, hi) if hi > 0 else 0
            rets[j] = sgn * (arr[s + hold] / arr[s] - 1.0) * 100.0
        null_means[b] = rets.mean()
    # one-sided in the direction of the observed edge
    if obs >= 0:
        p = (1 + np.sum(null_means >= obs)) / (N_PLACEBO + 1)
    else:
        p = (1 + np.sum(null_means <= obs)) / (N_PLACEBO + 1)
    return obs, null_means, float(p)


def main() -> None:
    trades = pd.read_csv(
        OUTPUTS_DIR / "v2_trades.csv",
        parse_dates=["entry_date", "exit_eff_date", "exit_eff5_date", "exit_eff10_date"])

    groups = [("ASX200", "short"), ("ASX200", "long"),
              ("ASX100", "short"), ("ASX100", "long"),
              ("ALL", "short"), ("ALL", "long")]

    rows = []
    for tier, side in groups:
        sel = (trades["side"] == side)
        if tier != "ALL":
            sel &= (trades["tier"] == tier)
        sub = trades[sel]
        for ex in EXITS:
            x = sub[f"trade_{ex}_pct"].dropna().to_numpy(float)
            if len(x) < 8:
                continue
            t, p_t = stats.ttest_1samp(x, 0.0)
            try:
                w, p_w = stats.wilcoxon(x)
            except ValueError:
                w, p_w = np.nan, np.nan
            lo, hi = bootstrap_ci(x)
            wins = int((x > 0).sum())
            p_sign = stats.binomtest(wins, len(x), 0.5).pvalue
            rows.append({
                "tier": tier, "side": side, "exit": ex, "n": len(x),
                "mean_pct": round(float(x.mean()), 3),
                "median_pct": round(float(np.median(x)), 3),
                "std_pct": round(float(x.std(ddof=1)), 3),
                "win_rate_pct": round(100 * wins / len(x), 1),
                "t_stat": round(float(t), 2), "p_ttest": round(float(p_t), 4),
                "p_wilcoxon": round(float(p_w), 4) if pd.notna(p_w) else np.nan,
                "boot_ci_lo": round(float(lo), 3), "boot_ci_hi": round(float(hi), 3),
                "ci_excludes_0": bool(lo > 0 or hi < 0),
                "p_sign": round(float(p_sign), 4),
            })
    stat = pd.DataFrame(rows)

    # ---- placebo (luck) test on the headline cells ----
    placebo_results = {}
    for tier, side, ex in [("ASX200", "short", "eff5"), ("ASX200", "short", "eff10"),
                           ("ASX200", "long", "eff5"), ("ALL", "short", "eff5")]:
        sel = (trades["side"] == side)
        if tier != "ALL":
            sel &= (trades["tier"] == tier)
        res = placebo_pvalue(trades[sel], ex)
        if res is not None:
            obs, null_means, p = res
            placebo_results[(tier, side, ex)] = (obs, null_means, p)
            mask = (stat.tier == tier) & (stat.side == side) & (stat.exit == ex)
            stat.loc[mask, "placebo_null_mean"] = round(float(null_means.mean()), 3)
            stat.loc[mask, "p_placebo"] = round(p, 4)

    stat.to_csv(OUTPUTS_DIR / "signal_stats.csv", index=False)

    # ---- chart 1: mean +/- bootstrap CI, flagged significant ----
    focus = stat[(stat.tier.isin(["ASX200", "ALL"]))].copy()
    focus["label"] = focus["tier"] + " " + focus["side"] + "\n@" + focus["exit"]
    focus = focus.sort_values(["tier", "side", "exit"])
    # Per-trade returns are fat-tailed, so the MEAN's CI is wide; the reliable
    # evidence is the non-parametric trio (median via Wilcoxon, win rate via the
    # sign test, and the placebo/luck test). Colour by that robust verdict.
    def robust_sig(r):
        votes = 0
        if pd.notna(r["p_wilcoxon"]) and r["p_wilcoxon"] < 0.05:
            votes += 1
        if r["p_sign"] < 0.05:
            votes += 1
        if pd.notna(r.get("p_placebo")) and r["p_placebo"] < 0.05:
            votes += 1
        # require the placebo test to pass when we ran it, else >=2 of the trio
        if pd.notna(r.get("p_placebo")):
            return (r["p_placebo"] < 0.05) and votes >= 2
        return votes >= 2

    fig, ax = plt.subplots(figsize=(13, 6.8))
    x = np.arange(len(focus))
    for i, (_, r) in enumerate(focus.iterrows()):
        sig = robust_sig(r)
        col = ("#2e7d32" if r["median_pct"] > 0 else "#c62828") if sig else "#bdbdbd"
        ax.bar(i, r["mean_pct"], 0.62, color=col)
        ax.errorbar(i, r["mean_pct"],
                    yerr=[[r["mean_pct"] - r["boot_ci_lo"]], [r["boot_ci_hi"] - r["mean_pct"]]],
                    fmt="none", ecolor="black", capsize=4, lw=1.3)
        star = " *" if sig else ""
        pp = f"\npp={r['p_placebo']:.3f}" if pd.notna(r.get("p_placebo")) else ""
        ax.annotate(f"med {r['median_pct']:+.1f}{star}\n{r['win_rate_pct']:.0f}% win{pp}",
                    (i, r["boot_ci_hi"] + 0.2 if r["mean_pct"] >= 0 else r["boot_ci_lo"] - 0.2),
                    ha="center", va="bottom" if r["mean_pct"] >= 0 else "top", fontsize=7)
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xticks(x); ax.set_xticklabels(focus["label"], fontsize=8)
    ax.set_ylabel("Mean per-trade return (%)  —  bar = mean, whisker = bootstrap 95% CI")
    ax.set_title("Are the signals real or luck?  Per-trade mean (±bootstrap 95% CI), labelled with median, win rate, placebo p\n"
                 "Coloured = robustly real (passes placebo + median/sign tests); grey = not distinguishable from luck. "
                 "Mean CIs are wide (fat tails) — the median/win-rate/placebo are the reliable evidence.")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout(); fig.savefig(DOCS / "signal_significance.png", dpi=140); plt.close(fig)
    print("  -> docs/figures/signal_significance.png")

    # ---- chart 2: placebo null vs observed (the luck picture) ----
    key = ("ASX200", "short", "eff5")
    if key in placebo_results:
        obs, null_means, p = placebo_results[key]
        fig, ax = plt.subplots(figsize=(11, 6))
        ax.hist(null_means, bins=60, color="#90a4ae", edgecolor="white",
                label=f"random-timing null (n={N_PLACEBO:,} books)")
        ax.axvline(float(null_means.mean()), color="#37474f", ls=":", lw=1.5,
                   label=f"null mean {null_means.mean():+.2f}% (just holding these stocks)")
        ax.axvline(obs, color="#c62828", lw=2.5,
                   label=f"REAL rebalance-timed mean {obs:+.2f}%  (p={p:.3f})")
        ax.set_xlabel("Mean per-trade return of a book (%)")
        ax.set_ylabel("Frequency")
        ax.set_title("The luck test — ASX 200 short removals, exit eff+5\n"
                     "Does the edge come from the REBALANCE timing, or just from shorting these stocks?")
        ax.legend(fontsize=9)
        fig.tight_layout(); fig.savefig(DOCS / "placebo_null.png", dpi=140); plt.close(fig)
        print("  -> docs/figures/placebo_null.png")

    # ---- console summary ----
    pd.set_option("display.width", 200)
    show = ["tier", "side", "exit", "n", "mean_pct", "median_pct", "win_rate_pct",
            "t_stat", "p_ttest", "p_wilcoxon", "boot_ci_lo", "boot_ci_hi",
            "ci_excludes_0", "p_sign", "p_placebo"]
    print("\nSIGNAL SIGNIFICANCE (per-trade returns):")
    print(stat[show].to_string(index=False))
    print("\nPlacebo (luck) test — real rebalance-timed mean vs random-timing null:")
    for (tier, side, ex), (obs, nm, p) in placebo_results.items():
        verdict = "REAL (beats luck)" if p < 0.05 else "NOT distinguishable from luck"
        print(f"  {tier} {side} @{ex}: real {obs:+.2f}%  vs null {nm.mean():+.2f}%  "
              f"p={p:.4f}  -> {verdict}")


if __name__ == "__main__":
    main()
