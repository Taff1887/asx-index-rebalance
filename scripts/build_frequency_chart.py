"""Trades-per-quarter frequency chart — spot missing / thin data.

The user asked: "plot a frequency chart of trades every quarter from the first
date to the end so i can see if we have missing data points."

We plot EVERY quarter on the x-axis from the first event quarter to the last
(continuous, so empty quarters are visible as gaps), and stack:

    valid additions (long, green)
    valid removals  (short, red)
    rejected: no price series  (dark grey, hatched)  <- survivorship hole
    rejected: other reasons    (light grey)

So you can immediately see (a) which quarters have no usable data and (b) how
much of each quarter was thrown away (and why).

Inputs : outputs/v2_trades.csv (valid), outputs/v2_rejected.csv (dropped)
Outputs: docs/figures/frequency_quarterly.png
         outputs/quarterly_counts.csv
"""

from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from asxrebalance.paths import OUTPUTS_DIR, REPO_ROOT  # noqa: E402

DOCS = REPO_ROOT / "docs" / "figures"
PDF_DIR = REPO_ROOT / "data" / "raw" / "marketindex" / "multi"
_MONTH_Q = {"march": 1, "june": 2, "september": 3, "december": 4}


def _quarter(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s).dt.to_period("Q")


def _pdf_covered_quarters() -> set:
    """Quarters for which we actually hold a source rebalance PDF."""
    covered = set()
    for p in PDF_DIR.glob("*-rebalance-*.pdf"):
        parts = p.stem.split("-rebalance-")
        if len(parts) != 2:
            continue
        year = parts[0]
        month = parts[1].split("-")[0]
        if month in _MONTH_Q and year.isdigit():
            covered.add(pd.Period(f"{year}Q{_MONTH_Q[month]}", freq="Q"))
    return covered


def main() -> None:
    valid = pd.read_csv(OUTPUTS_DIR / "v2_trades.csv")
    rej = pd.read_csv(OUTPUTS_DIR / "v2_rejected.csv")

    valid["q"] = _quarter(valid["announcement_date"])
    rej["q"] = _quarter(rej["announcement_date"])

    # full continuous quarter axis from first to last event
    qmin = min(valid["q"].min(), rej["q"].min())
    qmax = max(valid["q"].max(), rej["q"].max())
    quarters = pd.period_range(qmin, qmax, freq="Q")

    def count(df, mask):
        return df[mask].groupby("q").size().reindex(quarters, fill_value=0)

    v_long = count(valid, valid["side"] == "long")
    v_short = count(valid, valid["side"] == "short")
    r_missing = count(rej, rej["reason"] == "no_price_file")
    r_other = count(rej, rej["reason"] != "no_price_file")

    counts = pd.DataFrame({
        "valid_additions": v_long, "valid_removals": v_short,
        "rejected_no_price": r_missing, "rejected_other": r_other,
    })
    counts.index.name = "quarter"
    counts["total_events"] = counts.sum(axis=1)
    counts["valid_total"] = counts["valid_additions"] + counts["valid_removals"]
    counts.to_csv(OUTPUTS_DIR / "quarterly_counts.csv")

    covered = _pdf_covered_quarters()
    empty_q = counts.index[counts["total_events"] == 0]
    # Split empties: no source PDF (real data hole) vs PDF held but "No change".
    missing_pdf_q = [q for q in empty_q if q not in covered]
    no_change_q = [q for q in empty_q if q in covered]
    no_valid_q = counts.index[(counts["valid_total"] == 0) & (counts["total_events"] > 0)]

    x = np.arange(len(quarters))
    labels = [str(q) for q in quarters]

    fig, ax = plt.subplots(figsize=(16, 7.5))
    b = np.zeros(len(quarters))
    ax.bar(x, v_long.values, 0.82, bottom=b, color="#2e7d32", label="Valid additions (long)")
    b = b + v_long.values
    ax.bar(x, v_short.values, 0.82, bottom=b, color="#c62828", label="Valid removals (short)")
    b = b + v_short.values
    ax.bar(x, r_missing.values, 0.82, bottom=b, color="#37474f", hatch="//",
           edgecolor="white", linewidth=0.3, label="Rejected — no price series (survivorship)")
    b = b + r_missing.values
    ax.bar(x, r_other.values, 0.82, bottom=b, color="#bdbdbd",
           label="Rejected — other (illiquid / gap / reuse)")

    # Shade the contiguous missing-PDF region (source never archived).
    ymax = max(counts["total_events"].max(), 1)
    qlist = list(quarters)
    if missing_pdf_q:
        xs_missing = sorted(qlist.index(q) for q in missing_pdf_q)
        ax.axvspan(min(xs_missing) - 0.5, max(xs_missing) + 0.5,
                   color="#fdecea", zorder=0)
        ax.annotate("source PDFs never archived\n(S&P 403 / Wayback gap)",
                    ((min(xs_missing) + max(xs_missing)) / 2, ymax * 0.78),
                    ha="center", va="top", fontsize=9, color="#b71c1c",
                    fontweight="bold")
    # "No change" quarters (PDF held, but no 20/50/100/200 change) — legitimate.
    for q in no_change_q:
        xi = qlist.index(q)
        ax.annotate("no\nchange", (xi, 0.3), ha="center", va="bottom",
                    fontsize=7, color="#616161", style="italic")
    for q in no_valid_q:
        xi = qlist.index(q)
        ax.scatter([xi], [counts.loc[q, "total_events"] + 0.4], marker="v",
                   color="#b71c1c", s=30, zorder=5)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=90, fontsize=8)
    ax.set_ylabel("Number of index-rebalance events (ASX 20/50/100/200)")
    ax.set_title(
        "Rebalance events per quarter — coverage & missing data\n"
        f"{qmin} → {qmax} · {len(quarters)} quarters · "
        f"{int(counts['valid_total'].sum())} valid trades, "
        f"{int(counts[['rejected_no_price','rejected_other']].sum().sum())} rejected "
        f"({int(r_missing.sum())} have no price series)")
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(True, axis="y", alpha=0.3)
    ax.margins(x=0.01)
    fig.tight_layout()
    fig.savefig(DOCS / "frequency_quarterly.png", dpi=140)
    plt.close(fig)

    print(f"  -> docs/figures/frequency_quarterly.png")
    print(f"quarters spanned: {len(quarters)}  ({qmin} -> {qmax})")
    print(f"quarters with ZERO events: {len(empty_q)}"
          + (f"  {[str(q) for q in empty_q]}" if len(empty_q) else "  (none — every quarter has events)"))
    print(f"quarters with events but ZERO valid trades: {len(no_valid_q)}"
          + (f"  {[str(q) for q in no_valid_q]}" if len(no_valid_q) else ""))
    worst = counts.sort_values("rejected_no_price", ascending=False).head(6)
    print("\nQuarters losing the most names to 'no price series' (survivorship):")
    print(worst[["valid_total", "rejected_no_price", "rejected_other", "total_events"]].to_string())


if __name__ == "__main__":
    main()
