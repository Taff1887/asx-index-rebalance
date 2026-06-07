"""FMP vs Yahoo cross-validation.

Every comparison preserves the raw values from both sources and labels each
discrepancy with a severity bucket. The output of this module feeds the
reconciliation layer; reconciliation is intentionally a separate step so the
provenance of every reconciled value remains auditable.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..config import load_validation_config


@dataclass(frozen=True)
class Thresholds:
    price_low: float
    price_med: float
    price_high: float
    vol_low: float
    vol_med: float
    vol_high: float
    stale_days: int
    suspicious_ret_pct: float
    min_coverage_pct: float


def load_thresholds() -> Thresholds:
    cfg = load_validation_config()["validation"]
    return Thresholds(
        price_low=float(cfg["price_diff_threshold_pct_low"]),
        price_med=float(cfg["price_diff_threshold_pct_medium"]),
        price_high=float(cfg["price_diff_threshold_pct_high"]),
        vol_low=float(cfg["volume_diff_threshold_pct_low"]),
        vol_med=float(cfg["volume_diff_threshold_pct_medium"]),
        vol_high=float(cfg["volume_diff_threshold_pct_high"]),
        stale_days=int(cfg["stale_price_days"]),
        suspicious_ret_pct=float(cfg["suspicious_daily_return_pct"]),
        min_coverage_pct=float(cfg["min_required_source_coverage_pct"]),
    )


def _severity_pct(diff_pct: float, low: float, med: float, high: float) -> str:
    a = abs(diff_pct)
    if np.isnan(a):
        return "missing"
    if a >= high:
        return "high"
    if a >= med:
        return "medium"
    if a >= low:
        return "low"
    return "none"


def _merged(fmp_df: pd.DataFrame, yahoo_df: pd.DataFrame, fields: list[str]) -> pd.DataFrame:
    keep = ["date", "ticker", *fields]
    a = fmp_df[keep].rename(columns={f: f"{f}_fmp" for f in fields})
    b = yahoo_df[keep].rename(columns={f: f"{f}_yahoo" for f in fields})
    return a.merge(b, on=["date", "ticker"], how="outer")


def compare_price_sources(fmp_df: pd.DataFrame, yahoo_df: pd.DataFrame,
                          thresholds: Thresholds | None = None) -> pd.DataFrame:
    """Compare close and adjusted close for each (date, ticker)."""
    th = thresholds or load_thresholds()
    merged = _merged(fmp_df, yahoo_df, ["close", "adjusted_close"])

    out_rows = []
    for field in ("close", "adjusted_close"):
        f_col, y_col = f"{field}_fmp", f"{field}_yahoo"
        sub = merged[["date", "ticker", f_col, y_col]].copy()
        sub["abs_diff"] = (sub[f_col] - sub[y_col]).abs()
        denom = sub[[f_col, y_col]].abs().max(axis=1)
        sub["pct_diff"] = np.where(denom > 0, 100.0 * sub["abs_diff"] / denom, np.nan)
        sub["severity"] = sub["pct_diff"].apply(
            lambda v: _severity_pct(v, th.price_low, th.price_med, th.price_high)
        )
        sub["field"] = field
        sub = sub.rename(columns={f_col: "fmp_value", y_col: "yahoo_value"})
        out_rows.append(sub[["date", "ticker", "field", "fmp_value", "yahoo_value",
                             "abs_diff", "pct_diff", "severity"]])
    return pd.concat(out_rows, ignore_index=True)


def compare_volume_sources(fmp_df: pd.DataFrame, yahoo_df: pd.DataFrame,
                           thresholds: Thresholds | None = None) -> pd.DataFrame:
    th = thresholds or load_thresholds()
    merged = _merged(fmp_df, yahoo_df, ["volume"])
    merged["abs_diff"] = (merged["volume_fmp"] - merged["volume_yahoo"]).abs()
    denom = merged[["volume_fmp", "volume_yahoo"]].abs().max(axis=1)
    merged["pct_diff"] = np.where(denom > 0, 100.0 * merged["abs_diff"] / denom, np.nan)
    merged["severity"] = merged["pct_diff"].apply(
        lambda v: _severity_pct(v, th.vol_low, th.vol_med, th.vol_high)
    )
    merged["field"] = "volume"
    return merged.rename(columns={
        "volume_fmp": "fmp_value", "volume_yahoo": "yahoo_value",
    })[["date", "ticker", "field", "fmp_value", "yahoo_value",
        "abs_diff", "pct_diff", "severity"]]


def detect_missing_observations(fmp_df: pd.DataFrame, yahoo_df: pd.DataFrame) -> pd.DataFrame:
    """Report (date, ticker) pairs present in one source but absent in the other."""
    a = fmp_df[["date", "ticker"]].assign(in_fmp=True)
    b = yahoo_df[["date", "ticker"]].assign(in_yahoo=True)
    m = a.merge(b, on=["date", "ticker"], how="outer")
    m["in_fmp"] = m["in_fmp"].fillna(False)
    m["in_yahoo"] = m["in_yahoo"].fillna(False)
    missing = m.loc[~(m["in_fmp"] & m["in_yahoo"])].copy()
    missing["missing_from"] = np.where(missing["in_fmp"], "yahoo", "fmp")
    return missing[["date", "ticker", "missing_from"]].reset_index(drop=True)


def detect_stale_prices(df: pd.DataFrame, thresholds: Thresholds | None = None) -> pd.DataFrame:
    """Flag tickers with the same close for >= stale_days consecutive trading days."""
    th = thresholds or load_thresholds()
    df = df.sort_values(["ticker", "date"]).copy()
    df["stale_run"] = (
        df.groupby("ticker")["close"]
        .transform(lambda s: s.eq(s.shift()).astype(int).groupby(
            (s.ne(s.shift())).cumsum()
        ).cumcount() + 1)
    )
    flagged = df.loc[df["stale_run"] >= th.stale_days,
                     ["date", "ticker", "close", "stale_run"]].copy()
    flagged["issue"] = "stale_price"
    return flagged.reset_index(drop=True)


def detect_suspicious_jumps(df: pd.DataFrame, thresholds: Thresholds | None = None) -> pd.DataFrame:
    """Flag days with abs(return) exceeding the configured percent threshold."""
    th = thresholds or load_thresholds()
    df = df.sort_values(["ticker", "date"]).copy()
    df["ret"] = df.groupby("ticker")["adjusted_close"].pct_change()
    suspicious = df.loc[df["ret"].abs() * 100.0 >= th.suspicious_ret_pct,
                        ["date", "ticker", "adjusted_close", "ret"]].copy()
    suspicious["issue"] = "suspicious_jump"
    return suspicious.reset_index(drop=True)


def detect_corporate_action_mismatches(fmp_df: pd.DataFrame,
                                       yahoo_df: pd.DataFrame) -> pd.DataFrame:
    """Detect cases where (close / adjusted_close) diverges between sources.

    A different split/dividend treatment between Yahoo and FMP shows up as a
    persistent ratio gap. We flag rows where the ratio differs by more than 1%.
    """
    cols = ["date", "ticker", "close", "adjusted_close"]
    merged = fmp_df[cols].rename(columns={"close": "close_fmp",
                                          "adjusted_close": "adj_fmp"}).merge(
        yahoo_df[cols].rename(columns={"close": "close_yahoo",
                                       "adjusted_close": "adj_yahoo"}),
        on=["date", "ticker"], how="inner",
    )
    ratio_fmp = merged["adj_fmp"] / merged["close_fmp"]
    ratio_yahoo = merged["adj_yahoo"] / merged["close_yahoo"]
    merged["ratio_diff"] = (ratio_fmp - ratio_yahoo).abs()
    flagged = merged.loc[merged["ratio_diff"] > 0.01].copy()
    flagged["issue"] = "corporate_action_mismatch"
    return flagged[["date", "ticker", "ratio_diff", "issue"]].reset_index(drop=True)


def generate_data_quality_report(fmp_df: pd.DataFrame, yahoo_df: pd.DataFrame,
                                 thresholds: Thresholds | None = None) -> dict:
    th = thresholds or load_thresholds()
    price = compare_price_sources(fmp_df, yahoo_df, th)
    volume = compare_volume_sources(fmp_df, yahoo_df, th)
    missing = detect_missing_observations(fmp_df, yahoo_df)
    stale = detect_stale_prices(fmp_df, th)
    jumps = detect_suspicious_jumps(fmp_df, th)
    corp = detect_corporate_action_mismatches(fmp_df, yahoo_df)

    issues = pd.concat([
        price.assign(check="price_diff"),
        volume.assign(check="volume_diff"),
        missing.assign(check="missing_observation", severity=np.where(
            missing["missing_from"].notna(), "medium", "none"
        )),
        stale[["date", "ticker", "issue"]].assign(check="stale_price", severity="medium"),
        jumps[["date", "ticker", "issue"]].assign(check="suspicious_jump", severity="high"),
        corp[["date", "ticker", "issue"]].assign(check="corporate_action_mismatch",
                                                  severity="high"),
    ], ignore_index=True, sort=False)

    summary_rows = []
    for check, grp in issues.groupby("check"):
        for sev, sub in grp.groupby(grp.get("severity", "none")):
            summary_rows.append({"check": check, "severity": sev, "count": len(sub)})
    summary = pd.DataFrame(summary_rows)

    return {
        "summary": summary,
        "issues": issues,
        "price_differences": price,
        "volume_differences": volume,
        "missing": missing,
        "stale": stale,
        "jumps": jumps,
        "corp_actions": corp,
    }


__all__ = [
    "Thresholds",
    "load_thresholds",
    "compare_price_sources",
    "compare_volume_sources",
    "detect_missing_observations",
    "detect_stale_prices",
    "detect_suspicious_jumps",
    "detect_corporate_action_mismatches",
    "generate_data_quality_report",
]
