"""Fetch FMP historical market capitalisation for the ASX 200/300 boundary universe.

The pre-announcement predictor (run_predictor.py) ranks stocks by size near the
ASX 200 boundary, so it needs a market-cap history for every candidate name.
We fetch the boundary universe = every ticker that appears in an ASX 200 or ASX
300 rebalance event. Delisted names mostly return only stubs (same survivorship
limit as prices); live names return full history.

Output: data/raw/fmp_mktcap/<TICKER>.csv  (date, market_cap)
"""

from __future__ import annotations

import time

import pandas as pd
import requests
from dotenv import load_dotenv

load_dotenv()

from asxrebalance.config import get_env  # noqa: E402
from asxrebalance.paths import PROCESSED_LABELS_DIR, RAW_FMP_DIR  # noqa: E402

BASE = "https://financialmodelingprep.com/stable"
OUT = RAW_FMP_DIR.parent / "fmp_mktcap"


def fetch(sym: str, key: str):
    try:
        r = requests.get(f"{BASE}/historical-market-capitalization",
                         params={"symbol": sym, "from": "2011-01-01", "to": "2026-02-01",
                                 "limit": 6000, "apikey": key}, timeout=30)
        if r.status_code == 200 and isinstance(r.json(), list):
            return r.json()
    except requests.RequestException:
        return None
    return None


def main() -> None:
    key = get_env("FMP_API_KEY")
    OUT.mkdir(parents=True, exist_ok=True)
    labels = pd.read_csv(PROCESSED_LABELS_DIR / "rebalance_labels.csv")
    uni = sorted(labels[labels["index"].isin(["ASX200", "ASX300"])]["ticker"].astype(str).unique())
    # prioritise names we already have prices for (more likely to resolve)
    have = {p.stem for p in RAW_FMP_DIR.glob("*.csv")}
    uni = [t for t in uni if t in have] + [t for t in uni if t not in have]

    got, stub, fail = 0, 0, 0
    for i, t in enumerate(uni):
        dest = OUT / f"{t}.csv"
        if dest.exists() and dest.stat().st_size > 200:
            got += 1
            continue
        data = fetch(f"{t}.AX", key)
        if data and len(data) > 50:
            df = pd.DataFrame(data)[["date", "marketCap"]].rename(columns={"marketCap": "market_cap"})
            df["date"] = pd.to_datetime(df["date"])
            df.sort_values("date").to_csv(dest, index=False)
            got += 1
        elif data:
            stub += 1
        else:
            fail += 1
        if i % 100 == 0:
            print(f"  {i}/{len(uni)}  got={got} stub={stub} fail={fail}")
        time.sleep(0.05)
    print(f"\nDONE: {got} market-cap series, {stub} stubs, {fail} unavailable, of {len(uni)} universe")


if __name__ == "__main__":
    main()
