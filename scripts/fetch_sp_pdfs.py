"""Fetch every S&P/ASX 200 quarterly rebalance announcement PDF from
files.marketindex.com.au. Probes all 68 announcement dates (first Friday
of Mar/Jun/Sep/Dec, 2009-2025).
"""

from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path

import requests

from asxrebalance.paths import REPO_ROOT

OUT = REPO_ROOT / "data" / "raw" / "marketindex"
OUT.mkdir(parents=True, exist_ok=True)

HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}


def first_friday(year: int, month: int) -> date:
    d = date(year, month, 1)
    return d + timedelta(days=(4 - d.weekday()) % 7)


def main() -> None:
    # Announcement dates vary across years: usually first or second Friday of
    # the rebalance month, sometimes Wednesday/Thursday. Probe ALL Fridays
    # and weekdays in the first half of each rebalance month.
    candidates: list[date] = []
    for year in range(2009, 2027):
        for month in (3, 6, 9, 12):
            ff = first_friday(year, month)
            # Try first Friday, +/- 1 day, +1 week, +2 weeks.
            for offset in (-1, 0, 1, 7, 8, 14):
                candidates.append(ff + timedelta(days=offset))

    hits, misses = [], []
    seen_eff = set()
    for d in candidates:
        ymd = d.strftime("%Y%m%d")
        # Try a few naming variants the site has used over time.
        variants = [
            f"{ymd}-asx200-rebalance.pdf",
            f"{ymd}-asx-200-rebalance.pdf",
            f"{ymd}-asx-rebalance.pdf",
            f"{ymd}-quarterly-rebalance.pdf",
            f"{ymd}-rebalance.pdf",
            # Try the next-business-day variant (some early years).
            (d + timedelta(days=3)).strftime("%Y%m%d") + "-asx200-rebalance.pdf",
        ]
        got = False
        for v in variants:
            url = f"https://files.marketindex.com.au/files/announcements/{v}"
            try:
                r = requests.get(url, headers=HEADERS, timeout=15, allow_redirects=True)
            except requests.RequestException as exc:
                continue
            if r.status_code == 200 and len(r.content) > 5000:
                target = OUT / f"{ymd}-asx200-rebalance.pdf"
                if target.exists() and target.stat().st_size == len(r.content):
                    got = True
                    break
                target.write_bytes(r.content)
                hits.append((str(d), len(r.content), v))
                print(f"  OK  {d}  {v}  ({len(r.content):,} bytes)")
                got = True
                break
        if not got:
            misses.append(str(d))

    print(f"\nHits: {len(hits)} / {len(candidates)}")
    if misses:
        print(f"Misses ({len(misses)}):")
        for m in misses:
            print(f"  {m}")


if __name__ == "__main__":
    main()
