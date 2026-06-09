"""Download the FULL dated S&P/ASX 200 announcement archive from marketindex.

marketindex.com.au/asx200/announcements lists ~156 dated single-event PDFs going
back to 2011: quarterly rebalances AND off-cycle additions / removals / demergers
/ no-change notices. The earlier pipeline only had the 53 multi-index *quarterly*
PDFs, so every OFF-CYCLE index change (mostly M&A-driven removals and their
replacement additions — i.e. exactly where the delisted names live) was missing.

This grabs the whole dated archive to data/raw/marketindex/asx200/, plus the
handful of 2020-2021 quarterly rebalances that marketindex only links to as ASX
newswire pages (resolved to direct PDFs where known).

Outputs: data/raw/marketindex/asx200/<name>.pdf  (+ a manifest CSV)
"""

from __future__ import annotations

import re
import time

import pandas as pd
import requests

from asxrebalance.paths import REPO_ROOT

OUT = REPO_ROOT / "data" / "raw" / "marketindex" / "asx200"
H = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                   "(KHTML, like Gecko) Chrome/120 Safari/537.36", "Accept": "text/html"}
LIST_URL = "https://www.marketindex.com.au/asx200/announcements"

# Quarterly rebalances that marketindex links only via the iguana2 newswire.
# Resolved to direct, downloadable PDFs where a stable URL is known.
EXTRA_QUARTERLY = {
    "20210312-asx200-quarterly-march2021.pdf":
        "https://www.asx.com.au/asxpdf/20210312/pdf/44tm5n0xpsn800.pdf",
    "20210903-asx200-quarterly-september2021.pdf":
        "https://www.spglobal.com/spdji/en/documents/indexnews/announcements/"
        "20210903-1443002/1443002_20210903-quarta-200.pdf",
}


def list_pdfs() -> list[str]:
    r = requests.get(LIST_URL, headers=H, timeout=60)
    r.raise_for_status()
    pdfs = sorted(set(re.findall(r"https?://[^\s\"'<>]+?\.pdf", r.text)))
    return [p for p in pdfs if "asx200" in p]


def fetch(url: str, dest) -> int | None:
    for attempt in range(3):
        try:
            r = requests.get(url, headers={"User-Agent": H["User-Agent"]}, timeout=60)
            if r.status_code == 200 and r.content[:5] == b"%PDF-":
                dest.write_bytes(r.content)
                return len(r.content)
            return None
        except requests.RequestException:
            time.sleep(1.5 * (attempt + 1))
    return None


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    pdfs = list_pdfs()
    print(f"{len(pdfs)} dated ASX 200 PDFs listed on marketindex")

    rows, got, miss = [], 0, []
    for url in pdfs:
        name = url.rsplit("/", 1)[-1]
        dest = OUT / name
        if dest.exists() and dest.stat().st_size > 3000:
            rows.append({"name": name, "url": url, "bytes": dest.stat().st_size, "source": "cache"})
            got += 1
            continue
        n = fetch(url, dest)
        if n:
            rows.append({"name": name, "url": url, "bytes": n, "source": "marketindex"})
            got += 1
        else:
            miss.append(name)

    for name, url in EXTRA_QUARTERLY.items():
        dest = OUT / name
        n = dest.stat().st_size if (dest.exists() and dest.stat().st_size > 3000) else fetch(url, dest)
        if n:
            rows.append({"name": name, "url": url, "bytes": n, "source": "quarterly-direct"})
            got += 1
            print(f"  + quarterly {name} ({n:,}b)")
        else:
            miss.append(name)

    pd.DataFrame(rows).to_csv(OUT / "_manifest.csv", index=False)
    print(f"\nDownloaded/cached {got} PDFs to {OUT.relative_to(REPO_ROOT)}")
    if miss:
        print(f"Could not fetch ({len(miss)}): {miss}")

    # quick action-type tally from filenames
    kinds = {}
    for r in rows:
        m = re.search(r"-asx200-([a-z]+)", r["name"])
        k = m.group(1) if m else "other"
        kinds[k] = kinds.get(k, 0) + 1
    print("by type:", dict(sorted(kinds.items(), key=lambda x: -x[1])))


if __name__ == "__main__":
    main()
