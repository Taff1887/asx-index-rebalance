"""Parse S&P/ASX rebalance PDFs and extract events for ALL index tiers.

Each PDF contains separate tables for the S&P/ASX 20, 50, 100, 200, 300 and
All Australian / All Technology indices. We capture every (ticker, action,
index) tuple along with the announcement and effective dates.

Output:
    data/processed/labels/rebalance_labels.csv  - one row per event
        announcement_date, effective_date, index, action, ticker, company_name

Run after `python scripts/fetch_sp_pdfs.py` has populated data/raw/marketindex/.
"""

from __future__ import annotations

import re
from datetime import date, datetime
from pathlib import Path

import pandas as pd
import pypdf

from asxrebalance.paths import OUTPUTS_DIR, PROCESSED_LABELS_DIR, REPO_ROOT

PDF_DIR = REPO_ROOT / "data" / "raw" / "marketindex" / "multi"

# Header dates
ANNOUNCEMENT_RE = re.compile(
    r"SYDNEY,?\s+"
    r"(JANUARY|FEBRUARY|MARCH|APRIL|MAY|JUNE|JULY|AUGUST|SEPTEMBER|OCTOBER|NOVEMBER|DECEMBER|"
    r"Jan\.?|Feb\.?|Mar\.?|Apr\.?|May\.?|Jun\.?|Jul\.?|Aug\.?|Sep\.?|Oct\.?|Nov\.?|Dec\.?)"
    r"\s+(\d{1,2}),?\s*(\d{4})",
    re.IGNORECASE,
)

# Effective date is usually in the first paragraph: "effective ... on Month Day, Year"
EFFECTIVE_RE = re.compile(
    r"effective[^,]*?(?:on\s+(?:\w+,\s*)?|after\s+the\s+close\s+of\s+trading\s+on\s+)"
    r"(January|February|March|April|May|June|July|August|September|October|November|December|"
    r"Jan\.?|Feb\.?|Mar\.?|Apr\.?|May\.?|Jun\.?|Jul\.?|Aug\.?|Sep\.?|Oct\.?|Nov\.?|Dec\.?)"
    r"\s+(\d{1,2}),?\s*(\d{4})",
    re.IGNORECASE,
)

# Section header: "S&P/ASX <N> Index – ..." (rest of line varies year to year).
# Some headers carry "Effective ... on Month Day, Year", some carry
# "Month Day, Year After Market Close", some say "No Change". We capture
# the index name; the date is extracted with a separate regex on the
# header text if present.
INDEX_HEADER_RE = re.compile(
    r"S&P/ASX\s*(?P<idx>300|200|100|50|20|All\s+Australian\s+200|All\s+Australian\s+50|"
    r"All\s+Technology|All\s+Ordinaries)\b"
    r"\s*Index\s*[^\w\s][\s]*(?P<rest>[^\n]*)",
    re.IGNORECASE,
)

# Pull a date out of a section header line.
SECTION_DATE_RE = re.compile(
    r"(January|February|March|April|May|June|July|August|September|October|November|December|"
    r"Jan\.?|Feb\.?|Mar\.?|Apr\.?|May\.?|Jun\.?|Jul\.?|Aug\.?|Sep\.?|Oct\.?|Nov\.?|Dec\.?)"
    r"\s+(\d{1,2}),?\s*(\d{4})",
    re.IGNORECASE,
)

ACTION_RE = re.compile(
    r"^\s*"
    r"(?:S&P/ASX\s*(?:20|50|100|200|300)\s+)?"
    r"(Addition|Removal|Deletion|Inclusion|Exclusion)\s+"
    r"([A-Z0-9]{1,6})\s+(.+?)\s*$",
    re.IGNORECASE,
)

_MONTH_NORM = {
    "jan": "January", "feb": "February", "mar": "March", "apr": "April",
    "may": "May", "jun": "June", "jul": "July", "aug": "August",
    "sep": "September", "oct": "October", "nov": "November", "dec": "December",
}


def _norm_month(s: str) -> str:
    return _MONTH_NORM.get(s.strip().lower().rstrip(".")[:3], s.title())


def _parse_date(month: str, day: str, year: str) -> date:
    return datetime.strptime(f"{_norm_month(month)} {day} {year}", "%B %d %Y").date()


def _norm_index_name(name: str) -> str:
    n = name.strip().upper().replace(" ", "")
    if n in ("20", "50", "100", "200", "300"):
        return f"ASX{n}"
    if "ALLTECH" in n:
        return "ALLTECH"
    if "ALLORDINARIES" in n or "ALLORDS" in n:
        return "ALLORDS"
    if "ALLAUSTRALIAN" in n:
        return f"ALLAUS{re.sub(r'[^0-9]','', n)}"
    return n


def parse_pdf(path: Path) -> dict:
    reader = pypdf.PdfReader(str(path))
    full_text = "\n".join((p.extract_text() or "") for p in reader.pages)

    ann_match = ANNOUNCEMENT_RE.search(full_text)
    eff_match = EFFECTIVE_RE.search(full_text)
    if not ann_match:
        return {"path": path.name, "error": "no announcement date in header"}
    announcement = _parse_date(ann_match.group(1), ann_match.group(2), ann_match.group(3))

    default_eff: date | None = None
    if eff_match:
        default_eff = _parse_date(eff_match.group(1), eff_match.group(2), eff_match.group(3))

    # Split into sections by the index-header marker.
    headers = list(INDEX_HEADER_RE.finditer(full_text))
    events: list[dict] = []
    if not headers:
        # Fall back to a single ASX 200 section (older PDFs).
        if default_eff is None:
            return {"path": path.name, "error": "no effective date found",
                    "announcement_date": announcement}
        for line in full_text.splitlines():
            m = ACTION_RE.match(line)
            if m:
                events.append({
                    "index": "ASX200",
                    "effective_date": default_eff,
                    "action": _norm_action(m.group(1)),
                    "ticker": m.group(2).upper(),
                    "company_name": m.group(3).strip(),
                })
        return {
            "path": path.name,
            "announcement_date": announcement,
            "events": events,
        }

    for i, h in enumerate(headers):
        idx_name = _norm_index_name(h.group("idx"))
        start = h.end()
        end = headers[i + 1].start() if i + 1 < len(headers) else len(full_text)
        section_text = full_text[start:end]

        # Section-specific effective date (preferred), falling back to header.
        section_eff = default_eff
        rest = h.group("rest") or ""
        date_m = SECTION_DATE_RE.search(rest)
        if date_m:
            section_eff = _parse_date(date_m.group(1), date_m.group(2), date_m.group(3))

        if section_eff is None:
            continue

        for line in section_text.splitlines():
            m = ACTION_RE.match(line)
            if m:
                events.append({
                    "index": idx_name,
                    "effective_date": section_eff,
                    "action": _norm_action(m.group(1)),
                    "ticker": m.group(2).upper(),
                    "company_name": m.group(3).strip(),
                })

    return {
        "path": path.name,
        "announcement_date": announcement,
        "events": events,
    }


def _norm_action(raw: str) -> str:
    r = raw.strip().lower()
    if r in ("addition", "inclusion"):
        return "Addition"
    if r in ("removal", "deletion", "exclusion"):
        return "Removal"
    return raw.title()


def main() -> None:
    pdfs = sorted(PDF_DIR.glob("*-rebalance-*.pdf"))
    if not pdfs:
        # Older naming
        pdfs = sorted(PDF_DIR.glob("*-rebalance.pdf"))
    print(f"Parsing {len(pdfs)} PDFs from {PDF_DIR}...")
    rows = []
    report = []
    for p in pdfs:
        parsed = parse_pdf(p)
        if "error" in parsed:
            print(f"  SKIP {p.name}: {parsed['error']}")
            report.append({"file": p.name, "error": parsed["error"]})
            continue
        ann = parsed["announcement_date"]
        evs = parsed["events"]
        per_index = {}
        for evt in evs:
            key = evt["index"]
            per_index.setdefault(key, {"+": 0, "-": 0})
            per_index[key]["+" if evt["action"] == "Addition" else "-"] += 1
            rows.append({
                "announcement_date": pd.Timestamp(ann),
                "effective_date": pd.Timestamp(evt["effective_date"]),
                "index": evt["index"],
                "action": evt["action"],
                "ticker": evt["ticker"],
                "company_name": evt["company_name"],
            })
        breakdown = "  ".join(f"{k}=+{v['+']}/-{v['-']}" for k, v in sorted(per_index.items()))
        print(f"  OK   {p.name}  ann={ann}  {len(evs)} events  {breakdown}")
        report.append({
            "file": p.name,
            "announcement_date": ann.isoformat(),
            "n_events": len(evs),
            **{f"n_{k}": v["+"] + v["-"] for k, v in per_index.items()},
        })

    df = pd.DataFrame(rows).sort_values(
        ["announcement_date", "index", "action", "ticker"]
    )
    PROCESSED_LABELS_DIR.mkdir(parents=True, exist_ok=True)
    out_labels = PROCESSED_LABELS_DIR / "rebalance_labels.csv"
    df.to_csv(out_labels, index=False)
    pd.DataFrame(report).to_csv(OUTPUTS_DIR / "sp_pdf_parse_report.csv", index=False)

    print(f"\nTotal events: {len(df)}")
    print("By index:")
    print(df["index"].value_counts().to_string())
    print(f"\nUnique tickers: {df['ticker'].nunique()}")
    print(f"Quarters covered: {df['announcement_date'].nunique()}")
    print(f"-> {out_labels}")


if __name__ == "__main__":
    main()
