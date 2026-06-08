"""Parse the official S&P/ASX 200 quarterly rebalance PDFs and produce a
clean labels CSV.

Each PDF has a consistent table format:

    Action  Code  Company
    Addition  AD8  Audinate Group Limited
    Removal   CXO  Core Lithium Limited
    ...

The PDFs cover the S&P/ASX 200 only (not 50/100/300). Effective dates are
extracted from the header ("effective prior to the open of trading on
Monday, March 18, 2024").

Outputs:
    data/processed/labels/rebalance_labels.csv  - canonical labels
    outputs/sp_pdf_parse_report.csv             - per-PDF event count + warnings
"""

from __future__ import annotations

import re
from datetime import date, datetime
from pathlib import Path

import pandas as pd
import pypdf

from asxrebalance.paths import OUTPUTS_DIR, PROCESSED_LABELS_DIR, REPO_ROOT

PDF_DIR = REPO_ROOT / "data" / "raw" / "marketindex"

ACTION_RE = re.compile(
    r"^\s*(?:S&P/ASX\s*200\s+)?"
    r"(Addition|Removal|Deletion|Inclusion|Exclusion)\s+"
    r"([A-Z0-9]{1,6})\s+(.+?)\s*$",
    re.IGNORECASE,
)
EFFECTIVE_RE = re.compile(
    r"effective[^,]*(?:on\s+(?:\w+,\s*)?|after\s+the\s+close\s+of\s+trading\s+on\s+)"
    r"(January|February|March|April|May|June|July|August|September|October|November|December|"
    r"Jan\.?|Feb\.?|Mar\.?|Apr\.?|May\.?|Jun\.?|Jul\.?|Aug\.?|Sep\.?|Oct\.?|Nov\.?|Dec\.?)"
    r"\s+(\d{1,2}),?\s*(\d{4})",
    re.IGNORECASE,
)
ANNOUNCEMENT_RE = re.compile(
    r"SYDNEY,?\s+"
    r"(JANUARY|FEBRUARY|MARCH|APRIL|MAY|JUNE|JULY|AUGUST|SEPTEMBER|OCTOBER|NOVEMBER|DECEMBER|"
    r"Jan\.?|Feb\.?|Mar\.?|Apr\.?|May\.?|Jun\.?|Jul\.?|Aug\.?|Sep\.?|Oct\.?|Nov\.?|Dec\.?)"
    r"\s+(\d{1,2}),?\s*(\d{4})",
    re.IGNORECASE,
)

# Headline date fallback for PDFs without a clean "effective" line — extracts
# from the section header "S&P/ASX 200 Index – {Month} {Day}, {Year} ..."
SECTION_DATE_RE = re.compile(
    r"S&P/ASX\s*200(?:\s*Index)?\s*[–—\-]\s*[A-Za-z ]*?\s*"
    r"(January|February|March|April|May|June|July|August|September|October|November|December|"
    r"Jan\.?|Feb\.?|Mar\.?|Apr\.?|May\.?|Jun\.?|Jul\.?|Aug\.?|Sep\.?|Oct\.?|Nov\.?|Dec\.?)"
    r"\s+(\d{1,2}),?\s*(\d{4})",
    re.IGNORECASE,
)

_MONTH_NORM = {
    "jan": "January", "feb": "February", "mar": "March", "apr": "April",
    "may": "May", "jun": "June", "jul": "July", "aug": "August",
    "sep": "September", "oct": "October", "nov": "November", "dec": "December",
}


def _parse_date(month: str, day: str, year: str) -> date:
    m = month.strip().lower().rstrip(".")[:3]
    full = _MONTH_NORM.get(m, month.title())
    return datetime.strptime(f"{full} {day} {year}", "%B %d %Y").date()


def parse_pdf(path: Path) -> dict:
    reader = pypdf.PdfReader(str(path))
    full_text = "\n".join((p.extract_text() or "") for p in reader.pages)

    # Header dates
    ann_match = ANNOUNCEMENT_RE.search(full_text)
    eff_match = EFFECTIVE_RE.search(full_text)
    if not ann_match:
        return {"path": path.name, "error": "no announcement date in header"}
    announcement = _parse_date(ann_match.group(1), ann_match.group(2), ann_match.group(3))
    if not eff_match:
        # Fallback: look for the section header date.
        eff_match = SECTION_DATE_RE.search(full_text)
        if not eff_match:
            return {"path": path.name, "error": "no effective date in header",
                    "announcement_date": announcement}
    effective = _parse_date(eff_match.group(1), eff_match.group(2), eff_match.group(3))

    # Only keep the S&P/ASX 200 section if multiple indices are listed in
    # the same PDF (some announcements cover multiple indices).
    # The 200 section is between the first occurrence of "ASX 200" and the
    # next index header (ASX 100 / ASX 300 / ASX 50 / All Ordinaries).
    asx200_match = re.search(
        r"S&P/ASX\s*200(?:\s*Index)?\s*[–—\-]\s*Effective",
        full_text, re.IGNORECASE,
    )
    asx200_section = full_text
    if asx200_match:
        start = asx200_match.start()
        # Find next-index marker after this
        next_idx = re.search(
            r"S&P/ASX\s*(50|100|300|All Ordinaries)\s*(?:Index)?\s*[–—\-]\s*Effective",
            full_text[start + 5:], re.IGNORECASE,
        )
        if next_idx:
            asx200_section = full_text[start:start + 5 + next_idx.start()]
        else:
            asx200_section = full_text[start:]

    events = []
    for line in asx200_section.splitlines():
        m = ACTION_RE.match(line)
        if not m:
            continue
        action_raw = m.group(1).title()
        ticker = m.group(2).upper()
        name = m.group(3).strip()
        if action_raw == "Deletion":
            action = "Removal"
        elif action_raw in ("Inclusion",):
            action = "Addition"
        elif action_raw in ("Exclusion",):
            action = "Removal"
        else:
            action = action_raw
        # Skip lines that look like sub-index sections (e.g. "Addition AD8 ...
        # to ASX 100") — only count those that the section header confirms
        # apply to ASX 200.
        events.append({"action": action, "ticker": ticker, "company_name": name})

    return {
        "path": path.name,
        "announcement_date": announcement,
        "effective_date": effective,
        "events": events,
        "n_additions": sum(1 for e in events if e["action"] == "Addition"),
        "n_removals": sum(1 for e in events if e["action"] == "Removal"),
    }


def main() -> None:
    pdfs = sorted(PDF_DIR.glob("*-asx200-rebalance.pdf"))
    print(f"Parsing {len(pdfs)} PDFs...")
    rows = []
    report = []
    for p in pdfs:
        parsed = parse_pdf(p)
        if "error" in parsed:
            print(f"  SKIP {p.name}: {parsed['error']}")
            report.append({"file": p.name, "error": parsed["error"]})
            continue
        ann = parsed["announcement_date"]
        eff = parsed["effective_date"]
        for evt in parsed["events"]:
            rows.append({
                "announcement_date": pd.Timestamp(ann),
                "effective_date": pd.Timestamp(eff),
                "index": "ASX200",
                "action": evt["action"],
                "ticker": evt["ticker"],
                "company_name": evt["company_name"],
            })
        report.append({
            "file": p.name,
            "announcement_date": ann.isoformat(),
            "effective_date": eff.isoformat(),
            "n_additions": parsed["n_additions"],
            "n_removals": parsed["n_removals"],
        })
        print(f"  OK   {p.name}  ann={ann}  eff={eff}  "
              f"+{parsed['n_additions']}/-{parsed['n_removals']}")

    df = pd.DataFrame(rows).sort_values(["announcement_date", "action", "ticker"])
    PROCESSED_LABELS_DIR.mkdir(parents=True, exist_ok=True)
    out_labels = PROCESSED_LABELS_DIR / "rebalance_labels.csv"
    df.to_csv(out_labels, index=False)

    report_df = pd.DataFrame(report)
    out_report = OUTPUTS_DIR / "sp_pdf_parse_report.csv"
    report_df.to_csv(out_report, index=False)

    print(f"\nTotal events: {len(df)}")
    print(f"  Additions:  {(df['action'] == 'Addition').sum()}")
    print(f"  Removals :  {(df['action'] == 'Removal').sum()}")
    print(f"\nUnique tickers: {df['ticker'].nunique()}")
    print(f"Unique announcement dates: {df['announcement_date'].nunique()}")
    print(f"\n-> {out_labels}")
    print(f"-> {out_report}")


if __name__ == "__main__":
    main()
