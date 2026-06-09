"""Parse the dated S&P/ASX 200 announcement archive (data/raw/marketindex/asx200/).

Two layouts appear:

  A) PROSE single-event (removal / addition / demerger):
        "... remove Bingo Industries Limited (XASX: BIN) from the S&P/ASX 200 ..."
        "... effective prior to the open of trading on July 16, 2021."
        "... will be replaced by Centuria Capital Group (XASX: CNI) ..."   <- ADDITION
     => yields the primary event AND any replacement / spun-off addition.

  B) TABLE quarterly rebalance (same as the multi-index PDFs but ASX 200 only):
        "S&P/ASX 200 Index - Effective ... on June 22, 2026
         Action Code Company
         Addition ELV Elevra Lithium Limited
         Removal  ... "

Output: data/processed/labels/asx200_dated_events.csv
        announcement_date, effective_date, index, action, ticker, company_name,
        source_type, source_file
"""

from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path

import pandas as pd
import pypdf

from asxrebalance.paths import PROCESSED_LABELS_DIR, REPO_ROOT

PDF_DIR = REPO_ROOT / "data" / "raw" / "marketindex" / "asx200"

_MONTHS = ("January|February|March|April|May|June|July|August|September|October|"
           "November|December|Jan\\.?|Feb\\.?|Mar\\.?|Apr\\.?|May\\.?|Jun\\.?|Jul\\.?|"
           "Aug\\.?|Sep\\.?|Sept\\.?|Oct\\.?|Nov\\.?|Dec\\.?")
_MN = {"jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6, "jul": 7,
       "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12}

ANN_RE = re.compile(rf"SYDNEY,?\s+({_MONTHS})\s+(\d{{1,2}}),?\s*(\d{{4}})", re.IGNORECASE)
# Old (2011-12) format: "Sydney, Jun. 3, 2011" or a bare "October 26, 2011 -".
# Must skip the boilerplate template line "New York, January 7, 2010".
ANY_DATE_RE = re.compile(rf"({_MONTHS})\s+(\d{{1,2}}),?\s*(\d{{4}})", re.IGNORECASE)
# Section header inside multi-tier PDFs.
SECTION_RE = re.compile(r"S&P/ASX\s*(\d{2,3})\s*Index|All\s+Ordinaries|"
                        r"All\s+Australian\s*\d*|All\s+Technology|MidCap", re.IGNORECASE)
EFF_RE = re.compile(
    rf"effective\s+(?:prior to the open(?:\s+of trading)?\s+on|after the close of trading on|on)\s+"
    rf"(?:\w+,\s*)?({_MONTHS})\s+(\d{{1,2}}),?\s*(\d{{4}})", re.IGNORECASE)
# ticker like (XASX: BIN) / (ASX: FGL) / (XASX:WOW)
TICK_RE = re.compile(r"\(X?ASX:\s*([A-Z0-9]{1,4})\)")
TABLE_ROW_RE = re.compile(r"^\s*(Addition|Removal|Deletion|Inclusion)\s+([A-Z0-9]{1,4})\s+(.+?)\s*$",
                          re.IGNORECASE)


def _despace_line(line: str) -> str:
    toks = [t for t in line.split(" ") if t]
    if len(toks) >= 5 and sum(len(t) == 1 for t in toks) / len(toks) > 0.6:
        c = re.sub(r" {2,}", "\x00", line).replace(" ", "").replace("\x00", " ").strip()
        return re.sub(r"^(Addition|Removal|Deletion|Inclusion)([A-Z0-9]{1,4})\b", r"\1 \2", c)
    return line


def _text(path: Path) -> str:
    rd = pypdf.PdfReader(str(path))
    raw = "\n".join((p.extract_text() or "") for p in rd.pages)
    raw = "\n".join(_despace_line(ln) for ln in raw.split("\n"))
    return re.sub(r"[ \t]{2,}", " ", raw)


def _date(m) -> datetime.date | None:
    if not m:
        return None
    mo = _MN[m.group(1).strip(".").lower()[:3]]
    return datetime(int(m.group(3)), mo, int(m.group(2))).date()


def _ann_date(txt: str):
    m = ANN_RE.search(txt)
    if m:
        return _date(m)
    # old format: first real date that is NOT the boilerplate "New York, January 7, 2010"
    for m in ANY_DATE_RE.finditer(txt):
        d = _date(m)
        if d and not (d.year == 2010 and d.month == 1 and d.day == 7):
            return d
    return None


def _asx200_section(txt: str) -> str:
    """Return only the S&P/ASX 200 block of a multi-tier table PDF."""
    headers = [(m.start(), m.group(0)) for m in SECTION_RE.finditer(txt)]
    start = None
    for pos, h in headers:
        if re.search(r"S&P/ASX\s*200\s*Index", h, re.IGNORECASE):
            start = pos
            break
    if start is None:
        return txt  # ASX200-only doc, no other sections
    # end = next section header after start
    end = len(txt)
    for pos, h in headers:
        if pos > start and not re.search(r"S&P/ASX\s*200\s*Index", h, re.IGNORECASE):
            end = pos
            break
    return txt[start:end]


def parse_one(path: Path) -> list[dict]:
    txt = _text(path)
    ann = _ann_date(txt)
    eff = _date(EFF_RE.search(txt))
    fname = path.name
    ftype = re.search(r"-asx200-([a-z]+)", fname)
    ftype = ftype.group(1) if ftype else "other"
    events: list[dict] = []

    # ---- B) table rebalance ----
    if re.search(r"Action\s+Code\s+Company", txt, re.IGNORECASE) or ftype in ("rebalance", "quarterly"):
        # restrict to the ASX 200 section so multi-tier PDFs don't leak ASX 300 rows
        body = _asx200_section(txt)
        for ln in body.splitlines():
            m = TABLE_ROW_RE.match(ln.strip())
            if m:
                act = m.group(1).title().replace("Deletion", "Removal").replace("Inclusion", "Addition")
                events.append({"action": act, "ticker": m.group(2).upper(),
                               "company_name": m.group(3).strip()})
        if events:
            for e in events:
                e.update(announcement_date=ann, effective_date=eff,
                         source_type=ftype, source_file=fname)
            return _finalize(events)

    # ---- A) prose single-event ----
    low = txt.lower()
    ticks = TICK_RE.findall(txt)
    primary_action = None
    if "to be removed" in low or re.search(r"\bremove\b", low):
        primary_action = "Removal"
    elif "to be added" in low or re.search(r"\badd(ed)?\b", low):
        primary_action = "Addition"
    elif ftype == "demerger" or "spin-off" in low or "demerger" in low:
        primary_action = "Demerger"

    # the FIRST ticker is the subject company
    if ticks:
        subj = ticks[0]
        if primary_action == "Removal":
            events.append({"action": "Removal", "ticker": subj, "company_name": "",
                           "source_type": ftype, "source_file": fname,
                           "announcement_date": ann, "effective_date": eff})
            # replacement addition (second ticker after "replaced by")
            rep = re.search(r"replaced by[^()]*\(X?ASX:\s*([A-Z0-9]{1,4})\)", txt, re.IGNORECASE)
            if rep:
                events.append({"action": "Addition", "ticker": rep.group(1).upper(),
                               "company_name": "", "source_type": ftype + "-replacement",
                               "source_file": fname, "announcement_date": ann, "effective_date": eff})
        elif primary_action == "Addition":
            events.append({"action": "Addition", "ticker": subj, "company_name": "",
                           "source_type": ftype, "source_file": fname,
                           "announcement_date": ann, "effective_date": eff})
        elif primary_action == "Demerger":
            # the spun-off entity (usually the 2nd ticker) is ADDED
            add_tk = ticks[1] if len(ticks) > 1 else subj
            events.append({"action": "Addition", "ticker": add_tk.upper(), "company_name": "",
                           "source_type": "demerger", "source_file": fname,
                           "announcement_date": ann, "effective_date": eff})
    return _finalize(events)


def _finalize(events: list[dict]) -> list[dict]:
    out = []
    for e in events:
        if not e.get("ticker") or not e.get("announcement_date"):
            continue
        e["index"] = "ASX200"
        out.append(e)
    return out


def main() -> None:
    rows, failed = [], []
    for p in sorted(PDF_DIR.glob("*.pdf")):
        try:
            evs = parse_one(p)
        except Exception as exc:  # noqa: BLE001
            failed.append((p.name, str(exc)[:60]))
            continue
        if evs:
            rows.extend(evs)
        else:
            failed.append((p.name, "no events extracted"))

    df = pd.DataFrame(rows)
    df = df[["announcement_date", "effective_date", "index", "action", "ticker",
             "company_name", "source_type", "source_file"]]
    df = df.drop_duplicates(subset=["announcement_date", "action", "ticker"]).reset_index(drop=True)
    PROCESSED_LABELS_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(PROCESSED_LABELS_DIR / "asx200_dated_events.csv", index=False)

    print(f"Parsed {len(df)} ASX 200 events from {len(list(PDF_DIR.glob('*.pdf')))} dated PDFs")
    print("by action:", df["action"].value_counts().to_dict())
    print("by source_type:", df["source_type"].value_counts().to_dict())
    print(f"\nPDFs with no events ({len(failed)}):")
    for n, why in failed[:30]:
        print(f"  {n}: {why}")


if __name__ == "__main__":
    main()
