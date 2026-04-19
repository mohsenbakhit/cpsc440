#!/usr/bin/env python3
"""
normalize.py — Unifies Canadian Parliament and US Congress bill data
into a single CSV suitable for ML training.

Output
------
    data/normalized/bills.csv

Run
---
    python scripts/normalize.py

See data/NORMALIZATION.md for schema documentation and design decisions.
"""

import json
import re
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

# ── Paths ─────────────────────────────────────────────────────────────────────

ROOT_DIR  = Path(__file__).resolve().parent.parent
DATA_DIR  = ROOT_DIR / "data"
CANADA_DIR = DATA_DIR / "canada_parl"
US_DIR    = DATA_DIR / "us_cong"
OUT_DIR   = DATA_DIR / "normalized"

def _word_count(text: str | None) -> int:
    """Return number of whitespace-separated tokens in text."""
    if not text:
        return 0
    return len(text.split())


def _parse_iso_date(dt_str: str | None) -> datetime | None:
    """Parse a timezone-aware ISO datetime string; return None on failure."""
    if not dt_str:
        return None
    try:
        # Python 3.11+: datetime.fromisoformat handles offset-aware strings.
        # For 3.9/3.10 compatibility we strip the offset and treat as naive.
        return datetime.fromisoformat(dt_str)
    except ValueError:
        return None


def _days_between(start: str | None, end: str | None) -> int | None:
    """Return integer day count between two ISO datetime strings, or None."""
    d1 = _parse_iso_date(start)
    d2 = _parse_iso_date(end)
    if d1 is None or d2 is None:
        return None
    # Make both offset-naive for subtraction
    if d1.tzinfo is not None:
        d1 = d1.replace(tzinfo=None)
    if d2.tzinfo is not None:
        d2 = d2.replace(tzinfo=None)
    delta = (d2 - d1).days
    return delta if delta >= 0 else None


# ── Canada helpers ─────────────────────────────────────────────────────────────

# Map raw Canadian bill-type strings → normalised category.
# "Government Bill" / "Senate Government Bill"  → government_bill
# All private / public member bills              → private_member_bill
_CA_BILL_TYPE_MAP: dict[str, str] = {
    "Government Bill":            "government_bill",
    "Senate Government Bill":     "government_bill",
    "Private Member's Bill":      "private_member_bill",
    "Private Senator's Bill":     "private_member_bill",
    "Senate Public Bill":         "private_member_bill",
    "Senate Private Bill":        "private_member_bill",
}


def _ca_chamber(originating_chamber_id: int) -> str:
    """OriginatingChamberId 1 = House of Commons, 2 = Senate."""
    return "House" if originating_chamber_id == 1 else "Senate"


def _ca_introduced_date(record: dict) -> str | None:
    """
    Earliest first-reading date from either chamber.
    Returns ISO date string (YYYY-MM-DD) or None.
    """
    candidates = [
        record.get("PassedHouseFirstReadingDateTime"),
        record.get("PassedSenateFirstReadingDateTime"),
    ]
    dates = [c[:10] for c in candidates if c]
    return min(dates) if dates else None


def normalize_canada() -> pd.DataFrame:
    """
    Load every can-*.json file and return a normalised DataFrame.

    Notes
    -----
    - PoliticalAffiliationId is 0 for every record in the source data
      (the Parliament API does not expose it via this endpoint), so `party`
      is left empty for Canadian bills.
    - `passed` = 1 when ReceivedRoyalAssentDateTime is non-null (Royal Assent
      is the Canadian equivalent of enactment).
    """
    rows: list[dict] = []

    for json_file in sorted(CANADA_DIR.glob("can-*.json")):
        # Derive session code from filename, e.g. "can-35-1.json" → "35-1"
        session = json_file.stem[4:]

        with open(json_file, encoding="utf-8") as fh:
            bills = json.load(fh)

        for b in bills:
            short_title = (b.get("ShortTitleEn") or "").strip()
            long_title  = (b.get("LongTitleEn")  or "").strip()
            bill_type_raw = (b.get("BillTypeEn") or "").strip()
            introduced = _ca_introduced_date(b)
            year = int(introduced[:4]) if introduced else None

            title = short_title or long_title
            description = long_title

            # ── Reading milestones ────────────────────────────────────────
            h2 = b.get("PassedHouseSecondReadingDateTime")
            h3 = b.get("PassedHouseThirdReadingDateTime")
            s3 = b.get("PassedSenateThirdReadingDateTime")
            latest = b.get("LatestActivityDateTime")

            rows.append({
                # ── Core fields ───────────────────────────────────────────
                "bill_id":       f"ca_{b['BillId']}",
                "source":        "canada",
                "session":       session,
                "bill_number":   b.get("BillNumberFormatted", ""),
                "title":         title,
                "description":   description,
                "bill_type":     _CA_BILL_TYPE_MAP.get(bill_type_raw, "other"),
                "bill_type_raw": bill_type_raw,
                "chamber":       _ca_chamber(b.get("OriginatingChamberId", 1)),
                "sponsor":       (b.get("SponsorEn") or "").strip(),
                # PoliticalAffiliationId not populated in source data
                "party":         "",
                "introduced_date": introduced or "",
                "status":        (b.get("CurrentStatusEn") or "").strip(),
                # Royal Assent = enacted into law
                "passed":        1 if b.get("ReceivedRoyalAssentDateTime") else 0,
                "year":          year,
                # ── Derived numeric / boolean features ───────────────────
                "title_word_count":       _word_count(title),
                "description_word_count": _word_count(description),
                "month_introduced":       int(introduced[5:7]) if introduced else None,
                # Canada-specific structural features
                "parliament_number":      b.get("ParliamentNumber"),
                "session_number":         b.get("SessionNumber"),
                "reinstated":             1 if b.get("DidReinstateFromPreviousSession") else 0,
                # How far the bill progressed through readings (0/1 flags)
                "reached_house_second_reading": 1 if h2 else 0,
                "reached_house_third_reading":  1 if h3 else 0,
                "reached_senate_third_reading": 1 if s3 else 0,
                # Total days from first introduction to last recorded activity
                "days_active": _days_between(introduced, latest),
                # US-only fields (empty for Canada)
                "num_sponsors":    None,
                "num_history_steps": None,
                "num_text_versions": None,
                "num_rollcalls":   None,
                "final_yea_pct":   None,
                "has_committee":   None,
            })

    return pd.DataFrame(rows)


# ── US helpers ─────────────────────────────────────────────────────────────────

def _to_int(value) -> int | None:
    """Convert a pandas scalar to int, or None if missing/NaN."""
    try:
        v = float(value)
        return None if (v != v) else int(v)   # NaN check: NaN != NaN
    except (TypeError, ValueError):
        return None


def _to_float(value) -> float | None:
    """Convert a pandas scalar to float, or None if missing/NaN."""
    try:
        v = float(value)
        return None if (v != v) else v
    except (TypeError, ValueError):
        return None


# LegiScan status code 4 = "Passed" (enacted into law / became public law).
_US_PASSED_STATUS = 4

# Bill-number prefixes → (chamber, bill_type)
_US_PREFIX_MAP: dict[str, tuple[str, str]] = {
    "HB":  ("House",  "bill"),
    "SB":  ("Senate", "bill"),
    "HR":  ("House",  "resolution"),
    "SR":  ("Senate", "resolution"),
    "HJR": ("House",  "joint_resolution"),
    "SJR": ("Senate", "joint_resolution"),
    "HCR": ("House",  "concurrent_resolution"),
    "SCR": ("Senate", "concurrent_resolution"),
    "HCA": ("House",  "constitutional_amendment"),
    "SCA": ("Senate", "constitutional_amendment"),
}
_PREFIX_RE = re.compile(r"^([A-Za-z]+)")


def _us_parse_bill_number(bill_number: str) -> tuple[str, str]:
    """Return (chamber, bill_type) from a LegiScan bill number like 'HB1'."""
    m = _PREFIX_RE.match(bill_number or "")
    prefix = m.group(1).upper() if m else ""
    return _US_PREFIX_MAP.get(prefix, ("Unknown", "bill"))


def normalize_us() -> pd.DataFrame:
    """
    Load all {N}_congress/ directories and return a normalised DataFrame.

    Notes
    -----
    - `passed` = 1 when LegiScan status code == 4 (enacted / became law).
    - Chamber is derived from the bill-number prefix (H* = House, S* = Senate).
    - Primary sponsor is the person with position == 1 in sponsors.csv.
      If no position-1 entry exists the first listed sponsor is used.
    - `introduced_date` is the date of the first history entry (sequence 1).
    - Party abbreviations (D / R / I) are kept as-is from people.csv.
    - `final_yea_pct` is yea / (yea + nay) for the roll call whose description
      contains "On passage" (case-insensitive). NaN when no such vote exists.
    - `num_sponsors` is the total count of entries in sponsors.csv per bill
      (includes primary + co-sponsors).
    - `has_committee` is 1 when the bill was referred to a committee.
    """
    rows: list[dict] = []

    for session_dir in sorted(US_DIR.iterdir()):
        if not session_dir.is_dir():
            continue
        csv_dir = session_dir / "csv"
        if not csv_dir.exists():
            continue

        # "111_congress" → "111"
        session = session_dir.name.split("_")[0]

        bills     = pd.read_csv(csv_dir / "bills.csv",    dtype=str)
        people    = pd.read_csv(csv_dir / "people.csv",   dtype=str)
        sponsors  = pd.read_csv(csv_dir / "sponsors.csv", dtype=str)
        history   = pd.read_csv(csv_dir / "history.csv",  dtype=str)
        rollcalls = pd.read_csv(csv_dir / "rollcalls.csv", dtype=str)
        documents = pd.read_csv(csv_dir / "documents.csv", dtype=str)

        # ── Introduced date: first history action per bill ────────────────
        history["sequence"] = pd.to_numeric(history["sequence"], errors="coerce")
        intro_dates = (
            history.sort_values("sequence")
            .groupby("bill_id")["date"]
            .first()
            .reset_index()
            .rename(columns={"date": "introduced_date"})
        )

        # ── Number of history steps per bill ──────────────────────────────
        history_counts = (
            history.groupby("bill_id")["sequence"]
            .count()
            .reset_index()
            .rename(columns={"sequence": "num_history_steps"})
        )

        # ── Total sponsor count per bill ──────────────────────────────────
        sponsor_counts = (
            sponsors.groupby("bill_id")["people_id"]
            .count()
            .reset_index()
            .rename(columns={"people_id": "num_sponsors"})
        )

        # ── Primary sponsor: prefer position == "1", else any ─────────────
        sponsor_info = sponsors.merge(
            people[["people_id", "name", "party"]],
            on="people_id",
            how="left",
        )
        sponsor_info["position"] = pd.to_numeric(
            sponsor_info["position"], errors="coerce"
        ).fillna(0)
        primary = (
            sponsor_info.sort_values("position", ascending=False)
            .groupby("bill_id")
            .first()
            .reset_index()[["bill_id", "name", "party"]]
            .rename(columns={"name": "sponsor", "party": "sponsor_party"})
        )

        # ── Number of roll calls per bill ─────────────────────────────────
        rollcall_counts = (
            rollcalls.groupby("bill_id")["roll_call_id"]
            .count()
            .reset_index()
            .rename(columns={"roll_call_id": "num_rollcalls"})
        )

        # ── Final-passage yea percentage ──────────────────────────────────
        # Find roll calls whose description matches "On passage" (exact LegiScan label).
        passage_mask = rollcalls["description"].str.contains(
            r"\bon passage\b", case=False, na=False, regex=True
        )
        passage_votes = rollcalls[passage_mask].copy()
        passage_votes["yea"] = pd.to_numeric(passage_votes["yea"], errors="coerce")
        passage_votes["nay"] = pd.to_numeric(passage_votes["nay"], errors="coerce")
        passage_votes["final_yea_pct"] = (
            passage_votes["yea"] / (passage_votes["yea"] + passage_votes["nay"])
        ).round(4)
        # Keep the last passage vote per bill (conference/bicameral bills may
        # have more than one)
        final_yea = (
            passage_votes.sort_values("date")
            .groupby("bill_id")["final_yea_pct"]
            .last()
            .reset_index()
        )

        # ── Number of text versions (document revisions) ──────────────────
        doc_counts = (
            documents.groupby("bill_id")["document_id"]
            .count()
            .reset_index()
            .rename(columns={"document_id": "num_text_versions"})
        )

        # ── Merge all supplemental tables into bills ──────────────────────
        bills = (
            bills
            .merge(intro_dates,     on="bill_id", how="left")
            .merge(history_counts,  on="bill_id", how="left")
            .merge(sponsor_counts,  on="bill_id", how="left")
            .merge(primary,         on="bill_id", how="left")
            .merge(rollcall_counts, on="bill_id", how="left")
            .merge(final_yea,       on="bill_id", how="left")
            .merge(doc_counts,      on="bill_id", how="left")
        )

        # ── Build rows ────────────────────────────────────────────────────
        for _, row in bills.iterrows():
            bn = str(row.get("bill_number") or "")
            chamber, bill_type = _us_parse_bill_number(bn)
            introduced = str(row.get("introduced_date") or "")
            if introduced in ("", "nan", "NaT", "None"):
                introduced = ""
            last_action = str(row.get("last_action_date") or "")
            if last_action in ("", "nan", "NaT", "None"):
                last_action = ""
            year = int(introduced[:4]) if len(introduced) >= 4 else None

            try:
                passed = 1 if int(row.get("status", 0)) == _US_PASSED_STATUS else 0
            except (ValueError, TypeError):
                passed = 0

            title       = str(row.get("title")       or "")
            description = str(row.get("description") or "")

            # days_active: introduction date → last recorded action date
            days_active = _days_between(
                introduced if introduced else None,
                last_action if last_action else None,
            )

            rows.append({
                # ── Core fields ───────────────────────────────────────────
                "bill_id":         f"us_{row['bill_id']}",
                "source":          "us",
                "session":         session,
                "bill_number":     bn,
                "title":           title,
                "description":     description,
                "bill_type":       bill_type,
                "bill_type_raw":   bn.rstrip("0123456789 "),
                "chamber":         chamber,
                "sponsor":         str(row.get("sponsor")       or ""),
                "party":           str(row.get("sponsor_party") or ""),
                "introduced_date": introduced,
                "status":          str(row.get("status_desc")  or ""),
                "passed":          passed,
                "year":            year,
                # ── Derived numeric / boolean features ───────────────────
                "title_word_count":       _word_count(title),
                "description_word_count": _word_count(description),
                "month_introduced":       int(introduced[5:7]) if len(introduced) >= 7 else None,
                # Canada-only fields (empty for US)
                "parliament_number":      None,
                "session_number":         None,
                "reinstated":             None,
                "reached_house_second_reading": None,
                "reached_house_third_reading":  None,
                "reached_senate_third_reading": None,
                # US-specific features
                "days_active":     days_active,
                "num_sponsors":    _to_int(row.get("num_sponsors")),
                "num_history_steps": _to_int(row.get("num_history_steps")),
                "num_text_versions": _to_int(row.get("num_text_versions")),
                "num_rollcalls":   _to_int(row.get("num_rollcalls")),
                "final_yea_pct":   _to_float(row.get("final_yea_pct")),
                "has_committee":   0 if str(row.get("committee_id", "0")) in ("0", "", "nan") else 1,
            })

    return pd.DataFrame(rows)


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    print("Normalizing Canadian Parliament data …")
    ca = normalize_canada()
    print(f"  {len(ca):,} bills from {ca['session'].nunique()} sessions")

    print("Normalizing US Congress data …")
    us = normalize_us()
    print(f"  {len(us):,} bills from {us['session'].nunique()} congresses")

    combined = pd.concat([ca, us], ignore_index=True)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / "bills.csv"
    combined.to_csv(out_path, index=False)

    print(f"\nSaved {len(combined):,} bills → {out_path}")
    print("\nColumn summary:")
    print(combined.dtypes.to_string())
    print("\nPassed-rate by source:")
    print(combined.groupby("source")["passed"].mean().round(3).to_string())
    print("\nBill counts by source and type:")
    print(combined.groupby(["source", "bill_type"])["bill_id"].count().to_string())


if __name__ == "__main__":
    main()
