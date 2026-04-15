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
from pathlib import Path

import pandas as pd

# ── Paths ─────────────────────────────────────────────────────────────────────

ROOT_DIR  = Path(__file__).resolve().parent.parent
DATA_DIR  = ROOT_DIR / "data"
CANADA_DIR = DATA_DIR / "canada_parl"
US_DIR    = DATA_DIR / "us_cong"
OUT_DIR   = DATA_DIR / "normalized"

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

            rows.append({
                "bill_id":       f"ca_{b['BillId']}",
                "source":        "canada",
                "session":       session,
                "bill_number":   b.get("BillNumberFormatted", ""),
                # Prefer short title; fall back to long title
                "title":         short_title or long_title,
                "description":   long_title,
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
            })

    return pd.DataFrame(rows)


# ── US helpers ─────────────────────────────────────────────────────────────────

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

        bills    = pd.read_csv(csv_dir / "bills.csv",    dtype=str)
        people   = pd.read_csv(csv_dir / "people.csv",   dtype=str)
        sponsors = pd.read_csv(csv_dir / "sponsors.csv", dtype=str)
        history  = pd.read_csv(csv_dir / "history.csv",  dtype=str)

        # ── Introduced date: first history action per bill ────────────────
        history["sequence"] = pd.to_numeric(history["sequence"], errors="coerce")
        intro_dates = (
            history.sort_values("sequence")
            .groupby("bill_id")["date"]
            .first()
            .reset_index()
            .rename(columns={"date": "introduced_date"})
        )

        # ── Primary sponsor: prefer position == "1", else any ─────────────
        # Merge sponsors with people to bring in name and party
        sponsor_info = sponsors.merge(
            people[["people_id", "name", "party"]],
            on="people_id",
            how="left",
        )
        # Primary sponsors first, then pick one per bill
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

        # ── Merge supplemental tables into bills ──────────────────────────
        bills = (
            bills
            .merge(intro_dates, on="bill_id", how="left")
            .merge(primary,     on="bill_id", how="left")
        )

        # ── Build rows ────────────────────────────────────────────────────
        for _, row in bills.iterrows():
            bn = str(row.get("bill_number") or "")
            chamber, bill_type = _us_parse_bill_number(bn)
            introduced = str(row.get("introduced_date") or "")
            if introduced in ("", "nan", "NaT", "None"):
                introduced = ""
            year = int(introduced[:4]) if len(introduced) >= 4 else None

            try:
                passed = 1 if int(row.get("status", 0)) == _US_PASSED_STATUS else 0
            except (ValueError, TypeError):
                passed = 0

            rows.append({
                "bill_id":         f"us_{row['bill_id']}",
                "source":          "us",
                "session":         session,
                "bill_number":     bn,
                "title":           str(row.get("title")       or ""),
                "description":     str(row.get("description") or ""),
                "bill_type":       bill_type,
                "bill_type_raw":   bn.rstrip("0123456789 "),   # alpha prefix only
                "chamber":         chamber,
                "sponsor":         str(row.get("sponsor")       or ""),
                "party":           str(row.get("sponsor_party") or ""),
                "introduced_date": introduced,
                "status":          str(row.get("status_desc")  or ""),
                "passed":          passed,
                "year":            year,
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
