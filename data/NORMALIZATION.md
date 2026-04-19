# Data Normalization

`scripts/normalize.py` combines the Canadian Parliament and US Congress datasets
into a single file: `data/normalized/bills.csv`.

## Output schema

| Column | Type | Description |
|---|---|---|
| `bill_id` | str | Unique identifier: `ca_{BillId}` or `us_{bill_id}` |
| `source` | str | `"canada"` or `"us"` |
| `session` | str | Parliament/Congress session: `"35-1"`, `"111"`, etc. |
| `bill_number` | str | Formatted bill number: `"C-2"`, `"HB1"`, etc. |
| `title` | str | Short English title |
| `description` | str | Long English description |
| `bill_type` | str | Normalised bill type (see below) |
| `bill_type_raw` | str | Original type string from the source |
| `chamber` | str | Originating chamber: `"House"` or `"Senate"` |
| `sponsor` | str | Sponsor name (empty if not available) |
| `party` | str | Sponsor party (US only — see note) |
| `introduced_date` | str | Introduction date `YYYY-MM-DD` (empty if unknown) |
| `status` | str | Current/final status description |
| `passed` | int | **Primary label** — `1` = enacted, `0` = not enacted |
| `year` | int | Year introduced (derived from `introduced_date`) |
| **Shared derived features** | | |
| `title_word_count` | int | Number of words in `title` |
| `description_word_count` | int | Number of words in `description` |
| `month_introduced` | int | Month (1–12) the bill was introduced |
| **Canada-specific features** | | |
| `parliament_number` | int | Parliament number (35–45). Proxy for the governing era. |
| `session_number` | int | Session number within the parliament (1–3) |
| `reinstated` | int | `1` if bill was reinstated from a previous session, else `0` |
| `reached_house_second_reading` | int | `1` if the bill passed House second reading |
| `reached_house_third_reading` | int | `1` if the bill passed House third reading |
| `reached_senate_third_reading` | int | `1` if the bill passed Senate third reading |
| `days_active` | int | Days from introduction to last recorded activity |
| **US-specific features** | | |
| `num_sponsors` | int | Total sponsors in `sponsors.csv` (primary + co-sponsors) |
| `num_history_steps` | int | Number of recorded legislative actions in `history.csv` |
| `num_text_versions` | int | Number of bill-text documents in `documents.csv` (tracks amendment revisions) |
| `num_rollcalls` | int | Number of roll-call votes taken on the bill |
| `final_yea_pct` | float | `yea / (yea + nay)` for the last "On passage" roll call. `NaN` if no passage vote recorded. |
| `has_committee` | int | `1` if the bill was referred to a committee, else `0` |

> Columns specific to one source are `NaN` for the other source.

## Normalisation decisions

### `passed` label

| Source | Criterion |
|---|---|
| Canada | `ReceivedRoyalAssentDateTime` is non-null |
| US | LegiScan `status` code `== 4` ("Passed" / became Public Law) |

Both conditions identify bills that completed the full legislative process and
became law. Bills that failed, were withdrawn, or are still in progress receive `passed = 0`.

### `bill_type`

Canadian and US sources use different type vocabularies.
They are mapped to a shared set of values:

| Normalised value | Canadian source strings | US bill-number prefixes |
|---|---|---|
| `government_bill` | "Government Bill", "Senate Government Bill" | *(n/a)* |
| `private_member_bill` | "Private Member's Bill", "Private Senator's Bill", "Senate Public Bill", "Senate Private Bill" | *(n/a)* |
| `bill` | *(n/a)* | `HB`, `SB` |
| `resolution` | *(n/a)* | `HR`, `SR` |
| `joint_resolution` | *(n/a)* | `HJR`, `SJR` |
| `concurrent_resolution` | *(n/a)* | `HCR`, `SCR` |
| `constitutional_amendment` | *(n/a)* | `HCA`, `SCA` |
| `other` | anything else | *(n/a)* |

The raw source string is preserved in `bill_type_raw`.

### `chamber`

| Source | Mapping |
|---|---|
| Canada | `OriginatingChamberId == 1` → `"House"`, `== 2` → `"Senate"` |
| US | Bill-number alpha prefix starting with `H` → `"House"`, `S` → `"Senate"` |

### `introduced_date`

| Source | Derivation |
|---|---|
| Canada | `min(PassedHouseFirstReadingDateTime, PassedSenateFirstReadingDateTime)` |
| US | Date of the first `history.csv` entry (`sequence == 1`) |

### `sponsor` / `party`

| Source | Derivation |
|---|---|
| Canada | `SponsorEn` field directly. `party` is empty — `PoliticalAffiliationId` is `0` for all records in the source API response and does not encode party affiliation. |
| US | Joined from `sponsors.csv` → `people.csv`. The entry with `position == 1` (primary sponsor) is used; if absent, the first listed sponsor is used. `party` values are `D` (Democratic), `R` (Republican), `I` (Independent) as provided by LegiScan. |

### `title` / `description`

| Source | `title` | `description` |
|---|---|---|
| Canada | `ShortTitleEn` when non-empty, otherwise `LongTitleEn` | `LongTitleEn` |
| US | `title` column (LegiScan short title) | `description` column (LegiScan long description) |

## Source data summary

### Canadian Parliament (`data/canada_parl/`)

JSON files named `can-{parliament}-{session}.json`, covering Parliament 35
session 1 (1994) through Parliament 45 session 1 (2025).  Each file is a
JSON array of bill objects returned by the LEGISinfo API.

### US Congress (`data/us_cong/`)

CSV files from [LegiScan](https://legiscan.com/datasets), one directory per
Congress (111–118).  Relevant files:

| File | Contents |
|---|---|
| `bills.csv` | Bill metadata and status |
| `history.csv` | Ordered legislative actions per bill |
| `people.csv` | Legislator details including party |
| `sponsors.csv` | Bill–legislator sponsorship links |
