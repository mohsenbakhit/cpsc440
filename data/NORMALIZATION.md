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
