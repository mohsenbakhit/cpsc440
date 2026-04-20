"""
Post-extraction cleaner for Canadian bill text.
Assumes you've already extracted English text from the PDF (e.g., via pdfplumber).
"""

import re


# ── Regex patterns ─────────────────────────────────────────────────────────────

# Truncated running headers from the PDF renderer: "(cid:N) C. 13 Department of Na"
CID_HEADER_RE = re.compile(r'\(cid:\d+\)\s*C\.\s*\d+[^\n]*', re.IGNORECASE)

# French running headers: "1994 Ministère du Re"
FRENCH_HEADER_RE = re.compile(r'\d{4}\s+Minist[èe]re\s+du\s+\w+', re.IGNORECASE)

# Statute citation sidenotes: "R.S., c. N-16;", "1992, c. 1,", "s. 105", etc.
CITATION_RE = re.compile(
    r'(?:R\.S\.C?\.,?\s*c\.\s*[\w\-]+[;,]?'
    r'|S\.C\.\s*\d{4},\s*c\.\s*\d+'
    r'|\d{4}(?:-\d{2,4})*,\s*c\.\s*\d+[,;]?'
    r'|\bs\.\s*\d+(?:\(\d+\))?)',
    re.IGNORECASE
)

# Marginal notes: short capitalized phrases used as side-labels
# Covers multi-line broken marginal notes too (e.g. "Appropria-\ntions")
MARGINAL_NOTE_RE = re.compile(
    r'(?:Deputy\s+head|Idem(?:\s+re)?|Positions?|Appropriations?|Appropria-\s*\n?\s*tions?'
    r'|Powers?,?\s*duties?\s*\n?\s*or\s*\n?\s*functions?|References?(?:\s+to)?|Definition\s+of'
    r'|New\s+Terminology|Deputy\s*\n?\s*Minister|''employee''|temporary\s+or\s*\n?\s*acting\s+officers?)',
    re.IGNORECASE
)

# Hyphenated line-break rejoining: "em-\nployee" → "employee"
HYPHEN_BREAK_RE = re.compile(r'(\w+)-\s*\n\s*(\w+)')

# French sentence slip-through (contains accented French words)
FRENCH_LINE_RE = re.compile(
    r'^[^\n]*\b(?:du|au|de|le|la|les|et|ou|ministère|Revenu|national|impôt|accise)\b[^\n]*$',
    re.IGNORECASE | re.MULTILINE
)

# Boilerplate header block (everything up to and including the enacting formula)
ENACTING_FORMULA_RE = re.compile(
    r'^.*?Her Majesty.*?enacts as follows\s*:',
    re.DOTALL | re.IGNORECASE
)

# Multiple whitespace / blank lines
WHITESPACE_RE = re.compile(r'\n{3,}')
INLINE_SPACE_RE = re.compile(r'[ \t]{2,}')


def clean_canadian_bill(raw_text: str, strip_header: bool = True) -> str:
    """
    Clean extracted English text from a Canadian bilingual bill PDF.

    Steps:
        1. Rejoin hyphenated word splits across lines
        2. Remove CID/truncated running headers
        3. Remove French running headers
        4. Remove statute citation sidenotes
        5. Remove marginal notes
        6. Remove French sentence slip-through
        7. Optionally strip the boilerplate header block
        8. Normalize whitespace
        9. Lowercase

    Args:
        raw_text:      Raw extracted English text from the PDF
        strip_header:  If True, remove everything before "enacts as follows"

    Returns:
        Cleaned, lowercased text string.
    """
    text = raw_text

    # 1. Rejoin hyphenated line breaks: "em-\nployee" → "employee"
    text = HYPHEN_BREAK_RE.sub(r'\1\2', text)

    # 2. Remove CID/truncated running headers
    text = CID_HEADER_RE.sub('', text)

    # 3. Remove French running headers
    text = FRENCH_HEADER_RE.sub('', text)

    # 4. Remove statute citation sidenotes (inline, line-start)
    text = CITATION_RE.sub('', text)

    # 5. Remove marginal notes
    text = MARGINAL_NOTE_RE.sub('', text)

    # 6. Remove French sentence lines
    text = FRENCH_LINE_RE.sub('', text)

    # 7. Strip boilerplate header block (session info → enacting formula)
    if strip_header:
        match = ENACTING_FORMULA_RE.search(text)
        if match:
            text = text[match.end():]

    # 8. Normalize whitespace
    text = WHITESPACE_RE.sub('\n\n', text)
    text = INLINE_SPACE_RE.sub(' ', text)
    text = text.strip()

    # 9. Lowercase
    text = text.lower()

    return text

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to extracted text file")
    parser.add_argument("--output", required=True, help="Path to save cleaned text")
    args = parser.parse_args()

    raw = open(args.input, "r", encoding="utf-8").read()
    cleaned = clean_canadian_bill(raw)
    open(args.output, "w", encoding="utf-8").write(cleaned)
    print(f"Done. Saved to {args.output}")
