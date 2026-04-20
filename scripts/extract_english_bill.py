"""
extract_english_bill.py
-----------------------
Extracts English-only text from bilingual Canadian Parliament PDFs.

The PDFs use a two-column layout: English on the left, French on the right.
Strategy: crop each page to the left half, then extract text.

Usage:
    python extract_english_bill.py C-2.pdf
    python extract_english_bill.py C-2.pdf --output C-2_english.txt
    python extract_english_bill.py data/canada_bill_text/  # batch process all PDFs
"""

import re
import sys
import argparse
from pathlib import Path
import pdfplumber

# These tokens appear in the left margin as section annotations.
# They're structural noise, not bill text.
MARGIN_NOISE = re.compile(
    r"^\s*(R\.S\.,|L\.R\.,|c\.\s|ch\.\s|s\.\s|art\.\s|\d{4},)\s*$",
    re.MULTILINE
)

# Short standalone labels that appear in the left gutter (e.g. "Deputy head", "References")
# We keep these as they may carry signal, but you can strip them if desired.


def extract_english(pdf_path: str, col_split: float = 0.5) -> str:
    """
    Extract English text from a bilingual Canadian bill PDF.

    Args:
        pdf_path:   Path to the PDF file.
        col_split:  Fraction of page width used as the English/French boundary.
                    Default 0.5 (left half). Adjust if layout differs.

    Returns:
        Extracted English text as a single string.
    """
    pages_text = []

    with pdfplumber.open(pdf_path) as pdf:
        for i, page in enumerate(pdf.pages):
            split_x = page.width * col_split
            left_col = page.crop((0, 0, split_x, page.height))
            text = left_col.extract_text()

            if text:
                pages_text.append(text)

    full_text = "\n\n".join(pages_text)

    # Remove citation margin noise (e.g. "R.S., c. N-16;" on its own line)
    full_text = MARGIN_NOISE.sub("", full_text)

    # Collapse multiple blank lines
    full_text = re.sub(r"\n{3,}", "\n\n", full_text)

    return full_text.strip()


def main():
    parser = argparse.ArgumentParser(description="Extract English text from bilingual Canadian bill PDFs.")
    parser.add_argument("path", help="Path to a PDF file or directory of PDFs")
    parser.add_argument("--output", "-o", help="Output .txt file (single file mode only)")
    parser.add_argument("--split", type=float, default=0.5,
                        help="Column split fraction (default: 0.5). Increase slightly if English text is cut off.")
    args = parser.parse_args()

    path = Path(args.path)

    if path.is_file():
        # Single file mode
        english_text = extract_english(str(path), col_split=args.split)
        if args.output:
            with open(args.output, "w", encoding="utf-8") as f:
                f.write(english_text)
            print(f"Saved to {args.output}")
        else:
            print(english_text)
    elif path.is_dir():
        # Batch mode
        pdf_files = sorted(path.rglob("*.pdf"))
        if not pdf_files:
            print(f"No PDF files found in {path}")
            return

        print(f"Found {len(pdf_files)} PDF(s). Processing...")
        for pdf_path in pdf_files:
            try:
                english_text = extract_english(str(pdf_path), col_split=args.split)
                output_path = pdf_path.with_stem(pdf_path.stem + "_english")
                output_path = output_path.with_suffix(".txt")
                with open(output_path, "w", encoding="utf-8") as f:
                    f.write(english_text)
                print(f"✓ {pdf_path.relative_to(path)} → {output_path.name}")
            except Exception as e:
                print(f"✗ {pdf_path.relative_to(path)}: {e}")
    else:
        print(f"Error: {path} is not a valid file or directory")
        sys.exit(1)


if __name__ == "__main__":
    main()