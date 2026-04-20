#!/bin/bash

# Batch extract English text from Canadian bill PDFs

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Default directory
INPUT_DIR="${1:=$PROJECT_ROOT/data/canada_bill_text}"
SPLIT="${2:-0.5}"

echo "Extracting English text from PDFs in: $INPUT_DIR"
echo "Column split: $SPLIT"
echo ""

find "$INPUT_DIR" -name "*.pdf" -type f -print0 | xargs -0 -I {} python "$SCRIPT_DIR/extract_english_bill.py" {} --split "$SPLIT" -o "{}_english.txt"

echo ""
echo "Done! Check for _english.txt files in the same directories as the PDFs."
