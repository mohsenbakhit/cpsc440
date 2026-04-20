#!/bin/bash

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
INPUT_DIR="${1:-$PROJECT_ROOT/data/canada_bill_text}"

echo "Cleaning txt files in: $INPUT_DIR"
echo ""

find "$INPUT_DIR" -name "*.txt" -type f -print0 | while read -r -d '' txt_file; do
  temp_file="${txt_file}.tmp"
  python3 "$SCRIPT_DIR/clean_canadian_bill.py" --input "$txt_file" --output "$temp_file"
  mv "$temp_file" "$txt_file"
  echo "✓ Cleaned ${txt_file#$INPUT_DIR/}"
done

echo ""
echo "Done! All txt files have been cleaned."
