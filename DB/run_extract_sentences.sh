#!/bin/bash
# Helper script to extract sentences from TSV file for training

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "Extracting sentences from TSV file..."
python3 "$SCRIPT_DIR/extract_sentences.py" \
  --tsv_path "/info/corpus/Blizzard2023_segmented/segmented/NEB_train/reference_24khz/NEB_test_clean.tsv" \
  --output_path "$SCRIPT_DIR/sentences.txt" \
  --num_sentences 50

echo "Done!"
