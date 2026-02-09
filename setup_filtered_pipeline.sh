#!/bin/bash
# ============================================================================
# Setup script for FILTERED pipeline (min 6 words)
# ============================================================================
# This script generates the filtered FAISS index before training.
# Run this ONCE before launching run_pipeline_filtered.sh
# ============================================================================

set -e

BASE_DIR="/info/raid-etu/m2/s2405959/VO2/Agent"
SCRIPT_DIR="$BASE_DIR/scripts"
OUTPUT_DIR="/info/corpus/Blizzard2023_segmented/segmented/NEB_train/vectors_256_filtered_6words"

echo "============================================================================"
echo "Setting up FILTERED pipeline (min 6 words, 5000 sentences)"
echo "============================================================================"

# Check if filtered index already exists
if [ -f "$OUTPUT_DIR/prompts.index" ]; then
    echo ""
    echo "Filtered index already exists at: $OUTPUT_DIR"
    echo "To regenerate, delete the directory first:"
    echo "  rm -rf $OUTPUT_DIR"
    echo ""
    echo "Skipping index generation..."
else
    echo ""
    echo "Generating filtered embeddings index..."
    echo "  - Minimum words: 6"
    echo "  - Filter field: both (prompt AND text)"
    echo "  - Subset size: 5000"
    echo ""
    
    conda run -n agent_env python "$SCRIPT_DIR/create_filtered_subset_index.py" \
        --subset_size 5000 \
        --min_words 6 \
        --filter_field both \
        --output_dir "$OUTPUT_DIR"
    
    echo ""
    echo "Filtered index created successfully!"
fi

# Verify the setup
echo ""
echo "============================================================================"
echo "Verifying setup..."
echo "============================================================================"

# Check filtered index
if [ -f "$OUTPUT_DIR/prompts.index" ]; then
    echo "[OK] Filtered FAISS index exists"
else
    echo "[ERROR] Filtered FAISS index not found!"
    exit 1
fi

# Check isolated directories
PIPELINE_BASE="$BASE_DIR/pipeline_v2_filtered"
for dir in "model" "logs_agent" "checkpoint" "temp_pipeline" "Similarity"; do
    if [ -d "$PIPELINE_BASE/$dir" ]; then
        echo "[OK] Directory exists: $PIPELINE_BASE/$dir"
    else
        echo "Creating directory: $PIPELINE_BASE/$dir"
        mkdir -p "$PIPELINE_BASE/$dir"
    fi
done

# Check similarity script
if [ -f "$PIPELINE_BASE/Similarity/assess_similarity_filtered.py" ]; then
    echo "[OK] Filtered similarity script exists"
else
    echo "[ERROR] Filtered similarity script not found!"
    exit 1
fi

echo ""
echo "============================================================================"
echo "Setup complete! You can now launch the filtered pipeline:"
echo ""
echo "  sbatch $BASE_DIR/run_pipeline_filtered.sh"
echo ""
echo "============================================================================"
