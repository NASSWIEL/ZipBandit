#!/bin/bash
#SBATCH --job-name=demo_prompt_comparison
#SBATCH --output=/info/raid-etu/m2/s2405959/VO2/Agent/logs_agent/demo_%j.log
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=40G
#SBATCH --cpus-per-task=8
#SBATCH --time=02:00:00

# Demo Script: Prompt Selection Impact on CER
# This script demonstrates how different prompt selections affect CER for the same target text

set -e

BASE_DIR="/info/raid-etu/m2/s2405959/VO2/Agent"

# Activate conda environment
source $(conda info --base)/etc/profile.d/conda.sh
conda activate agent_env

echo "=========================================="
echo "Demo: Prompt Selection Impact on CER"
echo "=========================================="
echo ""

# Run the demo with default settings (uses preset text and 5 random prompts)
python3 "$BASE_DIR/demo_prompt_comparison.py" "$@"

echo ""
echo "=========================================="
echo "Demo completed!"
echo "=========================================="
