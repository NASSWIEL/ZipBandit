#!/bin/bash
#SBATCH --job-name=gen_raw_embeddings
#SBATCH --output=/info/raid-etu/m2/s2405959/VO2/Agent/logs_agent/gen_raw_embeddings_%j.log
#SBATCH --partition=gpu
#SBATCH --nodelist=gpue06,gpue07
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=18
#SBATCH --mem=50G
#SBATCH --time=300:00:00

set -e

# Setup environment
source $(conda info --base)/etc/profile.d/conda.sh
conda activate agent_env

echo "Starting raw embedding generation..."
python /info/raid-etu/m2/s2405959/VO2/Agent/DB/generate_raw_embeddings.py
echo "Raw embedding generation complete."
