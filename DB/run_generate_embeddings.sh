#!/bin/bash
#SBATCH --job-name=gen_embeddings
#SBATCH --output=/info/raid-etu/m2/s2405959/VO2/Agent/logs_agent/gen_embeddings_%j.log
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --exclude=gpue01,gpue02,gpue03
#SBATCH --cpus-per-task=18
#SBATCH --mem=50G
#SBATCH --time=300:00:00

set -e

# Setup environment
source $(conda info --base)/etc/profile.d/conda.sh
conda activate agent_env

echo "Starting embedding generation..."
python /info/raid-etu/m2/s2405959/VO2/Agent/DB/generate_embeddings.py
echo "Embedding generation complete."
