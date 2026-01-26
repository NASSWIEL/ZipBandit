#!/bin/bash
#SBATCH --job-name=gen_emb_256
#SBATCH --output=/info/raid-etu/m2/s2405959/VO2/Agent/logs_agent/generate_embeddings_256_%j.log
#SBATCH --partition=gpu
#SBATCH --mem=64G
#SBATCH --cpus-per-task=16
#SBATCH --time=200:00:00

set -e


# Activate conda environment
eval "$(conda shell.bash hook)"
conda activate agent_env

# Change to working directory
cd /info/raid-etu/m2/s2405959/VO2/Agent

# Create output directory
mkdir -p /info/corpus/Blizzard2023_segmented/segmented/NEB_train/vectors_256


# Run the generation script
python3 DB/generate_embeddings.py


# Verify the generated files
echo ""
echo "Verifying generated files:"
ls -lh /info/corpus/Blizzard2023_segmented/segmented/NEB_train/vectors_256/

echo ""
echo "Checking FAISS index dimension:"
python3 -c "
import faiss
index = faiss.read_index('/info/corpus/Blizzard2023_segmented/segmented/NEB_train/vectors_256/prompts.index')
print(f'Index dimension: {index.d}')
print(f'Number of vectors: {index.ntotal}')
print('Successfully generated 256-dim FAISS index' if index.d == 256 else 'ERROR: Wrong dimension!')
"

echo "Done!"
