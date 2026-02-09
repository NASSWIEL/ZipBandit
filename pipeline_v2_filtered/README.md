# Filtered Pipeline (Min 6 Words) - Isolated Training Environment

## Overview

This directory contains a **fully isolated** training pipeline that uses a **linguistically filtered** subset of embeddings. Only sentences with **at least 6 words** in both the prompt transcription and target text are included.

## Purpose

The filtered pipeline allows for a controlled comparison between:

| Pipeline | Description | Index Size |
|----------|-------------|------------|
| **Baseline** (`run_pipeline.sh`) | Unfiltered first 5,000 sentences | 5,000 |
| **Filtered** (`run_pipeline_filtered.sh`) | First 5,000 sentences with ≥6 words | 5,000 |

## Isolation Guarantees

Both pipelines are **completely isolated**:

```
Baseline Pipeline                    Filtered Pipeline
─────────────────                    ─────────────────
model/agent_model.pth          ≠     pipeline_v2_filtered/model/agent_model.pth
model/replay_buffer.pkl        ≠     pipeline_v2_filtered/model/replay_buffer.pkl
logs_agent/training_progress.csv ≠   pipeline_v2_filtered/logs_agent/training_progress.csv
checkpoint/                    ≠     pipeline_v2_filtered/checkpoint/
temp_pipeline/                 ≠     pipeline_v2_filtered/temp_pipeline/
vectors_256_subset/            ≠     vectors_256_filtered_6words/
```

**No shared state** between pipelines:
- Different model files
- Different replay buffers
- Different logs
- Different checkpoints
- Different temp files
- Different FAISS indices

## Directory Structure

```
pipeline_v2_filtered/
├── model/                    # Isolated model files
│   ├── agent_model.pth      # Trained model (created during training)
│   └── replay_buffer.pkl    # Replay buffer (created during training)
├── logs_agent/              # Isolated training logs
│   └── training_progress.csv
├── checkpoint/              # Isolated checkpoints (every 100 sentences)
├── temp_pipeline/           # Isolated temporary files
│   ├── vec_1024.npy
│   ├── vec_256.npy
│   ├── similarity_result.json
│   └── ...
├── Similarity/              # Filtered similarity search module
│   └── assess_similarity_filtered.py
└── README.md               # This file
```

## Setup Instructions

### Step 1: Generate the Filtered Index

Before running the filtered pipeline, you must generate the filtered FAISS index:

```bash
# Option A: Use the setup script (recommended)
chmod +x /info/raid-etu/m2/s2405959/VO2/Agent/setup_filtered_pipeline.sh
bash /info/raid-etu/m2/s2405959/VO2/Agent/setup_filtered_pipeline.sh

# Option B: Generate manually
conda activate agent_env
python /info/raid-etu/m2/s2405959/VO2/Agent/scripts/create_filtered_subset_index.py \
    --subset_size 5000 \
    --min_words 6 \
    --filter_field both
```

This creates the filtered index at:
```
/info/corpus/Blizzard2023_segmented/segmented/NEB_train/vectors_256_filtered_6words/
```

### Step 2: Launch the Filtered Pipeline

```bash
sbatch /info/raid-etu/m2/s2405959/VO2/Agent/run_pipeline_filtered.sh
```

## Running Both Pipelines in Parallel

To compare filtered vs unfiltered training under identical conditions:

```bash
# Terminal 1: Baseline pipeline (already running)
sbatch /info/raid-etu/m2/s2405959/VO2/Agent/run_pipeline.sh

# Terminal 2: Filtered pipeline
sbatch /info/raid-etu/m2/s2405959/VO2/Agent/run_pipeline_filtered.sh
```

Both jobs will run independently on separate GPU resources.

## Monitoring Progress

### Check job status:
```bash
squeue -u $USER
```

### View baseline logs:
```bash
tail -f /info/raid-etu/m2/s2405959/VO2/Agent/logs_agent/pipeline_*.log
```

### View filtered logs:
```bash
tail -f /info/raid-etu/m2/s2405959/VO2/Agent/pipeline_v2_filtered/logs_agent/pipeline_*.log
```

### Compare training progress:
```bash
# Baseline
cat /info/raid-etu/m2/s2405959/VO2/Agent/logs_agent/training_progress.csv | tail -20

# Filtered
cat /info/raid-etu/m2/s2405959/VO2/Agent/pipeline_v2_filtered/logs_agent/training_progress.csv | tail -20
```

## Filtering Criteria

The filter applied is:
- **Field**: Both `prompt_transcription` AND `text` (target)
- **Minimum words**: 6
- **Selection**: First 5,000 sentences that satisfy the condition

This means:
- Short utterances (< 6 words) are excluded
- Training focuses on more linguistically complex sentences
- Hypothesis: Longer sentences may provide better context for TTS prompt selection

## Expected Differences

| Metric | Baseline (unfiltered) | Filtered (6+ words) |
|--------|----------------------|---------------------|
| Avg. sentence length | Mixed | ≥ 6 words |
| Linguistic complexity | Variable | Higher |
| Training stability | TBD | TBD |
| Final CER | TBD | TBD |

## Troubleshooting

### "Filtered index not found"
Run the setup script first:
```bash
bash /info/raid-etu/m2/s2405959/VO2/Agent/setup_filtered_pipeline.sh
```

### "Model file not found"
The model is created fresh on first training. This is expected behavior for an isolated pipeline.

### Jobs interfering with each other
This should NOT happen due to complete path isolation. If you see issues:
1. Check SLURM job IDs are different
2. Verify no shared file paths in logs
3. Ensure temp directories are separate

## Files Created by This Pipeline

After training, you will find:

```
pipeline_v2_filtered/
├── model/
│   ├── agent_model.pth              # Final trained model
│   └── replay_buffer.pkl            # Final replay buffer
├── logs_agent/
│   ├── pipeline_*.log               # SLURM job logs
│   └── training_progress.csv        # Per-sentence metrics
├── checkpoint/
│   ├── agent_model_sentence_100.pth
│   ├── agent_model_sentence_200.pth
│   └── ...
└── temp_pipeline/
    └── baseline_reward.txt          # Running baseline for advantage estimation
```

## Author Notes

- **Date**: 2026-02-01
- **Purpose**: Ablation study comparing filtered vs unfiltered retrieval
- **Hypothesis**: Sentences with ≥6 words may provide better training signal
