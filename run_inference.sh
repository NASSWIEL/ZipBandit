#!/bin/bash
#SBATCH --job-name=agent_inference
#SBATCH --output=/info/raid-etu/m2/s2405959/VO2/Agent/logs_agent/inference_%j.log
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=40G
#SBATCH --cpus-per-task=16
#SBATCH --time=200:00:00

set -e

BASE_DIR="/info/raid-etu/m2/s2405959/VO2/Agent"
INPUT_JSON="/info/corpus/Blizzard2023_segmented/segmented/NEB_train/test_random_50vs_agent/target_prompt_map.json"

# Check if JSON input exists
if [ ! -f "$INPUT_JSON" ]; then
    echo "Error: Input JSON not found at $INPUT_JSON"
    exit 1
fi

# Use the latest checkpoint available
MODEL_PATH=$(ls -1 "$BASE_DIR"/checkpoint/agent_model_sentence_*.pth 2>/dev/null | sort -V | tail -n 1)
if [ -z "$MODEL_PATH" ]; then
    MODEL_PATH="$BASE_DIR/model/agent_model.pth"
fi

if [ ! -f "$MODEL_PATH" ]; then
  echo "Error: Model checkpoint not found at $MODEL_PATH"
  echo "Available checkpoints:"
  ls -lh "$BASE_DIR/checkpoint/"
  exit 1
fi

echo "========================================="
echo "INFERENCE MODE - NO TRAINING"
echo "========================================="
echo "Model: $MODEL_PATH"
echo "Input JSON: $INPUT_JSON"
echo ""

TEMP_DIR="$BASE_DIR/temp_inference"
mkdir -p "$TEMP_DIR"

VEC_1024="$TEMP_DIR/vec_1024.npy"
VEC_256="$TEMP_DIR/vec_256.npy"
RETRIEVED_VEC="$TEMP_DIR/retrieved_vec_256.npy"
SIM_OUTPUT="$TEMP_DIR/similarity_result.json"
AUDIO_PATH_FILE="$TEMP_DIR/generated_audio_path.txt"
CER_OUTPUT="$TEMP_DIR/cer_value.txt"
REWARD_OUTPUT="$TEMP_DIR/reward_value.txt"
INFERENCE_LOG="$BASE_DIR/logs_agent/inference_results.csv"
TARGETS_B64="$TEMP_DIR/targets_b64.txt"
PROMPTS_FILE="$TEMP_DIR/current_prompts.json"

# Initialize inference log
echo "run_num,target_num,iteration,input_sentence,selected_prompt,prompt_length,cosine_similarity,cer,reward" > "$INFERENCE_LOG"

source $(conda info --base)/etc/profile.d/conda.sh
conda activate agent_env
unset PYTHONPATH
unset VIRTUAL_ENV
export PYTHONNOUSERSITE=1
PYTHON_BIN="$CONDA_PREFIX/bin/python"

if [ ! -x "$PYTHON_BIN" ]; then
    echo "Error: Python not found at $PYTHON_BIN"
    exit 1
fi

echo "Inference Started"
echo "Reading target/prompt map from: $INPUT_JSON"

# Build a JSONL (base64) list of targets with prompts for safe iteration
"$PYTHON_BIN" - <<'PY' "$INPUT_JSON" > "$TARGETS_B64"
import base64
import json
import sys

data = json.load(open(sys.argv[1], "r", encoding="utf-8"))

for target, prompts in data.items():
    if isinstance(prompts, dict):
        prompts = [prompts[f"prompt_{i}"] for i in range(1, 11)]
    record = {"target": target, "prompts": prompts}
    b64 = base64.b64encode(json.dumps(record, ensure_ascii=False).encode("utf-8")).decode("utf-8")
    print(b64)
PY

TOTAL_TARGETS=$(wc -l < "$TARGETS_B64")
ITERATIONS=10
TOTAL_RUNS=$((TOTAL_TARGETS * ITERATIONS))
echo "Total targets: $TOTAL_TARGETS"
echo "Iterations per target: $ITERATIONS"
echo "Total inference runs: $TOTAL_RUNS"

# NO EXPLORATION - Pure greedy selection
EPSILON=0.0

RUN_NUM=0
TARGET_NUM=0
while IFS= read -r TARGET_B64 || [ -n "$TARGET_B64" ]; do
    TARGET_NUM=$((TARGET_NUM + 1))
    TARGET_JSON=$("$PYTHON_BIN" - <<'PY' "$TARGET_B64"
import base64
import json
import sys

data = json.loads(base64.b64decode(sys.argv[1]).decode("utf-8"))
print(json.dumps(data, ensure_ascii=False))
PY
)

    SENTENCE=$("$PYTHON_BIN" - <<'PY' "$TARGET_JSON"
import json
import sys

data = json.loads(sys.argv[1])
print(data["target"])
PY
)

    "$PYTHON_BIN" - <<'PY' "$TARGET_JSON" "$PROMPTS_FILE"
import json
import sys

data = json.loads(sys.argv[1])
with open(sys.argv[2], "w", encoding="utf-8") as f:
    json.dump(data["prompts"], f, ensure_ascii=False, indent=2)
PY

    for ITER in $(seq 1 $ITERATIONS); do
        RUN_NUM=$((RUN_NUM + 1))
    
    echo ""
    echo "========================================"
    echo "Run $RUN_NUM / $TOTAL_RUNS (Target $TARGET_NUM / $TOTAL_TARGETS, Iteration $ITER / $ITERATIONS)"
    echo "Input: \"$SENTENCE\""
    echo "Mode: INFERENCE (Epsilon=0.0, No Training)"
    echo "========================================"
    
    echo "[1/6] Running Text Encoder..."
    "$PYTHON_BIN" "$BASE_DIR/model/text_encoder.py" --sentence "$SENTENCE" --output "$VEC_1024"

    echo "[2/6] Running Agent Model (Greedy Selection)..."
    "$PYTHON_BIN" "$BASE_DIR/model/agent_model.py" \
      --input "$VEC_1024" \
      --output "$VEC_256" \
      --model_path "$MODEL_PATH" \
      --exploration_noise 0.0 \
      --epsilon "$EPSILON"

        echo "[3/6] Running Constrained Similarity Search..."
        "$PYTHON_BIN" "$BASE_DIR/Similarity/constrained_similarity.py" \
            --vector "$VEC_256" \
            --prompts_json "$PROMPTS_FILE" \
            --output "$SIM_OUTPUT" \
            --output_vector "$RETRIEVED_VEC"
    
    # Extract similarity score, prompt text, and prompt index
    if [ -f "$SIM_OUTPUT" ]; then
        SIMILARITY_SCORE=$("$PYTHON_BIN" -c "import json; print(json.load(open('$SIM_OUTPUT'))['cosine_similarity'])")
        PROMPT_TEXT=$("$PYTHON_BIN" -c "import json; print(json.load(open('$SIM_OUTPUT'))['prompt_transcription'])")
        PROMPT_LENGTH=$("$PYTHON_BIN" -c "import json; print(len(json.load(open('$SIM_OUTPUT'))['prompt_transcription'].split()))")
        echo "Selected Prompt: \"$PROMPT_TEXT\" (Length: $PROMPT_LENGTH words)"
        echo "Cosine Similarity: $SIMILARITY_SCORE"
    else
        SIMILARITY_SCORE="N/A"
        PROMPT_TEXT="N/A"
        PROMPT_LENGTH="0"
    fi

    echo "[4/6] Generating Audio with ZipVoice..."
    "$PYTHON_BIN" "$BASE_DIR/generate_audio/generate_with_zipVoice.py" \
      --similarity_output "$SIM_OUTPUT" \
      --target_text "$SENTENCE" \
      --output_path_file "$AUDIO_PATH_FILE"

    if [ -f "$AUDIO_PATH_FILE" ]; then
        GENERATED_AUDIO=$(cat "$AUDIO_PATH_FILE")
        echo "Generated Audio: $GENERATED_AUDIO"
    else
        echo "Error: Audio path file not found."
        exit 1
    fi

    echo "[5/6] Calculating CER..."
    "$PYTHON_BIN" "$BASE_DIR/assess_CER/calculate_cer.py" "$SENTENCE" "$GENERATED_AUDIO" --output_cer "$CER_OUTPUT"

    if [ -f "$CER_OUTPUT" ]; then
        CER_VALUE=$(cat "$CER_OUTPUT")
        echo "CER: $CER_VALUE"
    else
        echo "Error: CER output file not found."
        exit 1
    fi

    echo "[6/6] Calculating Weighted Reward..."
    "$PYTHON_BIN" "$BASE_DIR/assess_CER/weighted_cer.py" "$CER_VALUE" --output_reward "$REWARD_OUTPUT"

    if [ -f "$REWARD_OUTPUT" ]; then
        REWARD_VALUE=$(cat "$REWARD_OUTPUT")
        echo "Reward: $REWARD_VALUE"
    else
        echo "Error: Reward output file not found."
        exit 1
    fi
    
    # Log results (escape quotes in text for CSV)
    SENTENCE_ESCAPED=$(echo "$SENTENCE" | sed 's/"/""/g')
    PROMPT_ESCAPED=$(echo "$PROMPT_TEXT" | sed 's/"/""/g')
    echo "$RUN_NUM,$TARGET_NUM,$ITER,\"$SENTENCE_ESCAPED\",\"$PROMPT_ESCAPED\",$PROMPT_LENGTH,$SIMILARITY_SCORE,$CER_VALUE,$REWARD_VALUE" >> "$INFERENCE_LOG"
    
    echo "Completed run $RUN_NUM (NO TRAINING PERFORMED)"

    done

done < "$TARGETS_B64"

echo ""
echo "========================================"
echo "Inference Completed - Processed $TOTAL_RUNS runs"
echo "Results saved to: $INFERENCE_LOG"
echo "========================================"

# Generate summary statistics
echo ""
echo "INFERENCE SUMMARY:"
"$PYTHON_BIN" -c "
import pandas as pd
import numpy as np

df = pd.read_csv('$INFERENCE_LOG')
print(f'Average CER: {df[\"cer\"].mean():.4f}')
print(f'Average Reward: {df[\"reward\"].mean():.4f}')
print(f'Average Prompt Length: {df[\"prompt_length\"].mean():.2f} words')
print(f'Average Cosine Similarity: {df[\"cosine_similarity\"].mean():.4f}')
print(f'\\nBest Performance (Lowest CER):')
best_idx = df[\"cer\"].idxmin()
print(f'  Sentence: {df.loc[best_idx, \"input_sentence\"]}')
print(f'  Prompt: {df.loc[best_idx, \"selected_prompt\"]}')
print(f'  CER: {df.loc[best_idx, \"cer\"]:.4f}')
print(f'\\nWorst Performance (Highest CER):')
worst_idx = df[\"cer\"].idxmax()
print(f'  Sentence: {df.loc[worst_idx, \"input_sentence\"]}')
print(f'  Prompt: {df.loc[worst_idx, \"selected_prompt\"]}')
print(f'  CER: {df.loc[worst_idx, \"cer\"]:.4f}')
"

echo ""
echo "Inference pipeline complete."
