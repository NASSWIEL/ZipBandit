#!/bin/bash
#SBATCH --job-name=agent_pipeline
#SBATCH --output=/info/raid-etu/m2/s2405959/VO2/Agent/logs_agent/pipeline_%j.log
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=40G
#SBATCH --cpus-per-task=16
#SBATCH --time=20:00:00

set -e

if [ -z "$1" ]; then
  echo "Usage: $0 \"<sentence>\""
  exit 1
fi

SENTENCE="$1"
BASE_DIR="/info/raid-etu/m2/s2405959/VO2/Agent"
TEMP_DIR="$BASE_DIR/temp_pipeline"
mkdir -p "$TEMP_DIR"

VEC_1024="$TEMP_DIR/vec_1024.npy"
VEC_100="$TEMP_DIR/vec_100.npy"
RETRIEVED_VEC="$TEMP_DIR/retrieved_vec_100.npy"
SIM_OUTPUT="$TEMP_DIR/similarity_result.json"
AUDIO_PATH_FILE="$TEMP_DIR/generated_audio_path.txt"
CER_OUTPUT="$TEMP_DIR/cer_value.txt"
REWARD_OUTPUT="$TEMP_DIR/reward_value.txt"
MODEL_PATH="$BASE_DIR/model/agent_model.pth"

source $(conda info --base)/etc/profile.d/conda.sh
conda activate agent_env

echo "Pipeline Started"
echo "Input Sentence: \"$SENTENCE\""

echo "[1/7] Running Text Encoder..."
python3 "$BASE_DIR/model/text_encoder.py" --sentence "$SENTENCE" --output "$VEC_1024"

# Loop for Reinforcement Learning Iterations
NUM_ITERATIONS=10

for ((i=1; i<=NUM_ITERATIONS; i++)); do
    echo "Iteration $i / $NUM_ITERATIONS"

    echo "[2/7] Running Agent Model..."
    python3 "$BASE_DIR/model/agent_model.py" --input "$VEC_1024" --output "$VEC_100" --model_path "$MODEL_PATH"

    echo "[3/7] Running Similarity Search..."
    python3 "$BASE_DIR/Similarity/asess_similarty.py" --vector "$VEC_100" --output "$SIM_OUTPUT" --output_vector "$RETRIEVED_VEC"

    echo "[4/7] Generating Audio with ZipVoice..."
    # We need to ensure unique filenames or handle overwrites. 
    # generate_with_zipVoice.py handles naming based on text and prompt ID.
    # Since the prompt might change, the filename might change.
    # If the prompt is the same, it will overwrite, which is fine for this loop as we just need the latest for CER.
    python3 "$BASE_DIR/generate_audio/generate_with_zipVoice.py" \
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

    echo "[5/7] Calculating CER..."
    python3 "$BASE_DIR/assess_CER/calculate_cer.py" "$SENTENCE" "$GENERATED_AUDIO" --output_cer "$CER_OUTPUT"

    if [ -f "$CER_OUTPUT" ]; then
        CER_VALUE=$(cat "$CER_OUTPUT")
    else
        echo "Error: CER output file not found."
        exit 1
    fi

    echo "[6/7] Calculating Weighted Reward..."
    python3 "$BASE_DIR/assess_CER/weighted_cer.py" "$CER_VALUE" --output_reward "$REWARD_OUTPUT"

    if [ -f "$REWARD_OUTPUT" ]; then
        REWARD_VALUE=$(cat "$REWARD_OUTPUT")
        echo "Iteration $i Reward: $REWARD_VALUE"
    else
        echo "Error: Reward output file not found."
        exit 1
    fi

    echo "[7/7] Training Agent (Contextual Bandit Update)..."
    python3 "$BASE_DIR/model/train_agent.py" \
      --input_state "$VEC_1024" \
      --retrieved_action "$RETRIEVED_VEC" \
      --reward "$REWARD_VALUE" \
      --model_path "$MODEL_PATH"

done

echo "Pipeline Completed"
