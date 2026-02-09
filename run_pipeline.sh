#!/bin/bash
#SBATCH --job-name=agent_pipeline
#SBATCH --output=/info/raid-etu/m2/s2405959/VO2/Agent/logs_agent/pipeline_%j.log
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=40G
#SBATCH --cpus-per-task=16
#SBATCH --time=200:00:00

set -e

BASE_DIR="/info/raid-etu/m2/s2405959/VO2/Agent"
SENTENCES_FILE="$BASE_DIR/DB/sentences.txt"

# Check if sentences file exists
if [ ! -f "$SENTENCES_FILE" ]; then
  echo "Error: Sentences file not found at $SENTENCES_FILE"
  echo "Please run: python3 $BASE_DIR/DB/extract_sentences.py"
  exit 1
fi

# Backup existing model if present
if [ -f "$BASE_DIR/model/agent_model.pth" ]; then
  TIMESTAMP=$(date +%Y%m%d_%H%M%S)
  cp "$BASE_DIR/model/agent_model.pth" "$BASE_DIR/model/agent_model.pth.backup_$TIMESTAMP"
  echo "Backed up existing model to agent_model.pth.backup_$TIMESTAMP"
fi

if [ -f "$BASE_DIR/model/replay_buffer.pkl" ]; then
  TIMESTAMP=$(date +%Y%m%d_%H%M%S)
  cp "$BASE_DIR/model/replay_buffer.pkl" "$BASE_DIR/model/replay_buffer.pkl.backup_$TIMESTAMP"
  echo "Backed up replay buffer to replay_buffer.pkl.backup_$TIMESTAMP"
fi

TEMP_DIR="$BASE_DIR/temp_pipeline"
mkdir -p "$TEMP_DIR"

VEC_1024="$TEMP_DIR/vec_1024.npy"
VEC_256="$TEMP_DIR/vec_256.npy"
RETRIEVED_VEC="$TEMP_DIR/retrieved_vec_256.npy"
SIM_OUTPUT="$TEMP_DIR/similarity_result.json"
AUDIO_PATH_FILE="$TEMP_DIR/generated_audio_path.txt"
CER_OUTPUT="$TEMP_DIR/cer_value.txt"
REWARD_OUTPUT="$TEMP_DIR/reward_value.txt"
MODEL_PATH="$BASE_DIR/model/agent_model.pth"
BUFFER_PATH="$BASE_DIR/model/replay_buffer.pkl"
TRAINING_LOG="$BASE_DIR/logs_agent/training_progress.csv"
CHECKPOINT_DIR="$BASE_DIR/checkpoint"
CHECKPOINT_INTERVAL=100
AGENT_ENV_LIB="/info/etu/m2/s2405959/miniconda3/envs/agent_env/lib"

# Create checkpoint directory if it doesn't exist
mkdir -p "$CHECKPOINT_DIR"

# Initialize training log if it doesn't exist
if [ ! -f "$TRAINING_LOG" ]; then
  echo "sentence_num,iteration,cer,reward,similarity_score,epsilon" > "$TRAINING_LOG"
fi

echo "Pipeline Started"
echo "Reading sentences from: $SENTENCES_FILE"

# Count total sentences
TOTAL_SENTENCES=$(wc -l < "$SENTENCES_FILE")
echo "Total sentences to process: $TOTAL_SENTENCES"

# Epsilon decay schedule
# Start with moderate exploration (0.3) and decay to 0.05 over 500 episodes
EPSILON_START=0.3
EPSILON_END=0.05
EPSILON_DECAY_STEPS=500

# Entropy coefficient for diversity
ENTROPY_COEF=0.15

# Baseline reward tracking file for advantage estimation
BASELINE_FILE="$TEMP_DIR/baseline_reward.txt"
if [ ! -f "$BASELINE_FILE" ]; then
    echo "0.5" > "$BASELINE_FILE"  # Initialize baseline to 0.5
fi

# Outer loop: iterate over each sentence
SENTENCE_NUM=0
while IFS= read -r SENTENCE || [ -n "$SENTENCE" ]; do
    SENTENCE_NUM=$((SENTENCE_NUM + 1))
    
    # Linear epsilon decay
    # epsilon = max(EPSILON_END, EPSILON_START - (EPSILON_START - EPSILON_END) * step / DECAY_STEPS)
    if [ $SENTENCE_NUM -le $EPSILON_DECAY_STEPS ]; then
        EPSILON=$(python3 -c "print(max($EPSILON_END, $EPSILON_START - ($EPSILON_START - $EPSILON_END) * $SENTENCE_NUM / $EPSILON_DECAY_STEPS))")
    else
        EPSILON=$EPSILON_END
    fi
    
    echo ""
    echo "========================================"
    echo "Processing Sentence $SENTENCE_NUM / $TOTAL_SENTENCES"
    echo "Input: \"$SENTENCE\""
    echo "Epsilon (exploration): $EPSILON"
    echo "========================================"
    
    echo "[1/7] Running Text Encoder..."
    conda run -n agent_env python "$BASE_DIR/model/text_encoder.py" --sentence "$SENTENCE" --output "$VEC_1024"

    echo "[2/7] Running Agent Model..."
    conda run -n agent_env python "$BASE_DIR/model/agent_model.py" \
      --input "$VEC_1024" \
      --output "$VEC_256" \
      --model_path "$MODEL_PATH" \
      --exploration_noise 0.15 \
      --epsilon "$EPSILON"

    echo "[3/7] Running Similarity Search..."
    conda run -n agent_env python "$BASE_DIR/Similarity/asess_similarty.py" --vector "$VEC_256" --output "$SIM_OUTPUT" --output_vector "$RETRIEVED_VEC"
    
    # Extract similarity score and prompt index for logging and diversity tracking
    if [ -f "$SIM_OUTPUT" ]; then
        SIMILARITY_SCORE=$(conda run -n agent_env python -c "import json; print(json.load(open('$SIM_OUTPUT'))['cosine_similarity'])")
        PROMPT_IDX=$(conda run -n agent_env python -c "import json; print(json.load(open('$SIM_OUTPUT')).get('nearest_idx', -1))")
    else
        SIMILARITY_SCORE="N/A"
        PROMPT_IDX="-1"
    fi

    echo "[4/7] Generating Audio with ZipVoice..."
    conda run -n agent_env python "$BASE_DIR/generate_audio/generate_with_zipVoice.py" \
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
    env LD_LIBRARY_PATH="$AGENT_ENV_LIB${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
        conda run -n agent_env python "$BASE_DIR/assess_CER/calculate_cer.py" "$SENTENCE" "$GENERATED_AUDIO" --output_cer "$CER_OUTPUT"

    if [ -f "$CER_OUTPUT" ]; then
        CER_VALUE=$(cat "$CER_OUTPUT")
    else
        echo "Error: CER output file not found."
        exit 1
    fi

    echo "[6/7] Calculating Weighted Reward..."
    conda run -n agent_env python "$BASE_DIR/assess_CER/weighted_cer.py" "$CER_VALUE" --output_reward "$REWARD_OUTPUT"

    if [ -f "$REWARD_OUTPUT" ]; then
        REWARD_VALUE=$(cat "$REWARD_OUTPUT")
        echo "Sentence $SENTENCE_NUM Reward: $REWARD_VALUE"
    else
        echo "Error: Reward output file not found."
        exit 1
    fi

    # Read current baseline reward for advantage estimation
    BASELINE_REWARD=$(cat "$BASELINE_FILE")
    
    echo "[7/7] Training Agent (Improved Contrastive Learning with Advantage)..."
    echo "    Epsilon: $EPSILON, Baseline: $BASELINE_REWARD"
    conda run -n agent_env python "$BASE_DIR/model/train_agent.py" \
      --input_state "$VEC_1024" \
      --retrieved_action "$RETRIEVED_VEC" \
      --reward "$REWARD_VALUE" \
      --baseline_reward "$BASELINE_REWARD" \
      --prompt_idx "$PROMPT_IDX" \
      --model_path "$MODEL_PATH" \
      --buffer_path "$BUFFER_PATH" \
      --use_replay \
      --buffer_capacity 5000 \
      --entropy_coef "$ENTROPY_COEF" \
      --diversity_penalty_weight 0.05 \
      --batch_size 64 \
      --num_epochs 10 \
      --epsilon "$EPSILON"
    
    # Update baseline reward using exponential moving average (alpha=0.1)
    NEW_BASELINE=$(python3 -c "print(0.9 * $BASELINE_REWARD + 0.1 * $REWARD_VALUE)")
    echo "$NEW_BASELINE" > "$BASELINE_FILE"
    
    # Log training progress
    echo "$SENTENCE_NUM,1,$CER_VALUE,$REWARD_VALUE,$SIMILARITY_SCORE" >> "$TRAINING_LOG"
    
    # Evaluation protocol (every 50 sentences)
    if [ $((SENTENCE_NUM % 50)) -eq 0 ]; then
        echo ""
        echo "========================================"
        echo "[EVALUATION] Running evaluation at sentence $SENTENCE_NUM"
        echo "========================================"
        
        # Run validation on first 10 sentences with NO exploration
        VAL_FILE="$BASE_DIR/DB/sentences_val.txt"
        if [ -f "$VAL_FILE" ]; then
            python3 "$BASE_DIR/model/validate_agent.py" \
                --val_sentences_file "$VAL_FILE" \
                --model_path "$MODEL_PATH" \
                --max_sentences 10 \
                --epsilon 0.0
        else
            echo "[EVALUATION] Validation file not found: $VAL_FILE"
            echo "[EVALUATION] Skipping evaluation (create with: python scripts/split_sentences.py)"
        fi
        echo "========================================"
    fi
    
    # Save checkpoint every CHECKPOINT_INTERVAL sentences
    if [ $((SENTENCE_NUM % CHECKPOINT_INTERVAL)) -eq 0 ]; then
        echo "[CHECKPOINT] Saving checkpoint at sentence $SENTENCE_NUM..."
        CHECKPOINT_MODEL="$CHECKPOINT_DIR/agent_model_sentence_${SENTENCE_NUM}.pth"
        CHECKPOINT_BUFFER="$CHECKPOINT_DIR/replay_buffer_sentence_${SENTENCE_NUM}.pkl"
        cp "$MODEL_PATH" "$CHECKPOINT_MODEL"
        cp "$BUFFER_PATH" "$CHECKPOINT_BUFFER"
        echo "[CHECKPOINT] Model saved to: $CHECKPOINT_MODEL"
        echo "[CHECKPOINT] Replay buffer saved to: $CHECKPOINT_BUFFER"
    fi
    
    echo "Completed sentence $SENTENCE_NUM"

done < "$SENTENCES_FILE"

echo ""
echo "========================================"
echo "Pipeline Completed - Processed $TOTAL_SENTENCES sentences"
echo "========================================"
