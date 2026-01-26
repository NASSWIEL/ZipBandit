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

# Create checkpoint directory if it doesn't exist
mkdir -p "$CHECKPOINT_DIR"

# Initialize training log if it doesn't exist
if [ ! -f "$TRAINING_LOG" ]; then
  echo "sentence_num,iteration,cer,reward,similarity_score,epsilon" > "$TRAINING_LOG"
fi

source $(conda info --base)/etc/profile.d/conda.sh
conda activate agent_env

echo "Pipeline Started"
echo "Reading sentences from: $SENTENCES_FILE"

# Count total sentences
TOTAL_SENTENCES=$(wc -l < "$SENTENCES_FILE")
echo "Total sentences to process: $TOTAL_SENTENCES"

# EXPERT FIX: FIXED EPSILON (constant exploration)
# Don't decay epsilon until we confirm learning is happening
# Keep at 0.5 to maintain 50% exploration throughout training
FIXED_EPSILON=0.5  # Constant 50% exploration
ENTROPY_COEF=0.02  # Entropy regularization

# Outer loop: iterate over each sentence
SENTENCE_NUM=0
while IFS= read -r SENTENCE || [ -n "$SENTENCE" ]; do
    SENTENCE_NUM=$((SENTENCE_NUM + 1))
    
    # Use fixed epsilon (no decay) - keep exploration high until model shows learning
    EPSILON=$FIXED_EPSILON
    
    echo ""
    echo "========================================"
    echo "Processing Sentence $SENTENCE_NUM / $TOTAL_SENTENCES"
    echo "Input: \"$SENTENCE\""
    echo "Epsilon (exploration): $EPSILON"
    echo "========================================"
    
    echo "[1/7] Running Text Encoder..."
    python3 "$BASE_DIR/model/text_encoder.py" --sentence "$SENTENCE" --output "$VEC_1024"

    echo "[2/7] Running Agent Model..."
    python3 "$BASE_DIR/model/agent_model.py" \
      --input "$VEC_1024" \
      --output "$VEC_256" \
      --model_path "$MODEL_PATH" \
      --exploration_noise 0.15 \
      --epsilon "$EPSILON"

    echo "[3/7] Running Similarity Search..."
    python3 "$BASE_DIR/Similarity/asess_similarty.py" --vector "$VEC_256" --output "$SIM_OUTPUT" --output_vector "$RETRIEVED_VEC"
    
    # Extract similarity score and prompt index for logging and diversity tracking
    if [ -f "$SIM_OUTPUT" ]; then
        SIMILARITY_SCORE=$(python3 -c "import json; print(json.load(open('$SIM_OUTPUT'))['cosine_similarity'])")
        PROMPT_IDX=$(python3 -c "import json; print(json.load(open('$SIM_OUTPUT')).get('nearest_idx', -1))")
    else
        SIMILARITY_SCORE="N/A"
        PROMPT_IDX="-1"
    fi

    echo "[4/7] Generating Audio with ZipVoice..."
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
        echo "Sentence $SENTENCE_NUM Reward: $REWARD_VALUE"
    else
        echo "Error: Reward output file not found."
        exit 1
    fi

    echo "[7/7] Training Agent (Contrastive Learning with Fixed Exploration)..."
    python3 "$BASE_DIR/model/train_agent.py" \
      --input_state "$VEC_1024" \
      --retrieved_action "$RETRIEVED_VEC" \
      --reward "$REWARD_VALUE" \
      --prompt_idx "$PROMPT_IDX" \
      --model_path "$MODEL_PATH" \
      --buffer_path "$BUFFER_PATH" \
      --use_replay \
      --buffer_capacity 5000 \
      --entropy_coef 0.02 \
      --diversity_penalty_weight 0.1 \
      --batch_size 64 \
      --num_epochs 10 \
      --epsilon "$EPSILON"
    
    # Log training progress
    echo "$SENTENCE_NUM,1,$CER_VALUE,$REWARD_VALUE,$SIMILARITY_SCORE" >> "$TRAINING_LOG"
    
    # EXPERT FIX #4: EVALUATION PROTOCOL (every 50 sentences)
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
