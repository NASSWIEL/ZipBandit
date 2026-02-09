#!/bin/bash
#SBATCH --job-name=agent_filtered_6w
#SBATCH --output=/info/raid-etu/m2/s2405959/VO2/Agent/pipeline_v2_filtered/logs_agent/pipeline_%j.log
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=40G
#SBATCH --cpus-per-task=16
#SBATCH --time=200:00:00

# ============================================================================
# FILTERED PIPELINE - ISOLATED TRAINING (Min 6 words per sentence)
# ============================================================================
# This pipeline is COMPLETELY ISOLATED from the baseline pipeline.
# It uses:
#   - FILTERED embeddings index (6+ words, 5000 sentences)
#   - Separate model directory
#   - Separate logs directory
#   - Separate checkpoint directory
#   - Separate temporary files
#
# Purpose: Compare filtered (linguistically constrained) vs unfiltered training
# ============================================================================

set -e

# =========================== ISOLATION CONFIGURATION ===========================
# All paths must be under pipeline_v2_filtered to ensure full isolation
PIPELINE_BASE="/info/raid-etu/m2/s2405959/VO2/Agent/pipeline_v2_filtered"
SHARED_BASE="/info/raid-etu/m2/s2405959/VO2/Agent"

# Isolated directories for this pipeline
MODEL_DIR="$PIPELINE_BASE/model"
LOGS_DIR="$PIPELINE_BASE/logs_agent"
CHECKPOINT_DIR="$PIPELINE_BASE/checkpoint"
TEMP_DIR="$PIPELINE_BASE/temp_pipeline"

# Shared components (read-only, no writes to these)
SENTENCES_FILE="$SHARED_BASE/DB/sentences.txt"
TEXT_ENCODER_SCRIPT="$SHARED_BASE/model/text_encoder.py"
AGENT_MODEL_SCRIPT="$SHARED_BASE/model/agent_model.py"
TRAIN_AGENT_SCRIPT="$SHARED_BASE/model/train_agent.py"
VALIDATE_SCRIPT="$SHARED_BASE/model/validate_agent.py"
AUDIO_GEN_SCRIPT="$SHARED_BASE/generate_audio/generate_with_zipVoice.py"
CER_SCRIPT="$SHARED_BASE/assess_CER/calculate_cer.py"
REWARD_SCRIPT="$SHARED_BASE/assess_CER/weighted_cer.py"
VAL_FILE="$SHARED_BASE/DB/sentences_val.txt"

# ISOLATED SIMILARITY SEARCH (uses FILTERED index)
SIMILARITY_SCRIPT="$PIPELINE_BASE/Similarity/assess_similarity_filtered.py"

# ============================================================================

# Create all isolated directories
mkdir -p "$MODEL_DIR"
mkdir -p "$LOGS_DIR"
mkdir -p "$CHECKPOINT_DIR"
mkdir -p "$TEMP_DIR"

# Check if sentences file exists
if [ ! -f "$SENTENCES_FILE" ]; then
  echo "Error: Sentences file not found at $SENTENCES_FILE"
  exit 1
fi

# Backup existing model if present (in ISOLATED directory)
if [ -f "$MODEL_DIR/agent_model.pth" ]; then
  TIMESTAMP=$(date +%Y%m%d_%H%M%S)
  cp "$MODEL_DIR/agent_model.pth" "$MODEL_DIR/agent_model.pth.backup_$TIMESTAMP"
  echo "[FILTERED] Backed up existing model to agent_model.pth.backup_$TIMESTAMP"
fi

if [ -f "$MODEL_DIR/replay_buffer.pkl" ]; then
  TIMESTAMP=$(date +%Y%m%d_%H%M%S)
  cp "$MODEL_DIR/replay_buffer.pkl" "$MODEL_DIR/replay_buffer.pkl.backup_$TIMESTAMP"
  echo "[FILTERED] Backed up replay buffer to replay_buffer.pkl.backup_$TIMESTAMP"
fi

# Isolated temp files
VEC_1024="$TEMP_DIR/vec_1024.npy"
VEC_256="$TEMP_DIR/vec_256.npy"
RETRIEVED_VEC="$TEMP_DIR/retrieved_vec_256.npy"
SIM_OUTPUT="$TEMP_DIR/similarity_result.json"
AUDIO_PATH_FILE="$TEMP_DIR/generated_audio_path.txt"
CER_OUTPUT="$TEMP_DIR/cer_value.txt"
REWARD_OUTPUT="$TEMP_DIR/reward_value.txt"

# Isolated model and buffer paths
MODEL_PATH="$MODEL_DIR/agent_model.pth"
BUFFER_PATH="$MODEL_DIR/replay_buffer.pkl"
TRAINING_LOG="$LOGS_DIR/training_progress.csv"
CHECKPOINT_INTERVAL=100

# Environment library path
AGENT_ENV_LIB="/info/etu/m2/s2405959/miniconda3/envs/agent_env/lib"

# Initialize training log if it doesn't exist
if [ ! -f "$TRAINING_LOG" ]; then
  echo "sentence_num,iteration,cer,reward,similarity_score,epsilon" > "$TRAINING_LOG"
fi

echo "============================================================================"
echo "[FILTERED PIPELINE] Started - Min 6 Words Configuration"
echo "============================================================================"
echo ""
echo "ISOLATION STATUS:"
echo "  Model directory:      $MODEL_DIR"
echo "  Logs directory:       $LOGS_DIR"
echo "  Checkpoint directory: $CHECKPOINT_DIR"
echo "  Temp directory:       $TEMP_DIR"
echo "  Similarity script:    $SIMILARITY_SCRIPT"
echo ""
echo "Reading sentences from: $SENTENCES_FILE"

# Count total sentences
TOTAL_SENTENCES=$(wc -l < "$SENTENCES_FILE")
echo "Total sentences to process: $TOTAL_SENTENCES"

# EPSILON DECAY SCHEDULE (same as baseline for fair comparison)
EPSILON_START=0.3
EPSILON_END=0.05
EPSILON_DECAY_STEPS=500

# ENTROPY COEFFICIENT (same as baseline for fair comparison)
ENTROPY_COEF=0.15

# Baseline reward tracking (ISOLATED)
BASELINE_FILE="$TEMP_DIR/baseline_reward.txt"
if [ ! -f "$BASELINE_FILE" ]; then
    echo "0.5" > "$BASELINE_FILE"
fi

# ============================= MAIN TRAINING LOOP =============================
SENTENCE_NUM=0
while IFS= read -r SENTENCE || [ -n "$SENTENCE" ]; do
    SENTENCE_NUM=$((SENTENCE_NUM + 1))
    
    # Linear epsilon decay (identical to baseline)
    if [ $SENTENCE_NUM -le $EPSILON_DECAY_STEPS ]; then
        EPSILON=$(python3 -c "print(max($EPSILON_END, $EPSILON_START - ($EPSILON_START - $EPSILON_END) * $SENTENCE_NUM / $EPSILON_DECAY_STEPS))")
    else
        EPSILON=$EPSILON_END
    fi
    
    echo ""
    echo "========================================"
    echo "[FILTERED] Processing Sentence $SENTENCE_NUM / $TOTAL_SENTENCES"
    echo "Input: \"$SENTENCE\""
    echo "Epsilon (exploration): $EPSILON"
    echo "========================================"
    
    echo "[1/7] Running Text Encoder..."
    conda run -n agent_env python "$TEXT_ENCODER_SCRIPT" --sentence "$SENTENCE" --output "$VEC_1024"

    echo "[2/7] Running Agent Model (ISOLATED)..."
    conda run -n agent_env python "$AGENT_MODEL_SCRIPT" \
      --input "$VEC_1024" \
      --output "$VEC_256" \
      --model_path "$MODEL_PATH" \
      --exploration_noise 0.15 \
      --epsilon "$EPSILON"

    echo "[3/7] Running Similarity Search (FILTERED 6+ WORDS INDEX)..."
    conda run -n agent_env python "$SIMILARITY_SCRIPT" \
      --vector "$VEC_256" \
      --output "$SIM_OUTPUT" \
      --output_vector "$RETRIEVED_VEC"
    
    # Extract similarity score and prompt index for logging
    if [ -f "$SIM_OUTPUT" ]; then
        SIMILARITY_SCORE=$(conda run -n agent_env python -c "import json; print(json.load(open('$SIM_OUTPUT'))['cosine_similarity'])")
        PROMPT_IDX=$(conda run -n agent_env python -c "import json; print(json.load(open('$SIM_OUTPUT')).get('index', -1))")
    else
        SIMILARITY_SCORE="N/A"
        PROMPT_IDX="-1"
    fi

    echo "[4/7] Generating Audio with ZipVoice..."
    conda run -n agent_env python "$AUDIO_GEN_SCRIPT" \
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
        conda run -n agent_env python "$CER_SCRIPT" "$SENTENCE" "$GENERATED_AUDIO" --output_cer "$CER_OUTPUT"

    if [ -f "$CER_OUTPUT" ]; then
        CER_VALUE=$(cat "$CER_OUTPUT")
    else
        echo "Error: CER output file not found."
        exit 1
    fi

    echo "[6/7] Calculating Weighted Reward..."
    conda run -n agent_env python "$REWARD_SCRIPT" "$CER_VALUE" --output_reward "$REWARD_OUTPUT"

    if [ -f "$REWARD_OUTPUT" ]; then
        REWARD_VALUE=$(cat "$REWARD_OUTPUT")
        echo "[FILTERED] Sentence $SENTENCE_NUM Reward: $REWARD_VALUE"
    else
        echo "Error: Reward output file not found."
        exit 1
    fi

    # Read current baseline reward (ISOLATED)
    BASELINE_REWARD=$(cat "$BASELINE_FILE")
    
    echo "[7/7] Training Agent (ISOLATED MODEL)..."
    echo "    Epsilon: $EPSILON, Baseline: $BASELINE_REWARD"
    conda run -n agent_env python "$TRAIN_AGENT_SCRIPT" \
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
    
    # Update baseline reward (ISOLATED)
    NEW_BASELINE=$(python3 -c "print(0.9 * $BASELINE_REWARD + 0.1 * $REWARD_VALUE)")
    echo "$NEW_BASELINE" > "$BASELINE_FILE"
    
    # Log training progress (ISOLATED LOG)
    echo "$SENTENCE_NUM,1,$CER_VALUE,$REWARD_VALUE,$SIMILARITY_SCORE,$EPSILON" >> "$TRAINING_LOG"
    
    # Evaluation every 50 sentences
    if [ $((SENTENCE_NUM % 50)) -eq 0 ]; then
        echo ""
        echo "========================================"
        echo "[FILTERED EVALUATION] at sentence $SENTENCE_NUM"
        echo "========================================"
        
        if [ -f "$VAL_FILE" ]; then
            python3 "$VALIDATE_SCRIPT" \
                --val_sentences_file "$VAL_FILE" \
                --model_path "$MODEL_PATH" \
                --max_sentences 10 \
                --epsilon 0.0
        else
            echo "[EVALUATION] Validation file not found: $VAL_FILE"
        fi
        echo "========================================"
    fi
    
    # Save checkpoint (ISOLATED CHECKPOINT DIR)
    if [ $((SENTENCE_NUM % CHECKPOINT_INTERVAL)) -eq 0 ]; then
        echo "[FILTERED CHECKPOINT] Saving at sentence $SENTENCE_NUM..."
        CHECKPOINT_MODEL="$CHECKPOINT_DIR/agent_model_sentence_${SENTENCE_NUM}.pth"
        CHECKPOINT_BUFFER="$CHECKPOINT_DIR/replay_buffer_sentence_${SENTENCE_NUM}.pkl"
        cp "$MODEL_PATH" "$CHECKPOINT_MODEL"
        if [ -f "$BUFFER_PATH" ]; then
            cp "$BUFFER_PATH" "$CHECKPOINT_BUFFER"
            echo "[CHECKPOINT] Replay buffer saved to: $CHECKPOINT_BUFFER"
        fi
        echo "[CHECKPOINT] Model saved to: $CHECKPOINT_MODEL"
    fi
    
    echo "[FILTERED] Completed sentence $SENTENCE_NUM"

done < "$SENTENCES_FILE"

echo ""
echo "============================================================================"
echo "[FILTERED PIPELINE] Completed - Processed $TOTAL_SENTENCES sentences"
echo "============================================================================"
