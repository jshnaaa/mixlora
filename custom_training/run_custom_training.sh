#!/bin/bash

# Custom training script for MixLoRA on cultural datasets
# Supports automatic dataset splitting, best model saving, and comprehensive evaluation

# Dataset configuration
DATA_ID=2  # 2=culturalbench, 3=normad
BASE_MODEL="/root/autodl-tmp/CultureMoE/Culture_Alignment/Meta-Llama-3.1-8B-Instruct"

# MixLoRA parameters
NUM_EXPERTS=8
TOP_K=2
ROUTING_STRATEGY="mixlora"
ROUTER_AUX_LOSS_COEF=0.01
ROUTER_INIT_RANGE=0.02
JITTER_NOISE=0.0

# LoRA parameters
LORA_R=8
LORA_ALPHA=16
LORA_DROPOUT=0.05

# Training parameters
MAX_LENGTH=512
BATCH_SIZE=4
GRADIENT_ACCUMULATION_STEPS=4
LEARNING_RATE=1e-4
NUM_EPOCHS=3
WARMUP_RATIO=0.1
WEIGHT_DECAY=0.01

# Evaluation parameters
EVAL_INTERVAL=1  # Evaluate every epoch
SAVE_STEPS=500
LOGGING_STEPS=10

# Experiment tracking (optional)
WANDB_PROJECT="mixlora-cultural-datasets"
RUN_NAME="mixlora-data${DATA_ID}-$(date +%Y%m%d_%H%M%S)"

echo "Starting MixLoRA training on cultural dataset..."
echo "Dataset ID: $DATA_ID"
echo "Base model: $BASE_MODEL"
echo "Run name: $RUN_NAME"

# Check if base model exists
if [ ! -d "$BASE_MODEL" ]; then
    echo "Error: Base model directory not found at $BASE_MODEL"
    echo "Please update the BASE_MODEL variable with the correct path"
    exit 1
fi

# Run training
python train_mixlora_custom.py \
    --data_id $DATA_ID \
    --base_model "$BASE_MODEL" \
    --num_experts $NUM_EXPERTS \
    --top_k $TOP_K \
    --routing_strategy "$ROUTING_STRATEGY" \
    --router_aux_loss_coef $ROUTER_AUX_LOSS_COEF \
    --router_init_range $ROUTER_INIT_RANGE \
    --jitter_noise $JITTER_NOISE \
    --lora_r $LORA_R \
    --lora_alpha $LORA_ALPHA \
    --lora_dropout $LORA_DROPOUT \
    --max_length $MAX_LENGTH \
    --batch_size $BATCH_SIZE \
    --gradient_accumulation_steps $GRADIENT_ACCUMULATION_STEPS \
    --learning_rate $LEARNING_RATE \
    --num_epochs $NUM_EPOCHS \
    --warmup_ratio $WARMUP_RATIO \
    --weight_decay $WEIGHT_DECAY \
    --eval_interval $EVAL_INTERVAL \
    --save_steps $SAVE_STEPS \
    --logging_steps $LOGGING_STEPS \
    --wandb_project "$WANDB_PROJECT" \
    --run_name "$RUN_NAME"

echo "Training completed!"

# The script automatically:
# 1. Loads the dataset based on DATA_ID
# 2. Splits data into train/val/test (8:1:1)
# 3. Trains the model and saves best model based on validation accuracy
# 4. Evaluates the best model on test set
# 5. Saves all results to output directory

echo "Check the output directory for:"
echo "- best_model/: Best model adapter weights"
echo "- training_config.json: Training configuration"
echo "- validation_results.json: Validation results for each epoch"
echo "- generated_answers.json: Generated answers on validation set"
echo "- test_results.json: Final test set evaluation results"