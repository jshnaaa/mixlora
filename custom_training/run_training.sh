#!/bin/bash

# Training script for MixLoRA on choice question datasets
# Modify the parameters below according to your setup

# Model and data paths
BASE_MODEL="meta-llama/Llama-2-7b-hf"  # Change this to your preferred base model
DATASET_PATH="path/to/your/train_dataset.json"  # Change this to your dataset path
VALIDATION_DATASET_PATH="path/to/your/val_dataset.json"  # Optional validation dataset
OUTPUT_DIR="./mixlora_choice_model"

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

# Validation and logging
EVAL_STEPS=100
SAVE_STEPS=500
LOGGING_STEPS=10

# Experiment tracking (optional)
WANDB_PROJECT="mixlora-choice-questions"
RUN_NAME="llama2-7b-choice-qa-$(date +%Y%m%d_%H%M%S)"

# Check if dataset exists
if [ ! -f "$DATASET_PATH" ]; then
    echo "Error: Dataset file not found at $DATASET_PATH"
    echo "Please update the DATASET_PATH variable with the correct path to your dataset"
    exit 1
fi

# Create output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

# Run training
python train_mixlora.py \
    --base_model "$BASE_MODEL" \
    --dataset_path "$DATASET_PATH" \
    --validation_dataset_path "$VALIDATION_DATASET_PATH" \
    --output_dir "$OUTPUT_DIR" \
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
    --eval_steps $EVAL_STEPS \
    --save_steps $SAVE_STEPS \
    --logging_steps $LOGGING_STEPS \
    --wandb_project "$WANDB_PROJECT" \
    --run_name "$RUN_NAME"

echo "Training completed. Model saved to $OUTPUT_DIR"