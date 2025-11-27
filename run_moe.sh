#!/bin/bash

# MoE training script with shared experts for cultural datasets
# Based on MixLoRA but adds shared experts that are always activated
# Usage: ./run_moe.sh [BACKBONE] [DATA_ID] [NUM_GPU] [USE_SHARED]
# BACKBONE: llama (default) | qwen
# DATA_ID: 2 (culturalbench, default) | 3 (normad) | 0 (unified_small) | 1 (unified) | 4 (cultureLLM)
# NUM_GPU: 2 (dual-GPU, default) | 1 (single-GPU)
# USE_SHARED: true (default, use shared expert) | false (no shared expert, same as mixlora)

# Parse command line arguments
BACKBONE=${1:-"llama"}  # Default to llama
DATA_ID=${2:-2}         # Default to culturalbench
NUM_GPU=${3:-2}         # Default to dual-GPU
USE_SHARED=${4:-"true"} # Default to use shared expert

# Model configuration based on backbone
case $BACKBONE in
    "llama")
        BASE_MODEL="/root/autodl-tmp/CultureMoE/Culture_Alignment/Meta-Llama-3.1-8B-Instruct"
        ;;
    "qwen")
        BASE_MODEL="/root/autodl-tmp/CultureMoE/Culture_Alignment/Meta-Qwen-2.5-7B-Instruct"
        ;;
    *)
        echo "Error: Unsupported backbone '$BACKBONE'. Supported values: llama, qwen"
        exit 1
        ;;
esac

# Dataset configuration
case $DATA_ID in
    0)
        # unified_all_datasets_small
        DATASET_TAG="unified_small"
        if [ "$BACKBONE" = "qwen" ]; then
            LORA_WEIGHTS_PATH="/root/autodl-tmp/CultureMoE/Culture_Alignment/ft/ft_lora_only_gen_unified_small_qwen_20251111_1530/best_lora"
        else
            LORA_WEIGHTS_PATH="/root/autodl-tmp/CultureMoE/Culture_Alignment/ft/ft_lora_only_gen_unified_small_llama_20251111_1530/best_lora"
        fi
        echo "Using unified_all_datasets_small"
        ;;
    1)
        # unified_all_datasets
        DATASET_TAG="unified"
        if [ "$BACKBONE" = "qwen" ]; then
            LORA_WEIGHTS_PATH="/root/autodl-tmp/CultureMoE/Culture_Alignment/ft/ft_lora_only_gen_unified_qwen_20251111_1530/best_lora"
        else
            LORA_WEIGHTS_PATH="/root/autodl-tmp/CultureMoE/Culture_Alignment/ft/ft_lora_only_gen_unified_llama_20251111_1530/best_lora"
        fi
        echo "Using unified_all_datasets"
        ;;
    2)
        # CulturalBench
        DATASET_TAG="culturalbench"
        if [ "$BACKBONE" = "qwen" ]; then
            LORA_WEIGHTS_PATH="/root/autodl-tmp/CultureMoE/Culture_Alignment/ft/ft_lora_only_gen_CulturalBench_qwen_20251112_1228/best_lora"
        else
            LORA_WEIGHTS_PATH="/root/autodl-tmp/CultureMoE/Culture_Alignment/ft/ft_lora_only_gen_CulturalBench_llama_20251112_1141/best_lora"
        fi
        echo "Using CulturalBench dataset"
        ;;
    3)
        # NormAD
        DATASET_TAG="normad"
        if [ "$BACKBONE" = "qwen" ]; then
            LORA_WEIGHTS_PATH="/root/autodl-tmp/CultureMoE/Culture_Alignment/ft/ft_lora_only_gen_normad_qwen_20251111_1204/best_lora"
        else
            LORA_WEIGHTS_PATH="/root/autodl-tmp/CultureMoE/Culture_Alignment/ft/ft_lora_only_gen_normad_llama_20251112_1335/best_lora"
        fi
        echo "Using NormAD dataset"
        ;;
    4)
        # CultureLLM
        DATASET_TAG="cultureLLM"
        if [ "$BACKBONE" = "qwen" ]; then
            LORA_WEIGHTS_PATH="/root/autodl-tmp/CultureMoE/Culture_Alignment/ft/ft_lora_only_gen_cultureLLM_qwen_20251111_1530/best_lora"
        else
            LORA_WEIGHTS_PATH="/root/autodl-tmp/CultureMoE/Culture_Alignment/ft/ft_lora_only_gen_cultureLLM_llama_20251111_1530/best_lora"
        fi
        echo "Using CultureLLM dataset"
        ;;
    *)
        echo "Error: Unsupported data_id '$DATA_ID'. Supported values: 0, 1, 2, 3, 4"
        exit 1
        ;;
esac

# GPU configuration
case $NUM_GPU in
    1)
        GPU_CONFIG="single"
        echo "Configured for single-GPU training"
        ;;
    2)
        GPU_CONFIG="dual"
        echo "Configured for dual-GPU training"
        ;;
    *)
        echo "Error: Unsupported NUM_GPU '$NUM_GPU'. Supported values: 1, 2"
        exit 1
        ;;
esac

# Define output directory
OUTPUT_DIR="/root/autodl-fs/data/moe/${DATASET_TAG}_${BACKBONE}_$(date +%Y%m%d_%H%M)"

# MoE parameters
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

# Training parameters (optimized for 48G GPU with BF16/FP32)
MAX_LENGTH=512

# Adjust batch size based on GPU configuration
if [ "$GPU_CONFIG" = "single" ]; then
    BATCH_SIZE=6   # Smaller batch size for shared expert (more parameters)
    GRADIENT_ACCUMULATION_STEPS=10  # Total effective batch size = 6 * 10 = 60
    echo "Single-GPU (MoE): batch_size=$BATCH_SIZE, grad_accum=$GRADIENT_ACCUMULATION_STEPS, effective_batch=60"
else
    BATCH_SIZE=4   # Smaller per device batch size for shared expert
    GRADIENT_ACCUMULATION_STEPS=8  # Total effective batch size = 4 * 2 * 8 = 64
    echo "Dual-GPU (MoE): batch_size=$BATCH_SIZE, grad_accum=$GRADIENT_ACCUMULATION_STEPS, effective_batch=64"
fi

LEARNING_RATE=1e-4
NUM_EPOCHS=3
WARMUP_RATIO=0.1
WEIGHT_DECAY=0.01

# Evaluation parameters
EVAL_INTERVAL=1  # Evaluate every epoch
SAVE_STEPS=500
LOGGING_STEPS=10

# Experiment tracking (optional) - disabled by default
WANDB_PROJECT=""  # Empty to disable wandb
RUN_NAME="moe-data${DATA_ID}-$(date +%Y%m%d_%H%M%S)"

# Configure wandb arguments
if [ -n "$WANDB_PROJECT" ]; then
    WANDB_ARGS="--wandb_project \"$WANDB_PROJECT\" --run_name \"$RUN_NAME\""
    echo "Wandb tracking enabled: $WANDB_PROJECT"
else
    WANDB_ARGS=""
    echo "Wandb tracking disabled"
fi

echo "Starting MoE training on cultural dataset..."
echo "Backbone: $BACKBONE"
echo "Dataset ID: $DATA_ID ($DATASET_TAG)"
echo "Base model: $BASE_MODEL"
echo "Output directory: $OUTPUT_DIR"
echo "Run name: $RUN_NAME"
echo "Use shared expert: $USE_SHARED"

# Configure MoE arguments
if [ -n "$LORA_WEIGHTS_PATH" ]; then
    echo "Using LoRA weights: $LORA_WEIGHTS_PATH"

    if [ -e "$LORA_WEIGHTS_PATH" ]; then
        MOE_ARGS="--pretrained_lora_path \"$LORA_WEIGHTS_PATH\" --train_moe_only --use_shared_expert $USE_SHARED"
        echo "Will freeze pretrained LoRA and only train MoE components (router + shared experts + routing experts)"
    else
        echo "Warning: LoRA path not found, will use auto-detection in Python"
        MOE_ARGS="--use_shared_expert $USE_SHARED"
    fi
else
    echo "Error: No LoRA mapping found for backbone=$BACKBONE, data_id=$DATA_ID"
    echo "Will use auto-detection in Python"
    MOE_ARGS="--use_shared_expert $USE_SHARED"
fi

# Check if base model exists
if [ ! -d "$BASE_MODEL" ]; then
    echo "Error: Base model directory not found at $BASE_MODEL"
    echo "Please update the BASE_MODEL variable with the correct path"
    exit 1
fi

# Run training with appropriate method based on GPU count
if [ "$GPU_CONFIG" = "single" ]; then
    echo "Running single-GPU MoE training with memory optimizations..."
    python custom_training/train_moe.py \
        --data_id $DATA_ID \
        --backbone "$BACKBONE" \
        --base_model "$BASE_MODEL" \
        --output_dir "$OUTPUT_DIR" \
        --num_gpu $NUM_GPU \
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
        $WANDB_ARGS \
        $MOE_ARGS
else
    echo "Running multi-GPU MoE training with torchrun (DDP)..."
    torchrun --nproc_per_node=2 --master_port=29501 custom_training/train_moe.py \
        --data_id $DATA_ID \
        --backbone "$BACKBONE" \
        --base_model "$BASE_MODEL" \
        --output_dir "$OUTPUT_DIR" \
        --num_gpu $NUM_GPU \
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
        $WANDB_ARGS \
        $MOE_ARGS
fi

echo "Training completed!"

echo ""
echo "Usage examples:"
echo "  # MoE training with shared expert:"
echo "  ./run_moe.sh                         # Train with llama on culturalbench (dual-GPU, shared expert)"
echo "  ./run_moe.sh llama 2 2 true         # Train with llama on culturalbench (dual-GPU, shared expert)"
echo "  ./run_moe.sh llama 2 1 true         # Train with llama on culturalbench (single-GPU, shared expert)"
echo ""
echo "  # MoE training without shared expert (same as MixLoRA):"
echo "  ./run_moe.sh llama 2 2 false        # Train without shared expert"
echo ""
echo "  # Other examples:"
echo "  ./run_moe.sh qwen 3 2 true          # Train with qwen on normad (dual-GPU, shared expert)"
echo "  ./run_moe.sh llama 0 1 true         # Train with llama on unified_small (single-GPU, shared expert)"
echo ""
echo "  # Available datasets (DATA_ID):"
echo "  #   0: unified_all_datasets_small"
echo "  #   1: unified_all_datasets"
echo "  #   2: CulturalBench (default)"
echo "  #   3: NormAD"
echo "  #   4: CultureLLM"
echo ""
echo "Check the output directory for:"
echo "- best_model/: Best model adapter weights"
echo "- training_config.json: Training configuration"
echo "- validation_results.json: Validation results for each epoch"
echo "- generated_answers.json: Generated answers on validation set"
echo "- test_results.json: Final test set evaluation results"