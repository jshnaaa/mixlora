#!/bin/bash

# Custom training script for MixLoRA on cultural datasets
# Supports automatic dataset splitting, best model saving, and comprehensive evaluation
# Usage: ./run_custom_training.sh [BACKBONE] [DATA_ID] [NUM_GPU] [TRAINING_MODE]
# BACKBONE: llama (default) | qwen
# DATA_ID: 2 (culturalbench, default) | 3 (normad) | 0 (unified_small) | 1 (unified) | 4 (cultureLLM)
# NUM_GPU: 2 (dual-GPU, default) | 1 (single-GPU)
# TRAINING_MODE: full (full model training) | mixlora (default, MixLoRA training with auto LoRA path detection)

# Parse command line arguments
BACKBONE=${1:-"llama"}  # Default to llama
DATA_ID=${2:-2}         # Default to culturalbench
NUM_GPU=${3:-2}         # Default to dual-GPU
TRAINING_MODE=${4:-"mixlora"}  # Default to mixlora training

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
OUTPUT_DIR="/root/autodl-fs/data/mixlora/${DATASET_TAG}_${BACKBONE}_$(date +%Y%m%d_%H%M)"

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

# Training parameters (optimized for 48G GPU with BF16/FP32)
MAX_LENGTH=512

# Adjust batch size based on GPU configuration and training mode
if [ "$TRAINING_MODE" = "mixlora" ]; then
    # MixLoRA-only training: Can use larger batch sizes due to frozen parameters
    if [ "$GPU_CONFIG" = "single" ]; then
        BATCH_SIZE=8   # Larger batch size for MixLoRA-only training
        GRADIENT_ACCUMULATION_STEPS=8  # Total effective batch size = 8 * 8 = 64
        echo "Single-GPU (MixLoRA-only): batch_size=$BATCH_SIZE, grad_accum=$GRADIENT_ACCUMULATION_STEPS, effective_batch=64"
    else
        BATCH_SIZE=6   # Larger per device batch size for MixLoRA-only
        GRADIENT_ACCUMULATION_STEPS=6  # Total effective batch size = 6 * 2 * 6 = 72
        echo "Dual-GPU (MixLoRA-only): batch_size=$BATCH_SIZE, grad_accum=$GRADIENT_ACCUMULATION_STEPS, effective_batch=72"
    fi
else
    # Full model training: Use conservative batch sizes
    if [ "$GPU_CONFIG" = "single" ]; then
        BATCH_SIZE=4   # Much smaller batch size to avoid OOM
        GRADIENT_ACCUMULATION_STEPS=15  # Total effective batch size = 4 * 15 = 60
        echo "Single-GPU (Full Training): batch_size=$BATCH_SIZE, grad_accum=$GRADIENT_ACCUMULATION_STEPS, effective_batch=60"
    else
        BATCH_SIZE=2   # Very small per device batch size to avoid OOM
        GRADIENT_ACCUMULATION_STEPS=16  # Total effective batch size = 2 * 2 * 16 = 64
        echo "Dual-GPU (Full Training): batch_size=$BATCH_SIZE, grad_accum=$GRADIENT_ACCUMULATION_STEPS, effective_batch=64"
    fi
fi

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
echo "Backbone: $BACKBONE"
echo "Dataset ID: $DATA_ID ($DATASET_TAG)"
echo "Base model: $BASE_MODEL"
echo "Output directory: $OUTPUT_DIR"
echo "Run name: $RUN_NAME"

# Determine training mode and configure arguments
case $TRAINING_MODE in
    "full")
        echo "🎯 Full model training mode"
        MIXLORA_ARGS=""
        ;;
    "mixlora")
        echo "🎯 MixLoRA training mode with auto LoRA path detection"

        if [ -n "$LORA_WEIGHTS_PATH" ]; then
            echo "Using LoRA weights: $LORA_WEIGHTS_PATH"

            if [ -e "$LORA_WEIGHTS_PATH" ]; then
                MIXLORA_ARGS="--pretrained_lora_path \"$LORA_WEIGHTS_PATH\""
                echo "Will train all LoRA adapters and MixLoRA components (not freezing)"
            else
                echo "Warning: LoRA path not found, training from scratch without pretrained weights"
                MIXLORA_ARGS="--skip_lora_autodetect"
            fi
        else
            echo "Warning: No LoRA mapping found for backbone=$BACKBONE, data_id=$DATA_ID"
            echo "Training from scratch without pretrained weights"
            MIXLORA_ARGS="--skip_lora_autodetect"
        fi
        ;;
    *)
        echo "Error: Unsupported training mode '$TRAINING_MODE'. Supported values: full, mixlora"
        exit 1
        ;;
esac

# Check if base model exists
if [ ! -d "$BASE_MODEL" ]; then
    echo "Error: Base model directory not found at $BASE_MODEL"
    echo "Please update the BASE_MODEL variable with the correct path"
    exit 1
fi

# Run training with appropriate method based on GPU count
if [ "$GPU_CONFIG" = "single" ]; then
    echo "Running single-GPU training with memory optimizations..."
    python custom_training/train_mixlora_custom.py \
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
        --wandb_project "$WANDB_PROJECT" \
        --run_name "$RUN_NAME" \
        $MIXLORA_ARGS
else
    echo "Running multi-GPU training with torchrun (DDP instead of DataParallel)..."
    torchrun --nproc_per_node=2 --master_port=29500 custom_training/train_mixlora_custom.py \
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
        --wandb_project "$WANDB_PROJECT" \
        --run_name "$RUN_NAME" \
        $MIXLORA_ARGS
fi

echo "Training completed!"

# The script automatically:
# 1. Loads the dataset based on DATA_ID
# 2. Splits data into train/val/test (8:1:1)
# 3. Trains the model and saves best model based on validation accuracy
# 4. Evaluates the best model on test set
# 5. Saves all results to output directory

echo ""
echo "Usage examples:"
echo "  # Full model training:"
echo "  ./run_custom_training.sh                         # Train with llama on culturalbench (dual-GPU, full mode)"
echo "  ./run_custom_training.sh llama 2 2 full         # Train with llama on culturalbench (dual-GPU, full mode)"
echo "  ./run_custom_training.sh llama 2 1 full         # Train with llama on culturalbench (single-GPU, full mode)"
echo ""
echo "  # MixLoRA training with auto LoRA path detection:"
echo "  ./run_custom_training.sh llama 2 2 mixlora      # MixLoRA training with auto LoRA path (dual-GPU)"
echo "  ./run_custom_training.sh llama 2 1 mixlora      # MixLoRA training with auto LoRA path (single-GPU)"
echo "  ./run_custom_training.sh qwen 3 2 mixlora       # MixLoRA training with qwen on normad (dual-GPU)"
echo ""
echo "  # Other dataset examples:"
echo "  ./run_custom_training.sh qwen 2 2 full          # Train with qwen on culturalbench (dual-GPU, full mode)"
echo "  ./run_custom_training.sh llama 3 2 full         # Train with llama on normad (dual-GPU, full mode)"
echo "  ./run_custom_training.sh llama 0 1 mixlora      # Train with llama on unified_small (single-GPU, MixLoRA-only)"
echo "  ./run_custom_training.sh qwen 4 2 mixlora       # Train with qwen on cultureLLM (dual-GPU, MixLoRA-only)"
echo ""
echo "  # Available datasets (DATA_ID):"
echo "  #   0: unified_all_datasets_small"
echo "  #   1: unified_all_datasets"
echo "  #   2: CulturalBench (default)"
echo "  #   3: NormAD"
echo "  #   4: CultureLLM"
echo ""
echo "  # Training modes:"
echo "  #   full: Complete model training"
echo "  #   mixlora: MixLoRA training with auto LoRA path detection (default, trains all LoRA + MixLoRA components)"
echo ""
echo "Check the output directory for:"
echo "- best_model/: Best model adapter weights"
echo "- training_config.json: Training configuration"
echo "- validation_results.json: Validation results for each epoch"
echo "- generated_answers.json: Generated answers on validation set"
echo "- test_results.json: Final test set evaluation results"