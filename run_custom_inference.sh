#!/bin/bash

# Custom inference script for trained MixLoRA models on cultural datasets
# Supports loading adapter weights and evaluating on external test sets
# Usage: ./run_custom_inference.sh [OPTIONS]

# Default model configuration
BACKBONE="auto"  # Will be auto-detected from adapter path
BASE_MODEL=""    # Will be set based on backbone
ADAPTER_PATH=""  # Will be set below or provided as argument

# Inference parameters
BATCH_SIZE=8
MAX_NEW_TOKENS=10
TEMPERATURE=0.1

# Function to show usage
show_usage() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  --adapter_path PATH     Path to trained adapter weights (required, or 'auto' for latest)"
    echo "  --backbone TYPE         Model backbone (llama, qwen, or 'auto' for detection, default: auto)"
    echo "  --dataset_path PATH     Path to test dataset for evaluation"
    echo "  --output_file PATH      Output file for predictions"
    echo "  --interactive           Run interactive inference mode"
    echo "  --batch_size N          Batch size for inference (default: 8)"
    echo "  --max_new_tokens N      Maximum new tokens to generate (default: 10)"
    echo "  --temperature F         Sampling temperature (default: 0.1)"
    echo ""
    echo "Examples:"
    echo "  # Evaluate on external test dataset (auto-detect backbone)"
    echo "  $0 --adapter_path /path/to/adapter --dataset_path /path/to/test.json"
    echo ""
    echo "  # Specify backbone explicitly"
    echo "  $0 --adapter_path /path/to/adapter --backbone qwen --dataset_path /path/to/test.json"
    echo ""
    echo "  # Interactive mode with latest model"
    echo "  $0 --adapter_path auto --interactive"
    echo ""
    echo "  # Find latest qwen model automatically"
    echo "  $0 --adapter_path auto --backbone qwen --dataset_path /path/to/test.json"
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --adapter_path)
            ADAPTER_PATH="$2"
            shift 2
            ;;
        --backbone)
            BACKBONE="$2"
            shift 2
            ;;
        --dataset_path)
            DATASET_PATH="$2"
            shift 2
            ;;
        --output_file)
            OUTPUT_FILE="$2"
            shift 2
            ;;
        --interactive)
            INTERACTIVE=true
            shift
            ;;
        --batch_size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --max_new_tokens)
            MAX_NEW_TOKENS="$2"
            shift 2
            ;;
        --temperature)
            TEMPERATURE="$2"
            shift 2
            ;;
        -h|--help)
            show_usage
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            show_usage
            exit 1
            ;;
    esac
done

# Auto-detect latest trained model if requested
if [ "$ADAPTER_PATH" = "auto" ]; then
    echo "Auto-detecting latest trained model..."

    # Build search pattern based on backbone
    if [ "$BACKBONE" = "auto" ]; then
        SEARCH_PATTERN="*_*_*"  # Match any backbone
    else
        SEARCH_PATTERN="*_${BACKBONE}_*"  # Match specific backbone
    fi

    LATEST_MODEL=$(find /root/autodl-fs/data/mixlora -name "$SEARCH_PATTERN" -type d | sort | tail -1)
    if [ -n "$LATEST_MODEL" ] && [ -d "$LATEST_MODEL/best_model" ]; then
        ADAPTER_PATH="$LATEST_MODEL/best_model"
        echo "Found latest model: $ADAPTER_PATH"

        # Auto-detect backbone from path if not specified
        if [ "$BACKBONE" = "auto" ]; then
            if [[ "$LATEST_MODEL" == *"_llama_"* ]]; then
                BACKBONE="llama"
            elif [[ "$LATEST_MODEL" == *"_qwen_"* ]]; then
                BACKBONE="qwen"
            else
                echo "Warning: Could not auto-detect backbone from path, defaulting to llama"
                BACKBONE="llama"
            fi
            echo "Auto-detected backbone: $BACKBONE"
        fi
    else
        echo "Error: No trained models found in /root/autodl-fs/data/mixlora"
        echo "Search pattern: $SEARCH_PATTERN"
        echo "Please specify --adapter_path manually"
        exit 1
    fi
fi

# Auto-detect backbone from adapter path if still auto
if [ "$BACKBONE" = "auto" ] && [ -n "$ADAPTER_PATH" ]; then
    if [[ "$ADAPTER_PATH" == *"_llama_"* ]]; then
        BACKBONE="llama"
    elif [[ "$ADAPTER_PATH" == *"_qwen_"* ]]; then
        BACKBONE="qwen"
    else
        echo "Warning: Could not auto-detect backbone from adapter path, defaulting to llama"
        BACKBONE="llama"
    fi
    echo "Auto-detected backbone: $BACKBONE"
fi

# Set base model path based on backbone
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

# Check required parameters
if [ -z "$ADAPTER_PATH" ]; then
    echo "Error: --adapter_path is required"
    show_usage
    exit 1
fi

# Check if adapter path exists
if [ ! -d "$ADAPTER_PATH" ]; then
    echo "Error: Adapter path not found: $ADAPTER_PATH"
    exit 1
fi

# Check if adapter files exist
if [ ! -f "$ADAPTER_PATH/adapter_config.json" ] || [ ! -f "$ADAPTER_PATH/adapter_model.bin" ]; then
    echo "Error: Required adapter files not found in $ADAPTER_PATH"
    echo "Expected files: adapter_config.json, adapter_model.bin"
    exit 1
fi

# Check if base model exists
if [ ! -d "$BASE_MODEL" ]; then
    echo "Error: Base model directory not found at $BASE_MODEL"
    echo "Please update the BASE_MODEL variable"
    exit 1
fi

echo "Starting MixLoRA inference..."
echo "Backbone: $BACKBONE"
echo "Base model: $BASE_MODEL"
echo "Adapter path: $ADAPTER_PATH"

# Build command
CMD="python custom_training/inference_custom.py \
    --base_model_path \"$BASE_MODEL\" \
    --adapter_path \"$ADAPTER_PATH\" \
    --batch_size $BATCH_SIZE \
    --max_new_tokens $MAX_NEW_TOKENS \
    --temperature $TEMPERATURE"

# Add mode-specific parameters
if [ "$INTERACTIVE" = true ]; then
    echo "Running in interactive mode..."
    CMD="$CMD --interactive"
elif [ -n "$DATASET_PATH" ]; then
    echo "Evaluating on dataset: $DATASET_PATH"

    # Check if dataset exists
    if [ ! -f "$DATASET_PATH" ]; then
        echo "Error: Dataset file not found: $DATASET_PATH"
        exit 1
    fi

    CMD="$CMD --dataset_path \"$DATASET_PATH\""

    if [ -n "$OUTPUT_FILE" ]; then
        CMD="$CMD --output_file \"$OUTPUT_FILE\""
        echo "Results will be saved to: $OUTPUT_FILE"
    else
        OUTPUT_FILE="predictions_$(basename $DATASET_PATH .json)_$(date +%Y%m%d_%H%M%S).json"
        CMD="$CMD --output_file \"$OUTPUT_FILE\""
        echo "Results will be saved to: $OUTPUT_FILE"
    fi
else
    echo "Error: Either --interactive or --dataset_path must be specified"
    show_usage
    exit 1
fi

# Execute the command
eval $CMD

if [ $? -eq 0 ]; then
    echo "Inference completed successfully!"
    if [ -n "$OUTPUT_FILE" ] && [ -f "$OUTPUT_FILE" ]; then
        echo "Results saved to: $OUTPUT_FILE"
    fi
else
    echo "Inference failed!"
    exit 1
fi