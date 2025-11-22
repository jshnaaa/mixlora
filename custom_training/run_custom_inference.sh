#!/bin/bash

# Custom inference script for trained MixLoRA models on cultural datasets
# Supports loading adapter weights and evaluating on external test sets

# Model paths (update these based on your training output)
BASE_MODEL="/root/autodl-tmp/CultureMoE/Culture_Alignment/Meta-Llama-3.1-8B-Instruct"
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
    echo "  --adapter_path PATH     Path to trained adapter weights (required)"
    echo "  --dataset_path PATH     Path to test dataset for evaluation"
    echo "  --output_file PATH      Output file for predictions"
    echo "  --interactive           Run interactive inference mode"
    echo "  --batch_size N          Batch size for inference (default: 8)"
    echo "  --max_new_tokens N      Maximum new tokens to generate (default: 10)"
    echo "  --temperature F         Sampling temperature (default: 0.1)"
    echo ""
    echo "Examples:"
    echo "  # Evaluate on external test dataset"
    echo "  $0 --adapter_path /path/to/adapter --dataset_path /path/to/test.json --output_file results.json"
    echo ""
    echo "  # Interactive mode"
    echo "  $0 --adapter_path /path/to/adapter --interactive"
    echo ""
    echo "  # Find latest trained model automatically"
    echo "  $0 --adapter_path auto --dataset_path /path/to/test.json"
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --adapter_path)
            ADAPTER_PATH="$2"
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
    LATEST_MODEL=$(find /root/autodl-fs/data/mixlora -name "*_llama_*" -type d | sort | tail -1)
    if [ -n "$LATEST_MODEL" ] && [ -d "$LATEST_MODEL/best_model" ]; then
        ADAPTER_PATH="$LATEST_MODEL/best_model"
        echo "Found latest model: $ADAPTER_PATH"
    else
        echo "Error: No trained models found in /root/autodl-fs/data/mixlora"
        echo "Please specify --adapter_path manually"
        exit 1
    fi
fi

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
echo "Base model: $BASE_MODEL"
echo "Adapter path: $ADAPTER_PATH"

# Build command
CMD="python inference_custom.py \
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