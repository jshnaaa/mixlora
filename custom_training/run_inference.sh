#!/bin/bash

# Inference script for trained MixLoRA models on choice question datasets

# Model and data paths
MODEL_PATH="./mixlora_choice_model"  # Path to your trained model
DATASET_PATH="path/to/your/test_dataset.json"  # Path to test dataset
OUTPUT_FILE="predictions.json"  # Output file for predictions

# Inference parameters
BATCH_SIZE=8
MAX_NEW_TOKENS=10
TEMPERATURE=0.1

# Check if model exists
if [ ! -d "$MODEL_PATH" ]; then
    echo "Error: Model directory not found at $MODEL_PATH"
    echo "Please update the MODEL_PATH variable with the correct path to your trained model"
    exit 1
fi

# Check if dataset exists (for evaluation mode)
if [ "$1" == "eval" ] && [ ! -f "$DATASET_PATH" ]; then
    echo "Error: Dataset file not found at $DATASET_PATH"
    echo "Please update the DATASET_PATH variable with the correct path to your test dataset"
    exit 1
fi

# Run inference based on mode
if [ "$1" == "interactive" ]; then
    echo "Starting interactive inference mode..."
    python inference.py \
        --model_path "$MODEL_PATH" \
        --batch_size $BATCH_SIZE \
        --max_new_tokens $MAX_NEW_TOKENS \
        --temperature $TEMPERATURE \
        --interactive

elif [ "$1" == "eval" ]; then
    echo "Running evaluation on dataset..."
    python inference.py \
        --model_path "$MODEL_PATH" \
        --dataset_path "$DATASET_PATH" \
        --output_file "$OUTPUT_FILE" \
        --batch_size $BATCH_SIZE \
        --max_new_tokens $MAX_NEW_TOKENS \
        --temperature $TEMPERATURE
    echo "Evaluation completed. Results saved to $OUTPUT_FILE"

else
    echo "Usage: $0 [interactive|eval]"
    echo "  interactive: Run interactive inference mode"
    echo "  eval: Evaluate on test dataset"
    exit 1
fi