#!/usr/bin/env python3
"""
Example usage script for MixLoRA custom training on choice question datasets.
This script demonstrates how to use the custom training pipeline.
"""

import json
import os
import sys

# Add current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data_converter import DataConverter


def create_example_dataset():
    """Create an example dataset for demonstration."""
    print("Creating example dataset...")

    # Sample data in the required format
    example_data = [
        {
            "instruction": "### Question: Give me the answer from 1 to 4: What is the capital of France? 1. London 2. Berlin 3. Paris 4. Madrid. You can only choose one option.\n### Answer: ",
            "instruction_mask": "### Question: Give me the answer from 1 to 4: What is the capital of [MASK]? 1. London 2. Berlin 3. Paris 4. Madrid. You can only choose one option.\n### Answer: ",
            "input": "",
            "output": "3",
            "label": "3"
        },
        {
            "instruction": "### Question: Give me the answer from 1 to 4: Which planet is closest to the sun? 1. Venus 2. Mercury 3. Earth 4. Mars. You can only choose one option.\n### Answer: ",
            "instruction_mask": "### Question: Give me the answer from 1 to 4: Which planet is closest to the [MASK]? 1. Venus 2. Mercury 3. Earth 4. Mars. You can only choose one option.\n### Answer: ",
            "input": "",
            "output": "2",
            "label": "2"
        },
        {
            "instruction": "### Question: Give me the answer from 1 to 4: What is 2 + 2? 1. 3 2. 4 3. 5 4. 6. You can only choose one option.\n### Answer: ",
            "instruction_mask": "### Question: Give me the answer from 1 to 4: What is 2 + [MASK]? 1. 3 2. 4 3. 5 4. 6. You can only choose one option.\n### Answer: ",
            "input": "",
            "output": "2",
            "label": "2"
        },
        {
            "instruction": "### Question: Give me the answer from 1 to 4: Who wrote Romeo and Juliet? 1. Charles Dickens 2. William Shakespeare 3. Jane Austen 4. Mark Twain. You can only choose one option.\n### Answer: ",
            "instruction_mask": "### Question: Give me the answer from 1 to 4: Who wrote [MASK]? 1. Charles Dickens 2. William Shakespeare 3. Jane Austen 4. Mark Twain. You can only choose one option.\n### Answer: ",
            "input": "",
            "output": "2",
            "label": "2"
        },
        {
            "instruction": "### Question: Give me the answer from 1 to 4: What is the largest ocean on Earth? 1. Atlantic 2. Indian 3. Arctic 4. Pacific. You can only choose one option.\n### Answer: ",
            "instruction_mask": "### Question: Give me the answer from 1 to 4: What is the largest [MASK] on Earth? 1. Atlantic 2. Indian 3. Arctic 4. Pacific. You can only choose one option.\n### Answer: ",
            "input": "",
            "output": "4",
            "label": "4"
        }
    ]

    # Create larger dataset by repeating and modifying examples
    full_dataset = []
    for i in range(200):  # Create 200 examples
        base_example = example_data[i % len(example_data)]
        # Add some variation to make it more realistic
        example = base_example.copy()
        full_dataset.append(example)

    # Save the dataset
    with open('example_dataset.json', 'w', encoding='utf-8') as f:
        json.dump(full_dataset, f, indent=2, ensure_ascii=False)

    print(f"Created example dataset with {len(full_dataset)} samples: example_dataset.json")
    return 'example_dataset.json'


def demonstrate_data_conversion():
    """Demonstrate data format conversion."""
    print("\n=== Data Conversion Example ===")

    # Create a sample dataset in multiple choice format
    mc_data = [
        {
            "question": "What is the capital of France?",
            "choices": ["London", "Berlin", "Paris", "Madrid"],
            "answer": 2  # 0-based index
        },
        {
            "question": "Which planet is closest to the sun?",
            "choices": ["Venus", "Mercury", "Earth", "Mars"],
            "answer": 1  # 0-based index
        }
    ]

    with open('mc_dataset.json', 'w', encoding='utf-8') as f:
        json.dump(mc_data, f, indent=2)

    # Convert using the data converter
    converter = DataConverter()
    converter.convert_from_multiple_choice(
        input_file='mc_dataset.json',
        output_file='converted_dataset.json',
        question_field='question',
        choices_field='choices',
        answer_field='answer',
        answer_type='index'
    )

    print("Conversion completed. Check converted_dataset.json")


def demonstrate_validation():
    """Demonstrate dataset validation."""
    print("\n=== Dataset Validation Example ===")

    converter = DataConverter()
    converter.validate_dataset('example_dataset.json')


def demonstrate_splitting():
    """Demonstrate dataset splitting."""
    print("\n=== Dataset Splitting Example ===")

    converter = DataConverter()
    converter.split_dataset(
        input_file='example_dataset.json',
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15,
        random_seed=42
    )

    print("Dataset split completed. Check example_dataset_train.json, example_dataset_val.json, example_dataset_test.json")


def print_training_command():
    """Print the training command example."""
    print("\n=== Training Command Example ===")

    training_cmd = """
# Training with the example dataset
python train_mixlora.py \\
    --base_model meta-llama/Llama-2-7b-hf \\
    --dataset_path example_dataset_train.json \\
    --validation_dataset_path example_dataset_val.json \\
    --output_dir ./example_mixlora_model \\
    --num_experts 4 \\
    --top_k 2 \\
    --lora_r 8 \\
    --lora_alpha 16 \\
    --batch_size 2 \\
    --gradient_accumulation_steps 8 \\
    --num_epochs 3 \\
    --learning_rate 1e-4 \\
    --max_length 256 \\
    --eval_steps 50 \\
    --save_steps 100 \\
    --logging_steps 10
"""

    print("To train the model, run:")
    print(training_cmd)


def print_inference_command():
    """Print the inference command example."""
    print("\n=== Inference Command Example ===")

    inference_cmd = """
# Evaluate the trained model
python inference.py \\
    --model_path ./example_mixlora_model \\
    --dataset_path example_dataset_test.json \\
    --output_file test_predictions.json \\
    --batch_size 8 \\
    --max_new_tokens 5 \\
    --temperature 0.1

# Interactive inference
python inference.py \\
    --model_path ./example_mixlora_model \\
    --interactive
"""

    print("To run inference, use:")
    print(inference_cmd)


def main():
    """Main demonstration function."""
    print("=== MixLoRA Custom Training Example ===")
    print("This script demonstrates how to use the custom MixLoRA training pipeline.")

    # Create example dataset
    dataset_file = create_example_dataset()

    # Demonstrate data conversion
    demonstrate_data_conversion()

    # Demonstrate validation
    demonstrate_validation()

    # Demonstrate splitting
    demonstrate_splitting()

    # Print training and inference commands
    print_training_command()
    print_inference_command()

    print("\n=== Next Steps ===")
    print("1. Prepare your own dataset in the required format")
    print("2. Use data_converter.py to convert from other formats if needed")
    print("3. Split your dataset into train/val/test sets")
    print("4. Modify the training parameters in run_training.sh")
    print("5. Run the training script")
    print("6. Evaluate the trained model using the inference script")

    print("\n=== Files Created ===")
    print("- example_dataset.json: Example training dataset")
    print("- example_dataset_train.json: Training split")
    print("- example_dataset_val.json: Validation split")
    print("- example_dataset_test.json: Test split")
    print("- mc_dataset.json: Example multiple choice format")
    print("- converted_dataset.json: Converted from multiple choice format")

    print("\nExample setup completed!")


if __name__ == "__main__":
    main()