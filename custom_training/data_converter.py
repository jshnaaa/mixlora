"""
Data format converter for choice question datasets.
This script helps convert various data formats to the required format for MixLoRA training.
"""

import argparse
import json
import logging
from typing import Dict, List, Any, Optional
import os


class DataConverter:
    """Converter for choice question datasets."""

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        logging.basicConfig(level=logging.INFO)

    def convert_from_standard_format(
        self,
        input_file: str,
        output_file: str,
        instruction_field: str = "question",
        answer_field: str = "answer",
        choices_field: Optional[str] = None,
        instruction_template: str = "### Question: {question}\n### Answer: "
    ):
        """
        Convert from standard QA format to MixLoRA format.

        Args:
            input_file: Input JSON file path
            output_file: Output JSON file path
            instruction_field: Field name containing the question
            answer_field: Field name containing the answer
            choices_field: Field name containing choices (optional)
            instruction_template: Template for formatting instructions
        """
        self.logger.info(f"Converting {input_file} to MixLoRA format...")

        with open(input_file, 'r', encoding='utf-8') as f:
            if input_file.endswith('.jsonl'):
                data = [json.loads(line) for line in f]
            else:
                data = json.load(f)

        converted_data = []
        for item in data:
            # Extract question and answer
            question = item.get(instruction_field, "")
            answer = item.get(answer_field, "")

            # Format instruction
            if choices_field and choices_field in item:
                choices = item[choices_field]
                if isinstance(choices, list):
                    choices_text = " ".join([f"{i+1}. {choice}" for i, choice in enumerate(choices)])
                    question = f"{question} {choices_text}"

            instruction = instruction_template.format(question=question)

            # Create converted item
            converted_item = {
                "instruction": instruction,
                "instruction_mask": instruction,  # You can modify this for masking if needed
                "input": "",
                "output": str(answer),
                "label": str(answer)
            }

            converted_data.append(converted_item)

        # Save converted data
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(converted_data, f, indent=2, ensure_ascii=False)

        self.logger.info(f"Converted {len(converted_data)} items. Saved to {output_file}")

    def convert_from_multiple_choice(
        self,
        input_file: str,
        output_file: str,
        question_field: str = "question",
        choices_field: str = "choices",
        answer_field: str = "answer",
        answer_type: str = "index"  # "index" or "letter" or "text"
    ):
        """
        Convert from multiple choice format.

        Args:
            input_file: Input JSON file path
            output_file: Output JSON file path
            question_field: Field containing the question
            choices_field: Field containing the choices list
            answer_field: Field containing the correct answer
            answer_type: Type of answer ("index", "letter", "text")
        """
        self.logger.info(f"Converting multiple choice dataset from {input_file}...")

        with open(input_file, 'r', encoding='utf-8') as f:
            if input_file.endswith('.jsonl'):
                data = [json.loads(line) for line in f]
            else:
                data = json.load(f)

        converted_data = []
        for item in data:
            question = item.get(question_field, "")
            choices = item.get(choices_field, [])
            answer = item.get(answer_field, "")

            # Format choices
            choices_text = ""
            for i, choice in enumerate(choices):
                choices_text += f" {i+1}. {choice}"

            # Create instruction
            instruction = f"### Question: Give me the answer from 1 to {len(choices)}: {question}{choices_text}. You can only choose one option.\n### Answer: "

            # Convert answer to index format
            if answer_type == "index":
                # Answer is already an index (0-based or 1-based)
                if isinstance(answer, int):
                    if answer >= 1:  # 1-based
                        output_answer = str(answer)
                    else:  # 0-based
                        output_answer = str(answer + 1)
                else:
                    output_answer = str(answer)
            elif answer_type == "letter":
                # Answer is a letter (A, B, C, D)
                if isinstance(answer, str) and len(answer) == 1:
                    output_answer = str(ord(answer.upper()) - ord('A') + 1)
                else:
                    output_answer = "1"  # Default fallback
            elif answer_type == "text":
                # Answer is the text of the choice
                try:
                    choice_index = choices.index(answer) + 1
                    output_answer = str(choice_index)
                except ValueError:
                    # If exact match not found, try partial match
                    output_answer = "1"  # Default fallback
                    for i, choice in enumerate(choices):
                        if answer.lower() in choice.lower() or choice.lower() in answer.lower():
                            output_answer = str(i + 1)
                            break

            converted_item = {
                "instruction": instruction,
                "instruction_mask": instruction,
                "input": "",
                "output": output_answer,
                "label": output_answer
            }

            converted_data.append(converted_item)

        # Save converted data
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(converted_data, f, indent=2, ensure_ascii=False)

        self.logger.info(f"Converted {len(converted_data)} items. Saved to {output_file}")

    def validate_dataset(self, dataset_file: str):
        """
        Validate the dataset format.

        Args:
            dataset_file: Path to the dataset file
        """
        self.logger.info(f"Validating dataset: {dataset_file}")

        with open(dataset_file, 'r', encoding='utf-8') as f:
            if dataset_file.endswith('.jsonl'):
                data = [json.loads(line) for line in f]
            else:
                data = json.load(f)

        required_fields = ["instruction", "input", "output"]
        issues = []

        for i, item in enumerate(data):
            # Check required fields
            for field in required_fields:
                if field not in item:
                    issues.append(f"Item {i}: Missing field '{field}'")

            # Check if instruction contains actual content
            if "instruction" in item and not item["instruction"].strip():
                issues.append(f"Item {i}: Empty instruction")

            # Check if output is valid
            if "output" in item and not str(item["output"]).strip():
                issues.append(f"Item {i}: Empty output")

        # Check choice range consistency
        outputs = [item.get("output", "") for item in data if item.get("output", "")]
        unique_outputs = set(str(o) for o in outputs)

        self.logger.info(f"Dataset validation results:")
        self.logger.info(f"Total items: {len(data)}")
        self.logger.info(f"Unique outputs: {sorted(unique_outputs)}")

        if issues:
            self.logger.warning(f"Found {len(issues)} issues:")
            for issue in issues[:10]:  # Show first 10 issues
                self.logger.warning(f"  {issue}")
            if len(issues) > 10:
                self.logger.warning(f"  ... and {len(issues) - 10} more issues")
        else:
            self.logger.info("Dataset validation passed!")

    def split_dataset(
        self,
        input_file: str,
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
        test_ratio: float = 0.1,
        random_seed: int = 42
    ):
        """
        Split dataset into train/validation/test sets.

        Args:
            input_file: Input dataset file
            train_ratio: Ratio for training set
            val_ratio: Ratio for validation set
            test_ratio: Ratio for test set
            random_seed: Random seed for reproducibility
        """
        import random

        self.logger.info(f"Splitting dataset: {input_file}")

        with open(input_file, 'r', encoding='utf-8') as f:
            if input_file.endswith('.jsonl'):
                data = [json.loads(line) for line in f]
            else:
                data = json.load(f)

        # Shuffle data
        random.seed(random_seed)
        random.shuffle(data)

        # Calculate split sizes
        total_size = len(data)
        train_size = int(total_size * train_ratio)
        val_size = int(total_size * val_ratio)

        # Split data
        train_data = data[:train_size]
        val_data = data[train_size:train_size + val_size]
        test_data = data[train_size + val_size:]

        # Save splits
        base_name = os.path.splitext(input_file)[0]

        train_file = f"{base_name}_train.json"
        val_file = f"{base_name}_val.json"
        test_file = f"{base_name}_test.json"

        with open(train_file, 'w', encoding='utf-8') as f:
            json.dump(train_data, f, indent=2, ensure_ascii=False)

        with open(val_file, 'w', encoding='utf-8') as f:
            json.dump(val_data, f, indent=2, ensure_ascii=False)

        with open(test_file, 'w', encoding='utf-8') as f:
            json.dump(test_data, f, indent=2, ensure_ascii=False)

        self.logger.info(f"Dataset split completed:")
        self.logger.info(f"  Train: {len(train_data)} items -> {train_file}")
        self.logger.info(f"  Validation: {len(val_data)} items -> {val_file}")
        self.logger.info(f"  Test: {len(test_data)} items -> {test_file}")

    def create_sample_dataset(self, output_file: str, num_samples: int = 100):
        """
        Create a sample dataset for testing.

        Args:
            output_file: Output file path
            num_samples: Number of sample items to create
        """
        import random

        self.logger.info(f"Creating sample dataset with {num_samples} items...")

        sample_questions = [
            "What is the capital of France?",
            "Which planet is closest to the sun?",
            "What is 2 + 2?",
            "Who wrote Romeo and Juliet?",
            "What is the largest ocean on Earth?"
        ]

        sample_choices = [
            ["London", "Berlin", "Paris", "Madrid"],
            ["Venus", "Mercury", "Earth", "Mars"],
            ["3", "4", "5", "6"],
            ["Charles Dickens", "William Shakespeare", "Jane Austen", "Mark Twain"],
            ["Atlantic", "Indian", "Arctic", "Pacific"]
        ]

        sample_answers = ["3", "2", "2", "2", "4"]

        sample_data = []
        for i in range(num_samples):
            idx = i % len(sample_questions)
            question = sample_questions[idx]
            choices = sample_choices[idx]
            answer = sample_answers[idx]

            choices_text = " ".join([f"{j+1}. {choice}" for j, choice in enumerate(choices)])
            instruction = f"### Question: Give me the answer from 1 to 4: {question} {choices_text}. You can only choose one option.\n### Answer: "

            item = {
                "instruction": instruction,
                "instruction_mask": instruction,
                "input": "",
                "output": answer,
                "label": answer
            }

            sample_data.append(item)

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(sample_data, f, indent=2, ensure_ascii=False)

        self.logger.info(f"Sample dataset created: {output_file}")


def main():
    parser = argparse.ArgumentParser(description="Convert data formats for MixLoRA training")

    parser.add_argument("command", choices=["convert", "validate", "split", "sample"],
                       help="Command to execute")
    parser.add_argument("--input_file", type=str, help="Input file path")
    parser.add_argument("--output_file", type=str, help="Output file path")
    parser.add_argument("--format", type=str, choices=["standard", "multiple_choice"],
                       default="standard", help="Input format type")

    # Format-specific arguments
    parser.add_argument("--question_field", type=str, default="question",
                       help="Field name for question")
    parser.add_argument("--answer_field", type=str, default="answer",
                       help="Field name for answer")
    parser.add_argument("--choices_field", type=str, default="choices",
                       help="Field name for choices")
    parser.add_argument("--answer_type", type=str, choices=["index", "letter", "text"],
                       default="index", help="Type of answer format")

    # Split arguments
    parser.add_argument("--train_ratio", type=float, default=0.8,
                       help="Training set ratio")
    parser.add_argument("--val_ratio", type=float, default=0.1,
                       help="Validation set ratio")
    parser.add_argument("--test_ratio", type=float, default=0.1,
                       help="Test set ratio")
    parser.add_argument("--random_seed", type=int, default=42,
                       help="Random seed for splitting")

    # Sample arguments
    parser.add_argument("--num_samples", type=int, default=100,
                       help="Number of sample items to create")

    args = parser.parse_args()

    converter = DataConverter()

    if args.command == "convert":
        if not args.input_file or not args.output_file:
            print("Error: --input_file and --output_file are required for convert command")
            return

        if args.format == "standard":
            converter.convert_from_standard_format(
                input_file=args.input_file,
                output_file=args.output_file,
                instruction_field=args.question_field,
                answer_field=args.answer_field,
                choices_field=args.choices_field
            )
        elif args.format == "multiple_choice":
            converter.convert_from_multiple_choice(
                input_file=args.input_file,
                output_file=args.output_file,
                question_field=args.question_field,
                choices_field=args.choices_field,
                answer_field=args.answer_field,
                answer_type=args.answer_type
            )

    elif args.command == "validate":
        if not args.input_file:
            print("Error: --input_file is required for validate command")
            return
        converter.validate_dataset(args.input_file)

    elif args.command == "split":
        if not args.input_file:
            print("Error: --input_file is required for split command")
            return
        converter.split_dataset(
            input_file=args.input_file,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            test_ratio=args.test_ratio,
            random_seed=args.random_seed
        )

    elif args.command == "sample":
        if not args.output_file:
            print("Error: --output_file is required for sample command")
            return
        converter.create_sample_dataset(
            output_file=args.output_file,
            num_samples=args.num_samples
        )


if __name__ == "__main__":
    main()