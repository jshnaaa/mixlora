"""
Inference and evaluation script for trained MixLoRA models on choice question datasets.
"""

import argparse
import json
import logging
import re
import sys
import os
from typing import Dict, List, Optional, Tuple
from collections import Counter

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from tqdm import tqdm
import numpy as np

# Add parent directory to path to import mixlora
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mixlora import MixLoraModelForCausalLM
from dataset import ChoiceQuestionDataset, ChoiceQuestionCollator


class ChoiceQuestionInference:
    """Inference class for choice question datasets using MixLoRA models."""

    def __init__(
        self,
        model_path: str,
        device: Optional[str] = None,
        batch_size: int = 8,
        max_new_tokens: int = 10,
        temperature: float = 0.1,
        do_sample: bool = False
    ):
        """
        Initialize the inference engine.

        Args:
            model_path: Path to the trained MixLoRA model
            device: Device to run inference on
            batch_size: Batch size for inference
            max_new_tokens: Maximum new tokens to generate
            temperature: Sampling temperature
            do_sample: Whether to use sampling
        """
        self.model_path = model_path
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size = batch_size
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.do_sample = do_sample

        self.logger = logging.getLogger(__name__)
        logging.basicConfig(level=logging.INFO)

        # Load model and tokenizer
        self._load_model()

    def _load_model(self):
        """Load the trained MixLoRA model and tokenizer."""
        self.logger.info(f"Loading model from {self.model_path}")

        # Load model and config
        self.model, self.config = MixLoraModelForCausalLM.from_pretrained(
            self.model_path,
            torch_dtype=torch.float16,
            device_map=self.device
        )

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.base_model_name_or_path)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model.eval()
        self.logger.info("Model loaded successfully")

    def _extract_choice_from_generation(self, generated_text: str, choice_range: List[str]) -> Optional[str]:
        """
        Extract the choice answer from generated text.

        Args:
            generated_text: The generated text from the model
            choice_range: List of valid choices

        Returns:
            Extracted choice or None if not found
        """
        # Clean the generated text
        generated_text = generated_text.strip()

        # First, try to find an exact match with any of the choices
        for choice in choice_range:
            if generated_text == choice:
                return choice

        # Try to find the choice at the beginning of the text
        for choice in choice_range:
            if generated_text.startswith(choice):
                return choice

        # Try to find any choice in the text (prefer earlier occurrences)
        found_choices = []
        for choice in choice_range:
            if choice in generated_text:
                pos = generated_text.find(choice)
                found_choices.append((pos, choice))

        if found_choices:
            # Return the choice that appears first
            found_choices.sort(key=lambda x: x[0])
            return found_choices[0][1]

        # Try regex patterns for numbers
        if all(choice.isdigit() for choice in choice_range):
            # Look for any digit that's in our choice range
            digits = re.findall(r'\d+', generated_text)
            for digit in digits:
                if digit in choice_range:
                    return digit

        # If nothing found, return None
        return None

    def predict_single(self, instruction: str, input_text: str = "") -> Tuple[str, str]:
        """
        Predict a single example.

        Args:
            instruction: The instruction text
            input_text: Additional input text (usually empty)

        Returns:
            Tuple of (generated_text, extracted_choice)
        """
        # Combine instruction and input
        prompt = instruction
        if input_text.strip():
            prompt += input_text

        # Tokenize
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=512
        ).to(self.device)

        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature,
                do_sample=self.do_sample,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

        # Decode
        generated_text = self.tokenizer.decode(
            outputs[0][inputs['input_ids'].shape[1]:],
            skip_special_tokens=True
        ).strip()

        return generated_text, generated_text

    def evaluate_dataset(
        self,
        dataset_path: str,
        choice_range: Optional[List[str]] = None,
        save_predictions: bool = True,
        output_file: Optional[str] = None
    ) -> Dict[str, float]:
        """
        Evaluate the model on a dataset.

        Args:
            dataset_path: Path to the evaluation dataset
            choice_range: Valid choice range (auto-detected if None)
            save_predictions: Whether to save predictions
            output_file: Output file for predictions

        Returns:
            Dictionary with evaluation metrics
        """
        self.logger.info(f"Evaluating on dataset: {dataset_path}")

        # Load dataset
        dataset = ChoiceQuestionDataset(
            data_path=dataset_path,
            tokenizer=self.tokenizer,
            max_length=512,
            choice_range=choice_range
        )

        if choice_range is None:
            choice_range = dataset.choice_range

        # Setup data loader
        collator = ChoiceQuestionCollator(tokenizer=self.tokenizer)
        dataloader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=collator
        )

        # Run evaluation
        all_predictions = []
        all_targets = []
        all_generated = []
        correct = 0
        total = 0

        self.model.eval()
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Evaluating"):
                # Get the original instructions for generation
                batch_instructions = []
                for i in range(len(batch['choice_answers'])):
                    # Reconstruct the instruction from input_ids
                    input_ids = batch['input_ids'][i]
                    # Find where the target starts (where labels are not -100)
                    labels = batch['labels'][i]
                    instruction_length = (labels == -100).sum().item()
                    instruction_ids = input_ids[:instruction_length]
                    instruction = self.tokenizer.decode(instruction_ids, skip_special_tokens=True)
                    batch_instructions.append(instruction)

                # Generate predictions for each sample in batch
                for i, instruction in enumerate(batch_instructions):
                    generated_text, _ = self.predict_single(instruction)
                    predicted_choice = self._extract_choice_from_generation(generated_text, choice_range)

                    target_choice = batch['choice_answers'][i]

                    all_predictions.append(predicted_choice)
                    all_targets.append(target_choice)
                    all_generated.append(generated_text)

                    if predicted_choice == target_choice:
                        correct += 1
                    total += 1

        # Calculate metrics
        accuracy = correct / total if total > 0 else 0.0

        # Calculate per-choice metrics
        choice_metrics = {}
        for choice in choice_range:
            choice_correct = sum(1 for p, t in zip(all_predictions, all_targets)
                               if t == choice and p == choice)
            choice_total = sum(1 for t in all_targets if t == choice)
            choice_predicted = sum(1 for p in all_predictions if p == choice)

            precision = choice_correct / choice_predicted if choice_predicted > 0 else 0.0
            recall = choice_correct / choice_total if choice_total > 0 else 0.0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

            choice_metrics[choice] = {
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'support': choice_total
            }

        # Calculate macro averages
        macro_precision = np.mean([m['precision'] for m in choice_metrics.values()])
        macro_recall = np.mean([m['recall'] for m in choice_metrics.values()])
        macro_f1 = np.mean([m['f1'] for m in choice_metrics.values()])

        metrics = {
            'accuracy': accuracy,
            'macro_precision': macro_precision,
            'macro_recall': macro_recall,
            'macro_f1': macro_f1,
            'total_samples': total,
            'correct_predictions': correct,
            'choice_metrics': choice_metrics
        }

        # Print results
        self.logger.info(f"Evaluation Results:")
        self.logger.info(f"Accuracy: {accuracy:.4f}")
        self.logger.info(f"Macro F1: {macro_f1:.4f}")
        self.logger.info(f"Total samples: {total}")

        # Save predictions if requested
        if save_predictions:
            if output_file is None:
                output_file = f"predictions_{os.path.basename(dataset_path)}.json"

            predictions_data = []
            for i in range(len(all_targets)):
                # Get original data
                original_data = dataset.data[i]
                predictions_data.append({
                    'instruction': original_data['instruction'],
                    'input': original_data.get('input', ''),
                    'target': all_targets[i],
                    'predicted': all_predictions[i],
                    'generated_text': all_generated[i],
                    'correct': all_predictions[i] == all_targets[i]
                })

            # Save predictions and metrics
            output_data = {
                'metrics': metrics,
                'predictions': predictions_data
            }

            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(output_data, f, indent=2, ensure_ascii=False)

            self.logger.info(f"Predictions saved to {output_file}")

        return metrics

    def interactive_inference(self):
        """Run interactive inference mode."""
        self.logger.info("Starting interactive inference mode. Type 'quit' to exit.")

        while True:
            try:
                instruction = input("\nEnter instruction: ").strip()
                if instruction.lower() in ['quit', 'exit', 'q']:
                    break

                input_text = input("Enter input (optional): ").strip()

                generated_text, extracted_choice = self.predict_single(instruction, input_text)

                print(f"Generated: {generated_text}")
                print(f"Extracted choice: {extracted_choice}")

            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"Error: {e}")

        self.logger.info("Interactive mode ended.")


def main():
    parser = argparse.ArgumentParser(description="Inference with trained MixLoRA model")

    parser.add_argument("--model_path", type=str, required=True, help="Path to trained model")
    parser.add_argument("--dataset_path", type=str, help="Path to evaluation dataset")
    parser.add_argument("--output_file", type=str, help="Output file for predictions")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for inference")
    parser.add_argument("--max_new_tokens", type=int, default=10, help="Maximum new tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.1, help="Sampling temperature")
    parser.add_argument("--do_sample", action="store_true", help="Use sampling instead of greedy decoding")
    parser.add_argument("--interactive", action="store_true", help="Run in interactive mode")
    parser.add_argument("--device", type=str, help="Device to run inference on")

    args = parser.parse_args()

    # Create inference engine
    inference_engine = ChoiceQuestionInference(
        model_path=args.model_path,
        device=args.device,
        batch_size=args.batch_size,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        do_sample=args.do_sample
    )

    if args.interactive:
        # Run interactive mode
        inference_engine.interactive_inference()
    elif args.dataset_path:
        # Evaluate on dataset
        metrics = inference_engine.evaluate_dataset(
            dataset_path=args.dataset_path,
            output_file=args.output_file
        )
        print(f"Final Accuracy: {metrics['accuracy']:.4f}")
    else:
        print("Please specify either --dataset_path for evaluation or --interactive for interactive mode")


if __name__ == "__main__":
    main()