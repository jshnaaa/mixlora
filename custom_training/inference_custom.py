"""
Custom inference script for trained MixLoRA models on cultural datasets.
Supports loading adapter weights and evaluating on external test sets.
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
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report

# Add parent directory to path to import mixlora
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mixlora.config import MixLoraConfig
from mixlora.model import inject_adapter_in_model
from dataset import ChoiceQuestionDataset, ChoiceQuestionCollator


class CustomMixLoRAInference:
    """Inference class for cultural datasets using trained MixLoRA models."""

    def __init__(
        self,
        base_model_path: str,
        adapter_path: str,
        device: Optional[str] = None,
        batch_size: int = 8,
        max_new_tokens: int = 10,
        temperature: float = 0.1,
        do_sample: bool = False
    ):
        """
        Initialize the inference engine.

        Args:
            base_model_path: Path to the base model
            adapter_path: Path to the trained adapter weights
            device: Device to run inference on
            batch_size: Batch size for inference
            max_new_tokens: Maximum new tokens to generate
            temperature: Sampling temperature
            do_sample: Whether to use sampling
        """
        self.base_model_path = base_model_path
        self.adapter_path = adapter_path
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size = batch_size
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.do_sample = do_sample

        self.logger = logging.getLogger(__name__)
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

        # Load model and tokenizer
        self._load_model()

    def _load_model(self):
        """Load the base model and inject trained adapter weights."""
        self.logger.info(f"Loading base model from {self.base_model_path}")

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.base_model_path)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Load base model
        self.model = AutoModelForCausalLM.from_pretrained(
            self.base_model_path,
            torch_dtype=torch.float16,
            device_map="auto"
        )

        # Load adapter config
        config_path = os.path.join(self.adapter_path, "adapter_config.json")
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Adapter config not found at {config_path}")

        with open(config_path, 'r') as f:
            config_dict = json.load(f)

        self.mixlora_config = MixLoraConfig.from_config(config_dict)
        self.mixlora_config.dtype_ = torch.float16

        # Create dummy weights for injection
        dummy_weights = self._create_dummy_weights()

        # Inject MixLoRA adapters
        inject_adapter_in_model(self.model, self.mixlora_config, dummy_weights)

        # Load trained adapter weights
        adapter_weights_path = os.path.join(self.adapter_path, "adapter_model.bin")
        if not os.path.exists(adapter_weights_path):
            raise FileNotFoundError(f"Adapter weights not found at {adapter_weights_path}")

        adapter_weights = torch.load(adapter_weights_path, map_location=self.device)
        missing_keys, unexpected_keys = self.model.load_state_dict(adapter_weights, strict=False)

        self.model.eval()
        self.logger.info("Model loaded successfully with trained adapter weights")

    def _create_dummy_weights(self) -> Dict[str, torch.Tensor]:
        """Create dummy weights for MixLoRA injection."""
        weights = {}

        # Get model dimensions
        hidden_size = self.model.config.hidden_size
        num_layers = self.model.config.num_hidden_layers

        # Get actual layer for dimension inspection
        sample_layer = self.model.model.layers[0]

        for layer_idx in range(num_layers):
            # Router gate weights
            weights[f"mixlora.layers.{layer_idx}.mlp.moe_gate.weight"] = torch.randn(
                self.mixlora_config.num_experts_, hidden_size, dtype=torch.float16
            ) * self.mixlora_config.router_init_range_

            # Expert LoRA weights for each target module
            for module_name, is_target in self.mixlora_config.target_modules_.items():
                if not is_target:
                    continue

                # Get actual dimensions from the model layer
                try:
                    if hasattr(sample_layer.mlp, module_name):
                        actual_layer = getattr(sample_layer.mlp, module_name)
                        if hasattr(actual_layer, 'weight'):
                            out_features, in_features = actual_layer.weight.shape
                        elif hasattr(actual_layer, 'in_features') and hasattr(actual_layer, 'out_features'):
                            in_features = actual_layer.in_features
                            out_features = actual_layer.out_features
                        else:
                            # Fallback based on module type
                            if module_name in ["gate_proj", "up_proj"]:
                                in_features = hidden_size
                                out_features = getattr(self.model.config, 'intermediate_size', hidden_size * 4)
                            elif module_name == "down_proj":
                                in_features = getattr(self.model.config, 'intermediate_size', hidden_size * 4)
                                out_features = hidden_size
                            elif module_name == "fc1":
                                in_features = hidden_size
                                out_features = getattr(self.model.config, 'intermediate_size', hidden_size * 4)
                            elif module_name == "fc2":
                                in_features = getattr(self.model.config, 'intermediate_size', hidden_size * 4)
                                out_features = hidden_size
                            else:
                                in_features = hidden_size
                                out_features = hidden_size
                    else:
                        # Module doesn't exist, skip
                        continue

                    # Create LoRA weights for each expert
                    for expert_idx in range(self.mixlora_config.num_experts_):
                        prefix = f"mixlora.layers.{layer_idx}.mlp.{module_name}.experts.{expert_idx}"

                        # LoRA A matrix
                        weights[f"{prefix}.lora_A.weight"] = torch.randn(
                            self.mixlora_config.lora_r_, in_features, dtype=torch.float16
                        ) * 0.01

                        # LoRA B matrix
                        weights[f"{prefix}.lora_B.weight"] = torch.zeros(
                            out_features, self.mixlora_config.lora_r_, dtype=torch.float16
                        )

                except Exception as e:
                    self.logger.warning(f"Could not determine dimensions for {module_name}: {e}")
                    continue

            # Attention LoRA weights
            for module_name, is_target in self.mixlora_config.target_modules_.items():
                if not is_target or module_name in ["gate_proj", "up_proj", "down_proj", "fc1", "fc2", "gate_up_proj"]:
                    continue

                prefix = f"mixlora.layers.{layer_idx}.self_attn.{module_name}"

                # Get actual dimensions from the model layer
                try:
                    if hasattr(sample_layer.self_attn, module_name):
                        actual_layer = getattr(sample_layer.self_attn, module_name)
                        if hasattr(actual_layer, 'weight'):
                            out_features, in_features = actual_layer.weight.shape
                        elif hasattr(actual_layer, 'in_features') and hasattr(actual_layer, 'out_features'):
                            in_features = actual_layer.in_features
                            out_features = actual_layer.out_features
                        else:
                            # Fallback to config values
                            in_features = hidden_size
                            out_features = hidden_size
                    else:
                        # Module doesn't exist, skip
                        continue

                    # LoRA A matrix (in_features -> rank)
                    weights[f"{prefix}.lora_A.weight"] = torch.randn(
                        self.mixlora_config.lora_r_, in_features, dtype=torch.float16
                    ) * 0.01

                    # LoRA B matrix (rank -> out_features)
                    weights[f"{prefix}.lora_B.weight"] = torch.zeros(
                        out_features, self.mixlora_config.lora_r_, dtype=torch.float16
                    )

                except Exception as e:
                    self.logger.warning(f"Could not determine dimensions for attention module {module_name}: {e}")
                    continue

        return weights

    def _extract_answer_from_generation(self, generated_text: str, choice_range: List[str]) -> Optional[str]:
        """
        Extract the choice answer from generated text with improved logic.

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

        # Extract all numbers from the text
        numbers = re.findall(r'\d+', generated_text)

        # Try to find valid choices among the extracted numbers
        for num in numbers:
            if num in choice_range:
                return num

        # If still no match, try to find any digit that could be converted to a valid choice
        for char in generated_text:
            if char.isdigit() and char in choice_range:
                return char

        # Try to find choices in the text (case insensitive)
        for choice in choice_range:
            if choice.lower() in generated_text.lower():
                return choice

        # If nothing found, return None
        return None

    def predict_single(self, instruction: str, input_text: str = "", choice_range: List[str] = None) -> Tuple[str, str]:
        """
        Predict a single example.

        Args:
            instruction: The instruction text
            input_text: Additional input text (usually empty)
            choice_range: Valid choice range

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

        # Extract choice
        extracted_choice = None
        if choice_range:
            extracted_choice = self._extract_answer_from_generation(generated_text, choice_range)

        return generated_text, extracted_choice

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
        detailed_results = []

        self.model.eval()
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Evaluating"):
                # Get the original instructions for generation
                for i in range(len(batch['choice_answers'])):
                    # Reconstruct the instruction from input_ids
                    input_ids = batch['input_ids'][i]
                    labels = batch['labels'][i]
                    instruction_length = (labels == -100).sum().item()
                    instruction_ids = input_ids[:instruction_length]
                    instruction = self.tokenizer.decode(instruction_ids, skip_special_tokens=True)

                    # Generate prediction
                    generated_text, predicted_choice = self.predict_single(
                        instruction, choice_range=choice_range
                    )

                    target_choice = batch['choice_answers'][i]

                    all_predictions.append(predicted_choice)
                    all_targets.append(target_choice)
                    all_generated.append(generated_text)

                    # Store detailed result with required fields
                    detailed_results.append({
                        'instruction': instruction,  # 原始问题
                        'target': target_choice,     # 正确答案
                        'predicted': predicted_choice,  # 预测答案
                        'correct': predicted_choice == target_choice,  # 是否正确
                        'generated_text': generated_text  # 生成的原始文本
                    })

        # Handle None predictions by treating them as wrong
        valid_predictions = []
        valid_targets = []

        for pred, target in zip(all_predictions, all_targets):
            if pred is not None:
                valid_predictions.append(pred)
                valid_targets.append(target)
            else:
                # Treat None as wrong prediction, use first choice as placeholder
                valid_predictions.append(choice_range[0])
                valid_targets.append(target)

        # Calculate metrics
        accuracy = accuracy_score(valid_targets, valid_predictions)
        precision, recall, f1, support = precision_recall_fscore_support(
            valid_targets, valid_predictions, average='macro', zero_division=0
        )

        # Calculate per-choice metrics
        choice_metrics = {}
        for choice in choice_range:
            choice_correct = sum(1 for p, t in zip(valid_predictions, valid_targets)
                               if t == choice and p == choice)
            choice_total = sum(1 for t in valid_targets if t == choice)
            choice_predicted = sum(1 for p in valid_predictions if p == choice)

            choice_precision = choice_correct / choice_predicted if choice_predicted > 0 else 0.0
            choice_recall = choice_correct / choice_total if choice_total > 0 else 0.0
            choice_f1 = 2 * choice_precision * choice_recall / (choice_precision + choice_recall) if (choice_precision + choice_recall) > 0 else 0.0

            choice_metrics[choice] = {
                'precision': choice_precision,
                'recall': choice_recall,
                'f1': choice_f1,
                'support': choice_total
            }

        metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'total_samples': len(all_targets),
            'valid_predictions': len([p for p in all_predictions if p is not None]),
            'choice_metrics': choice_metrics
        }

        # Print results
        self.logger.info(f"Evaluation Results:")
        self.logger.info(f"Accuracy: {accuracy:.4f}")
        self.logger.info(f"Precision: {precision:.4f}")
        self.logger.info(f"Recall: {recall:.4f}")
        self.logger.info(f"F1: {f1:.4f}")
        self.logger.info(f"Valid predictions: {metrics['valid_predictions']}/{metrics['total_samples']}")

        # Save predictions if requested
        if save_predictions:
            if output_file is None:
                output_file = f"predictions_{os.path.basename(dataset_path)}.json"

            # Save predictions and metrics (generated_answers.json format)
            output_data = {
                'metrics': metrics,
                'predictions': detailed_results  # Contains instruction, target, predicted, correct fields
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
                choice_range_input = input("Enter valid choices (comma-separated, e.g., 1,2,3,4): ").strip()

                choice_range = None
                if choice_range_input:
                    choice_range = [c.strip() for c in choice_range_input.split(',')]

                generated_text, extracted_choice = self.predict_single(
                    instruction, input_text, choice_range
                )

                print(f"Generated: {generated_text}")
                print(f"Extracted choice: {extracted_choice}")

            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"Error: {e}")

        self.logger.info("Interactive mode ended.")


def main():
    parser = argparse.ArgumentParser(description="Inference with trained MixLoRA adapter")

    parser.add_argument("--base_model_path", type=str, required=True,
                       help="Path to base model")
    parser.add_argument("--adapter_path", type=str, required=True,
                       help="Path to trained adapter weights")
    parser.add_argument("--dataset_path", type=str,
                       help="Path to evaluation dataset")
    parser.add_argument("--output_file", type=str,
                       help="Output file for predictions")
    parser.add_argument("--batch_size", type=int, default=8,
                       help="Batch size for inference")
    parser.add_argument("--max_new_tokens", type=int, default=10,
                       help="Maximum new tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.1,
                       help="Sampling temperature")
    parser.add_argument("--do_sample", action="store_true",
                       help="Use sampling instead of greedy decoding")
    parser.add_argument("--interactive", action="store_true",
                       help="Run in interactive mode")
    parser.add_argument("--device", type=str,
                       help="Device to run inference on")

    args = parser.parse_args()

    # Create inference engine
    inference_engine = CustomMixLoRAInference(
        base_model_path=args.base_model_path,
        adapter_path=args.adapter_path,
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
        print(f"Final Results:")
        print(f"Accuracy: {metrics['accuracy']:.4f}")
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall: {metrics['recall']:.4f}")
        print(f"F1: {metrics['f1']:.4f}")
    else:
        print("Please specify either --dataset_path for evaluation or --interactive for interactive mode")


if __name__ == "__main__":
    main()