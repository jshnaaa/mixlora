"""
Custom MixLoRA training script for cultural datasets.
Supports automatic dataset splitting, best model saving, and comprehensive evaluation.
"""

import argparse
import json
import logging
import os
import sys
import math
import re
from typing import Dict, Optional, List, Tuple
from dataclasses import dataclass, field
from datetime import datetime

# Set memory optimization environment variables first
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# Enable dual-GPU training for large memory GPUs (48G+)
# Users can override this by setting CUDA_VISIBLE_DEVICES before running
if "CUDA_VISIBLE_DEVICES" not in os.environ:
    import torch
    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        if gpu_count >= 2:
            # Use first two GPUs for training
            os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
            print(f"Using dual-GPU for training: {os.environ['CUDA_VISIBLE_DEVICES']}")
        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = "0"
            print(f"Using single GPU for training: {os.environ['CUDA_VISIBLE_DEVICES']}")

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    TrainerCallback,
    TrainerState,
    TrainerControl
)
from transformers.trainer_utils import get_last_checkpoint
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report

# Add parent directory to path to import mixlora
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mixlora.config import MixLoraConfig
from mixlora.model import inject_adapter_in_model, load_adapter_weights, MixLoraModelForCausalLM
from dataset import ChoiceQuestionDataset, ChoiceQuestionCollator


@dataclass
class CustomTrainingArguments:
    """Custom training arguments for cultural datasets."""

    # Dataset configuration
    data_id: int = field(default=2, metadata={"help": "Dataset ID (2=culturalbench, 3=normad)"})
    backbone: str = field(default="llama", metadata={"help": "Model backbone (llama, qwen)"})
    base_model: str = field(default="/root/autodl-tmp/CultureMoE/Culture_Alignment/Meta-Llama-3.1-8B-Instruct",
                           metadata={"help": "Base model path"})
    output_dir: str = field(default="./mixlora_output", metadata={"help": "Output directory"})

    # MixLoRA parameters
    num_experts: int = field(default=8, metadata={"help": "Number of experts"})
    top_k: int = field(default=2, metadata={"help": "Top-k routing"})
    routing_strategy: str = field(default="mixlora", metadata={"help": "Routing strategy"})
    router_aux_loss_coef: float = field(default=0.01, metadata={"help": "Router auxiliary loss coefficient"})
    router_init_range: float = field(default=0.02, metadata={"help": "Router initialization range"})
    jitter_noise: float = field(default=0.0, metadata={"help": "Jitter noise"})

    # LoRA parameters
    lora_r: int = field(default=8, metadata={"help": "LoRA rank"})
    lora_alpha: int = field(default=16, metadata={"help": "LoRA alpha"})
    lora_dropout: float = field(default=0.05, metadata={"help": "LoRA dropout"})
    use_dora: bool = field(default=False, metadata={"help": "Use DoRA"})
    use_rslora: bool = field(default=False, metadata={"help": "Use RSLoRA"})

    # Training parameters
    max_length: int = field(default=512, metadata={"help": "Maximum sequence length"})
    batch_size: int = field(default=4, metadata={"help": "Training batch size"})
    gradient_accumulation_steps: int = field(default=4, metadata={"help": "Gradient accumulation steps"})
    learning_rate: float = field(default=1e-4, metadata={"help": "Learning rate"})
    num_epochs: int = field(default=3, metadata={"help": "Number of training epochs"})
    warmup_ratio: float = field(default=0.1, metadata={"help": "Warmup ratio"})
    weight_decay: float = field(default=0.01, metadata={"help": "Weight decay"})

    # Evaluation parameters
    eval_interval: int = field(default=1, metadata={"help": "Evaluate every N epochs"})
    save_steps: int = field(default=500, metadata={"help": "Save steps"})
    logging_steps: int = field(default=10, metadata={"help": "Logging steps"})

    # Experiment tracking
    wandb_project: Optional[str] = field(default=None, metadata={"help": "Weights & Biases project name"})
    run_name: Optional[str] = field(default=None, metadata={"help": "Run name for logging"})


class BestModelTracker(TrainerCallback):
    """Callback to track and save the best model based on validation accuracy."""

    def __init__(self, trainer_instance):
        self.best_accuracy = 0.0
        self.best_model_path = None
        self.trainer_instance = trainer_instance
        self.eval_results = []

    def on_evaluate(self, args, state, control, model, eval_dataloader, **kwargs):
        """Called after evaluation."""
        if hasattr(state, 'log_history') and len(state.log_history) > 0:
            last_log = state.log_history[-1]
            if 'eval_accuracy' in last_log:
                current_accuracy = last_log['eval_accuracy']

                # Save evaluation result
                eval_result = {
                    'epoch': state.epoch,
                    'step': state.global_step,
                    'eval_accuracy': current_accuracy,
                    'eval_loss': last_log.get('eval_loss', 0.0)
                }
                self.eval_results.append(eval_result)

                # Check if this is the best model so far
                if current_accuracy > self.best_accuracy:
                    self.best_accuracy = current_accuracy

                    # Save the best model (only adapter weights)
                    best_model_dir = os.path.join(args.output_dir, "best_model")
                    os.makedirs(best_model_dir, exist_ok=True)

                    # Save adapter weights
                    self.trainer_instance.save_adapter_weights(best_model_dir)
                    self.best_model_path = best_model_dir

                    logging.info(f"New best model saved with accuracy: {current_accuracy:.4f}")

                # Save evaluation results
                eval_results_path = os.path.join(args.output_dir, "validation_results.json")
                with open(eval_results_path, 'w') as f:
                    json.dump(self.eval_results, f, indent=2)


class CustomMixLoRATrainer:
    """Custom trainer for MixLoRA on cultural datasets."""

    def __init__(self, args: CustomTrainingArguments):
        self.args = args
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Set up logging
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
        self.logger = logging.getLogger(__name__)

        # Get dataset configuration
        self.dataset_config = self._get_dataset_config()

        # Set up output directory (use the one passed from shell script)
        self.output_dir = self.args.output_dir
        os.makedirs(self.output_dir, exist_ok=True)

        self.logger.info(f"Output directory: {self.output_dir}")

    def _get_dataset_config(self) -> Dict[str, str]:
        """Get dataset configuration based on data_id."""
        configs = {
            2: {
                "path": "/root/autodl-fs/CulturalBench_merge_gen.json",
                "tag": "culturalbench"
            },
            3: {
                "path": "/root/autodl-fs/normad_merge_gen.json",
                "tag": "normad"
            }
        }

        if self.args.data_id not in configs:
            raise ValueError(f"Unsupported data_id: {self.args.data_id}. Supported values: {list(configs.keys())}")

        return configs[self.args.data_id]

    def _get_default_target_modules(self, model_type: str) -> List[str]:
        """Get default target modules based on model type."""
        if model_type in ["llama", "mistral", "gemma", "gemma2", "qwen2"]:
            return ["q_proj", "v_proj", "gate_proj", "up_proj", "down_proj"]
        elif model_type == "phi":
            return ["q_proj", "v_proj", "fc1", "fc2"]
        elif model_type == "phi3":
            return ["qkv_proj", "o_proj", "gate_up_proj", "down_proj"]
        else:
            self.logger.warning(f"Unknown model type {model_type}, using default target modules")
            return ["q_proj", "v_proj", "gate_proj", "up_proj", "down_proj"]

    def setup_model_and_tokenizer(self):
        """Setup the base model, tokenizer, and inject MixLoRA adapters."""
        self.logger.info(f"Loading base model: {self.args.base_model}")

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.args.base_model)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Load base model with smart GPU allocation
        gpu_count = torch.cuda.device_count()
        self.logger.info(f"Available GPUs: {gpu_count}")

        if gpu_count >= 2:
            # Dual-GPU setup: load on CPU first, then move to primary GPU
            self.logger.info("Using dual-GPU setup with DDP")
            self.model = AutoModelForCausalLM.from_pretrained(
                self.args.base_model,
                torch_dtype=torch.float16,
                trust_remote_code=True,
                low_cpu_mem_usage=True,
                device_map=None  # Load on CPU first
            )
            self.model_dtype = torch.float16
        else:
            # Single GPU setup
            self.logger.info(f"Using single GPU: {self.device}")
            self.model = AutoModelForCausalLM.from_pretrained(
                self.args.base_model,
                torch_dtype=torch.float16,
                trust_remote_code=True,
                low_cpu_mem_usage=True
            )
            self.model_dtype = torch.float16

        self.logger.info(f"Model loaded successfully with dtype: {self.model_dtype}")

        # Determine target modules
        target_modules = self._get_default_target_modules(self.model.config.model_type)

        # Get the activation function from the base model
        act_fn = None
        if hasattr(self.model.config, 'hidden_act'):
            act_fn = self.model.config.hidden_act
        elif hasattr(self.model, 'model') and hasattr(self.model.model, 'layers'):
            # Try to get activation function from the first layer
            first_layer = self.model.model.layers[0]
            if hasattr(first_layer, 'mlp') and hasattr(first_layer.mlp, 'act_fn'):
                # For models where act_fn is an attribute
                act_fn_obj = first_layer.mlp.act_fn
                # Map common activation functions to string names
                if hasattr(act_fn_obj, '__class__'):
                    act_fn_name = act_fn_obj.__class__.__name__.lower()
                    if 'silu' in act_fn_name or 'swish' in act_fn_name:
                        act_fn = 'silu'
                    elif 'gelu' in act_fn_name:
                        act_fn = 'gelu'
                    elif 'relu' in act_fn_name:
                        act_fn = 'relu'

        # Default to silu if not found (common for LLaMA models)
        if act_fn is None:
            act_fn = 'silu'
            self.logger.warning(f"Could not determine activation function, defaulting to 'silu'")
        else:
            self.logger.info(f"Detected activation function: {act_fn}")

        # Create MixLoRA config
        mixlora_config_dict = {
            "base_model_name_or_path": self.args.base_model,
            "task_type": "CAUSAL_LM",
            "peft_type": "MIXLORA",
            "r": self.args.lora_r,
            "lora_alpha": self.args.lora_alpha,
            "lora_dropout": self.args.lora_dropout,
            "target_modules": {module: True for module in target_modules},
            "use_dora": self.args.use_dora,
            "use_rslora": self.args.use_rslora,
            "routing_strategy": self.args.routing_strategy,
            "num_experts": self.args.num_experts,
            "top_k": self.args.top_k,
            "router_aux_loss_coef": self.args.router_aux_loss_coef,
            "router_init_range": self.args.router_init_range,
            "jitter_noise": self.args.jitter_noise,
            "act_fn": act_fn,  # Add the activation function
        }

        self.mixlora_config = MixLoraConfig.from_config(mixlora_config_dict)
        self.mixlora_config.dtype_ = self.model_dtype  # Use same dtype as model

        # Initialize MixLoRA weights
        dummy_weights = self._create_dummy_weights()

        # Inject MixLoRA adapters
        inject_adapter_in_model(self.model, self.mixlora_config, dummy_weights)

        # Ensure model and all adapters are properly placed
        # Always move everything to self.device for consistency
        self.model = self.model.to(self.device)

        # Also ensure all MixLoRA components are on the correct device
        for layer in self.model.model.layers:
            if hasattr(layer.mlp, 'mixlora_moes'):
                for moe_name, moe_layer in layer.mlp.mixlora_moes.items():
                    moe_layer.to(self.device)
                    if hasattr(moe_layer, 'gate_') and moe_layer.gate_ is not None:
                        moe_layer.gate_ = moe_layer.gate_.to(self.device)
                    for expert_name, expert in moe_layer.experts_.items():
                        expert.to(self.device)

            if hasattr(layer.self_attn, 'mixlora_loras'):
                for lora_name, lora_layer in layer.self_attn.mixlora_loras.items():
                    lora_layer.to(self.device)

        # Verify all components are on the correct device
        self._verify_device_consistency()
        self.logger.info(f"MixLoRA adapters injected successfully, model on device: {next(self.model.parameters()).device}")

    def _create_dummy_weights(self) -> Dict[str, torch.Tensor]:
        """Create dummy weights for MixLoRA injection."""
        weights = {}

        # Get model dimensions
        hidden_size = self.model.config.hidden_size
        num_layers = self.model.config.num_hidden_layers

        # Get actual layer for dimension inspection and device detection
        sample_layer = self.model.model.layers[0]

        # Determine the target device for weights
        # Always use self.device (cuda:0) for consistency
        target_device = self.device
        self.logger.info(f"Creating weights on device: {target_device}")

        for layer_idx in range(num_layers):
            # Router gate weights
            weights[f"mixlora.layers.{layer_idx}.mlp.moe_gate.weight"] = torch.randn(
                self.args.num_experts, hidden_size, dtype=self.model_dtype, device=target_device
            ) * self.args.router_init_range

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
                    for expert_idx in range(self.args.num_experts):
                        prefix = f"mixlora.layers.{layer_idx}.mlp.{module_name}.experts.{expert_idx}"

                        # LoRA A matrix
                        weights[f"{prefix}.lora_A.weight"] = torch.randn(
                            self.args.lora_r, in_features, dtype=self.model_dtype, device=target_device
                        ) * 0.01

                        # LoRA B matrix
                        weights[f"{prefix}.lora_B.weight"] = torch.zeros(
                            out_features, self.args.lora_r, dtype=self.model_dtype, device=target_device
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
                        self.args.lora_r, in_features, dtype=self.model_dtype, device=target_device
                    ) * 0.01

                    # LoRA B matrix (rank -> out_features)
                    weights[f"{prefix}.lora_B.weight"] = torch.zeros(
                        out_features, self.args.lora_r, dtype=self.model_dtype, device=target_device
                    )

                except Exception as e:
                    self.logger.warning(f"Could not determine dimensions for attention module {module_name}: {e}")
                    continue

        return weights

    def _verify_device_consistency(self):
        """Verify that all model components are on the correct device."""
        target_device = self.device

        # Check main model parameters
        for name, param in self.model.named_parameters():
            if param.device != target_device:
                self.logger.warning(f"Parameter {name} on device {param.device}, expected {target_device}")

        # Check MixLoRA components specifically
        for layer_idx, layer in enumerate(self.model.model.layers):
            if hasattr(layer.mlp, 'mixlora_moes'):
                for moe_name, moe_layer in layer.mlp.mixlora_moes.items():
                    # Check gate weights
                    if hasattr(moe_layer, 'gate_') and moe_layer.gate_ is not None:
                        if moe_layer.gate_.device != target_device:
                            self.logger.warning(f"Layer {layer_idx} MoE gate on device {moe_layer.gate_.device}, expected {target_device}")

                    # Check expert weights
                    for expert_name, expert in moe_layer.experts_.items():
                        for param_name, param in expert.named_parameters():
                            if param.device != target_device:
                                self.logger.warning(f"Layer {layer_idx} expert {expert_name} param {param_name} on device {param.device}, expected {target_device}")

        self.logger.info(f"Device consistency check completed for device: {target_device}")

    def setup_datasets(self):
        """Setup training, validation, and test datasets with 8:1:1 split."""
        self.logger.info("Loading and splitting datasets...")

        # Load full dataset
        full_dataset = ChoiceQuestionDataset(
            data_path=self.dataset_config['path'],
            tokenizer=self.tokenizer,
            max_length=self.args.max_length,
            choice_range=None  # Auto-detect
        )

        # Get choice range for later use
        self.choice_range = full_dataset.choice_range
        self.logger.info(f"Detected choice range: {self.choice_range}")

        # Split dataset randomly with 8:1:1 ratio
        total_size = len(full_dataset)
        train_size = int(0.8 * total_size)
        val_size = int(0.1 * total_size)
        test_size = total_size - train_size - val_size

        # Use generator for reproducible splits
        generator = torch.Generator().manual_seed(42)
        self.train_dataset, self.val_dataset, self.test_dataset = random_split(
            full_dataset, [train_size, val_size, test_size], generator=generator
        )

        self.logger.info(f"Dataset split - Train: {len(self.train_dataset)}, Val: {len(self.val_dataset)}, Test: {len(self.test_dataset)}")

        # Setup data collator
        self.data_collator = ChoiceQuestionCollator(tokenizer=self.tokenizer)

    def save_adapter_weights(self, save_path: str):
        """Save only the adapter weights (not the full model)."""
        os.makedirs(save_path, exist_ok=True)

        # Extract adapter weights from the model
        adapter_weights = {}

        for name, param in self.model.named_parameters():
            if any(adapter_name in name for adapter_name in ['lora_A', 'lora_B', 'moe_gate', 'experts']):
                adapter_weights[name] = param.data.clone()

        # Save adapter weights
        torch.save(adapter_weights, os.path.join(save_path, "adapter_model.bin"))

        # Save adapter config
        config_path = os.path.join(save_path, "adapter_config.json")
        with open(config_path, 'w') as f:
            json.dump(self.mixlora_config.export(), f, indent=2)

        self.logger.info(f"Adapter weights saved to {save_path}")

    def extract_answer_from_generation(self, generated_text: str) -> Optional[str]:
        """Extract answer choice from generated text."""
        # Clean the text
        generated_text = generated_text.strip()

        # First try exact match with choice range
        for choice in self.choice_range:
            if generated_text == choice:
                return choice

        # Try to find digits in the text
        digits = re.findall(r'\d+', generated_text)
        for digit in digits:
            if digit in self.choice_range:
                return digit

        # If no valid choice found, return None
        return None

    def evaluate_on_dataset(self, dataset, dataset_name: str) -> Tuple[Dict[str, float], List[Dict]]:
        """Evaluate model on a dataset and return metrics and detailed results."""
        self.model.eval()

        dataloader = DataLoader(
            dataset,
            batch_size=self.args.batch_size,
            shuffle=False,
            collate_fn=self.data_collator
        )

        all_predictions = []
        all_targets = []
        detailed_results = []

        with torch.no_grad():
            for batch in dataloader:
                # Get original data for each sample in batch
                for i in range(len(batch['choice_answers'])):
                    # Reconstruct instruction from input_ids
                    input_ids = batch['input_ids'][i]
                    labels = batch['labels'][i]
                    instruction_length = (labels == -100).sum().item()
                    instruction_ids = input_ids[:instruction_length]
                    instruction = self.tokenizer.decode(instruction_ids, skip_special_tokens=True)

                    # Generate response
                    inputs = self.tokenizer(
                        instruction,
                        return_tensors="pt",
                        truncation=True,
                        max_length=self.args.max_length
                    ).to(self.device)

                    outputs = self.model.generate(
                        **inputs,
                        max_new_tokens=10,
                        temperature=0.1,
                        do_sample=False,
                        pad_token_id=self.tokenizer.eos_token_id,
                    )

                    generated_text = self.tokenizer.decode(
                        outputs[0][inputs['input_ids'].shape[1]:],
                        skip_special_tokens=True
                    ).strip()

                    predicted_choice = self.extract_answer_from_generation(generated_text)
                    target_choice = batch['choice_answers'][i]

                    all_predictions.append(predicted_choice)
                    all_targets.append(target_choice)

                    # Store detailed result
                    detailed_results.append({
                        'instruction': instruction,
                        'target': target_choice,
                        'predicted': predicted_choice,
                        'generated_text': generated_text,
                        'correct': predicted_choice == target_choice
                    })

        # Calculate metrics
        # Handle None predictions by treating them as wrong
        valid_predictions = []
        valid_targets = []

        for pred, target in zip(all_predictions, all_targets):
            if pred is not None:
                valid_predictions.append(pred)
                valid_targets.append(target)
            else:
                # Treat None as a wrong prediction, use first choice as placeholder
                valid_predictions.append(self.choice_range[0])
                valid_targets.append(target)

        # Calculate metrics
        accuracy = accuracy_score(valid_targets, valid_predictions)
        precision, recall, f1, support = precision_recall_fscore_support(
            valid_targets, valid_predictions, average='macro', zero_division=0
        )

        metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'total_samples': len(all_targets),
            'valid_predictions': len([p for p in all_predictions if p is not None])
        }

        self.logger.info(f"{dataset_name} Results - Accuracy: {accuracy:.4f}, F1: {f1:.4f}")

        return metrics, detailed_results

    def compute_metrics(self, eval_pred):
        """Compute metrics for the trainer."""
        predictions, labels = eval_pred

        # This is a simplified version - the main evaluation happens in evaluate_on_dataset
        # We'll use this just to get basic accuracy for the trainer callback

        # For now, return a dummy accuracy that will be overridden
        return {"accuracy": 0.0}

    def train(self):
        """Run the training loop."""
        self.logger.info("Starting training...")

        # Calculate eval steps based on eval_interval
        total_steps = len(self.train_dataset) // (self.args.batch_size * self.args.gradient_accumulation_steps) * self.args.num_epochs
        eval_steps = max(1, total_steps // (self.args.num_epochs // self.args.eval_interval)) if self.args.eval_interval > 0 else total_steps

        # Check precision support and GPU setup
        gpu_count = torch.cuda.device_count()
        bf16_available = torch.cuda.is_bf16_supported()
        self.logger.info(f"BF16 support: {bf16_available}")

        # Use FP16 for memory efficiency on both single and dual GPU
        if gpu_count >= 2:
            self.logger.info(f"Using FP16 precision for dual-GPU training ({gpu_count} GPUs)")
        else:
            self.logger.info("Using FP16 precision for single GPU training")

        use_fp16 = True
        use_bf16 = False

        # Use the specified batch size per device
        per_device_batch_size = self.args.batch_size
        total_batch_size = per_device_batch_size * max(1, gpu_count) * self.args.gradient_accumulation_steps
        self.logger.info(f"Batch size per device: {per_device_batch_size}")
        self.logger.info(f"Total effective batch size: {total_batch_size} (GPUs: {gpu_count})")

        # Setup training arguments
        training_args = TrainingArguments(
            output_dir=self.output_dir,
            per_device_train_batch_size=per_device_batch_size,
            per_device_eval_batch_size=per_device_batch_size,
            gradient_accumulation_steps=self.args.gradient_accumulation_steps,
            learning_rate=self.args.learning_rate,
            num_train_epochs=self.args.num_epochs,
            warmup_ratio=self.args.warmup_ratio,
            weight_decay=self.args.weight_decay,
            logging_steps=self.args.logging_steps,
            eval_steps=eval_steps,
            save_steps=self.args.save_steps,
            evaluation_strategy="steps",
            save_strategy="steps",
            load_best_model_at_end=False,  # We handle this manually
            # Dynamic precision based on GPU setup
            bf16=use_bf16,
            fp16=use_fp16,
            # Additional stability settings
            gradient_checkpointing=True,  # Enable to save memory
            max_grad_norm=1.0,  # Gradient clipping
            # Memory optimization
            save_safetensors=True,  # Use safer tensor format
            optim="adamw_torch",  # Use PyTorch AdamW for better memory
            dataloader_pin_memory=False,
            remove_unused_columns=False,
            report_to=None,  # Disable wandb for now
            # Multi-GPU settings
            dataloader_num_workers=0,  # Avoid multiprocessing issues
            dataloader_drop_last=True,  # Ensure even batch distribution
            # DDP settings for dual-GPU training
            ddp_find_unused_parameters=False,  # Set to False for better performance
            ddp_timeout=1800,  # 30 minutes timeout for large models
        )

        # Create trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.val_dataset,
            data_collator=self.data_collator,
            tokenizer=self.tokenizer,
            compute_metrics=self.compute_metrics,
        )

        # Add best model tracker callback
        best_model_tracker = BestModelTracker(self)
        trainer.add_callback(best_model_tracker)

        # Store reference for saving adapter weights
        self.trainer = trainer

        # Final device verification before training
        self.logger.info(f"Final device check - Model device: {next(self.model.parameters()).device}")
        self.logger.info(f"Training device: {self.device}")

        # Train
        trainer.train()

        # Save final training config
        config_path = os.path.join(self.output_dir, "training_config.json")
        with open(config_path, 'w') as f:
            json.dump(vars(self.args), f, indent=2)

        # Load best model and evaluate on test set
        self.logger.info("Loading best model for test evaluation...")

        if best_model_tracker.best_model_path:
            # Load the best adapter weights
            best_adapter_path = os.path.join(best_model_tracker.best_model_path, "adapter_model.bin")
            if os.path.exists(best_adapter_path):
                adapter_weights = torch.load(best_adapter_path, map_location=self.device)

                # Load adapter weights into model
                missing_keys, unexpected_keys = self.model.load_state_dict(adapter_weights, strict=False)
                self.logger.info(f"Loaded best model adapter weights")

                # Evaluate on test set
                test_metrics, test_results = self.evaluate_on_dataset(self.test_dataset, "Test")

                # Save test results
                test_results_path = os.path.join(self.output_dir, "test_results.json")
                with open(test_results_path, 'w') as f:
                    json.dump({
                        'metrics': test_metrics,
                        'detailed_results': test_results
                    }, f, indent=2)

                # Save generated answers for validation set
                val_metrics, val_results = self.evaluate_on_dataset(self.val_dataset, "Validation")
                generated_answers_path = os.path.join(self.output_dir, "generated_answers.json")
                with open(generated_answers_path, 'w') as f:
                    json.dump(val_results, f, indent=2)

                self.logger.info(f"Final test accuracy: {test_metrics['accuracy']:.4f}")
                self.logger.info(f"Results saved to {self.output_dir}")
            else:
                self.logger.error("Best model adapter weights not found!")
        else:
            self.logger.error("No best model was saved during training!")


def main():
    parser = argparse.ArgumentParser(description="Train MixLoRA on cultural datasets")

    # Dataset configuration
    parser.add_argument("--data_id", type=int, default=2, help="Dataset ID (2=culturalbench, 3=normad)")
    parser.add_argument("--backbone", type=str, default="llama", help="Model backbone (llama, qwen)")
    parser.add_argument("--base_model", type=str,
                       default="/root/autodl-tmp/CultureMoE/Culture_Alignment/Meta-Llama-3.1-8B-Instruct",
                       help="Base model path")
    parser.add_argument("--output_dir", type=str, default="./mixlora_output", help="Output directory")

    # MixLoRA parameters
    parser.add_argument("--num_experts", type=int, default=8, help="Number of experts")
    parser.add_argument("--top_k", type=int, default=2, help="Top-k routing")
    parser.add_argument("--routing_strategy", type=str, default="mixlora", help="Routing strategy")
    parser.add_argument("--router_aux_loss_coef", type=float, default=0.01, help="Router auxiliary loss coefficient")
    parser.add_argument("--router_init_range", type=float, default=0.02, help="Router initialization range")
    parser.add_argument("--jitter_noise", type=float, default=0.0, help="Jitter noise")

    # LoRA parameters
    parser.add_argument("--lora_r", type=int, default=8, help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=16, help="LoRA alpha")
    parser.add_argument("--lora_dropout", type=float, default=0.05, help="LoRA dropout")
    parser.add_argument("--use_dora", action="store_true", help="Use DoRA")
    parser.add_argument("--use_rslora", action="store_true", help="Use RSLoRA")

    # Training parameters
    parser.add_argument("--max_length", type=int, default=512, help="Maximum sequence length")
    parser.add_argument("--batch_size", type=int, default=4, help="Training batch size")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4, help="Gradient accumulation steps")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--num_epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--warmup_ratio", type=float, default=0.1, help="Warmup ratio")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay")

    # Evaluation parameters
    parser.add_argument("--eval_interval", type=int, default=1, help="Evaluate every N epochs")
    parser.add_argument("--save_steps", type=int, default=500, help="Save steps")
    parser.add_argument("--logging_steps", type=int, default=10, help="Logging steps")

    # Experiment tracking
    parser.add_argument("--wandb_project", type=str, help="Weights & Biases project name")
    parser.add_argument("--run_name", type=str, help="Run name for logging")

    args = parser.parse_args()

    # Convert to dataclass
    training_args = CustomTrainingArguments(**vars(args))

    # Create trainer and run training
    trainer = CustomMixLoRATrainer(training_args)
    trainer.setup_model_and_tokenizer()
    trainer.setup_datasets()
    trainer.train()


if __name__ == "__main__":
    main()