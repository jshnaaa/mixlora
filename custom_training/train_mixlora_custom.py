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

# CRITICAL: Disable automatic DataParallel at the very beginning
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"  # For debugging
os.environ["TORCH_USE_CUDA_DSA"] = "1"     # Additional debugging

# GPU configuration will be set based on command line argument
# Users can still override by setting CUDA_VISIBLE_DEVICES before running
# Note: The actual GPU setup is done in the trainer initialization

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

# NUCLEAR OPTION: Completely disable DataParallel
# Replace DataParallel with Identity to prevent any usage
original_DataParallel = torch.nn.DataParallel

def DisabledDataParallel(module, *args, **kwargs):
    print("[WARNING] DataParallel usage detected and blocked! Using single device instead.")
    return module  # Just return the original module without wrapping

# Monkey patch to disable DataParallel
torch.nn.DataParallel = DisabledDataParallel
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
    num_gpu: int = field(default=2, metadata={"help": "Number of GPUs to use (1=single, 2=dual)"})

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
        # Since we use safe_collator, we need to do our own evaluation with choice_answers
        logging.info("Performing custom evaluation with choice answer extraction...")

        # Get validation metrics using the trainer's evaluate_on_dataset method
        val_metrics, val_results = self.trainer_instance.evaluate_on_dataset(
            self.trainer_instance.val_dataset, "Validation"
        )
        current_accuracy = val_metrics['accuracy']

        # Save evaluation result
        eval_result = {
            'epoch': state.epoch,
            'step': state.global_step,
            'eval_accuracy': current_accuracy,
            'eval_loss': 0.0  # We don't track loss in our custom evaluation
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

        # Configure GPU setup based on num_gpu parameter
        self._setup_gpu_configuration()

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

    def _setup_gpu_configuration(self):
        """Setup GPU configuration based on num_gpu parameter."""
        if torch.cuda.is_available():
            available_gpus = torch.cuda.device_count()

            if self.args.num_gpu == 1:
                # FORCE single GPU training - override any existing setting
                os.environ["CUDA_VISIBLE_DEVICES"] = "0"
                print(f"FORCE single-GPU training: {os.environ['CUDA_VISIBLE_DEVICES']}")
                print("Note: Using BF16/FP32 precision to avoid gradient scaling issues")

                # Additional check: ensure torch sees only 1 GPU
                import torch
                torch.cuda.empty_cache()  # Clear cache
                print(f"After setting CUDA_VISIBLE_DEVICES=0, torch sees {torch.cuda.device_count()} GPU(s)")

            elif self.args.num_gpu == 2:
                # Dual GPU training
                if available_gpus >= 2:
                    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
                    print(f"Configured for dual-GPU training: {os.environ['CUDA_VISIBLE_DEVICES']}")
                    print("Note: Using BF16/FP32 precision to avoid gradient scaling issues")
                else:
                    print(f"Warning: Requested {self.args.num_gpu} GPUs but only {available_gpus} available")
                    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
                    print(f"Falling back to single-GPU training: {os.environ['CUDA_VISIBLE_DEVICES']}")
                    self.args.num_gpu = 1  # Update the argument
            else:
                raise ValueError(f"Unsupported num_gpu: {self.args.num_gpu}. Supported values: 1, 2")
        else:
            print("CUDA not available, using CPU")

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

        # Load base model based on configured GPU setup
        gpu_count = torch.cuda.device_count()
        self.logger.info(f"Available GPUs: {gpu_count}, Configured to use: {self.args.num_gpu}")

        # Determine model dtype to avoid gradient scaling issues
        if torch.cuda.is_bf16_supported():
            model_dtype = torch.bfloat16
            self.logger.info("Using bfloat16 for model weights")
        else:
            model_dtype = torch.float32
            self.logger.info("Using float32 for model weights (bfloat16 not supported)")

        if self.args.num_gpu >= 2:
            # Multi-GPU setup
            self.logger.info("Loading model for dual-GPU training")
        else:
            # Single GPU setup
            self.logger.info(f"Loading model for single GPU training on: {self.device}")

        self.model = AutoModelForCausalLM.from_pretrained(
            self.args.base_model,
            torch_dtype=model_dtype,
            trust_remote_code=True,
            low_cpu_mem_usage=True
        )

        self.model_dtype = model_dtype

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

        # For multi-GPU training, let Trainer handle device placement automatically
        # Only move to device if single GPU training
        if self.args.num_gpu == 1:
            self.model = self.model.to(self.device)

            # Also ensure all MixLoRA components are on the correct device for single GPU
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
        else:
            # For multi-GPU training, keep model on CPU initially
            # Trainer will handle device placement and parallelization
            self.logger.info("Multi-GPU training: letting Trainer handle device placement")

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

        # Determine the target device and dtype for weights
        # For multi-GPU training, use CPU; for single GPU, use self.device
        if self.args.num_gpu > 1:
            target_device = torch.device("cpu")
        else:
            target_device = self.device
        target_dtype = self.model_dtype
        self.logger.info(f"Creating weights on device: {target_device} with dtype: {target_dtype}")

        for layer_idx in range(num_layers):
            # Router gate weights
            weights[f"mixlora.layers.{layer_idx}.mlp.moe_gate.weight"] = torch.randn(
                self.args.num_experts, hidden_size, dtype=target_dtype, device=target_device
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
                            self.args.lora_r, in_features, dtype=target_dtype, device=target_device
                        ) * 0.01

                        # LoRA B matrix
                        weights[f"{prefix}.lora_B.weight"] = torch.zeros(
                            out_features, self.args.lora_r, dtype=target_dtype, device=target_device
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
                        self.args.lora_r, in_features, dtype=target_dtype, device=target_device
                    ) * 0.01

                    # LoRA B matrix (rank -> out_features)
                    weights[f"{prefix}.lora_B.weight"] = torch.zeros(
                        out_features, self.args.lora_r, dtype=target_dtype, device=target_device
                    )

                except Exception as e:
                    self.logger.warning(f"Could not determine dimensions for attention module {module_name}: {e}")
                    continue

        return weights

    def _verify_device_consistency(self):
        """Verify that all model components are on the correct device."""
        if self.args.num_gpu > 1:
            self.logger.info("Multi-GPU training: skipping device consistency check (Trainer will handle device placement)")
            return

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
            collate_fn=self.data_collator  # Use original collator for evaluation
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

        # Since we use safe_collator, we don't have access to choice_answers here
        # The real evaluation with choice_answers happens in BestModelTracker
        # Return dummy metrics for trainer compatibility
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

        # Use BF16 if available, otherwise FP32 to avoid gradient scaling issues
        if bf16_available:
            if self.args.num_gpu >= 2:
                self.logger.info("Using BF16 precision for dual-GPU training")
            else:
                self.logger.info("Using BF16 precision for single GPU training")
            use_fp16 = False
            use_bf16 = True
        else:
            if self.args.num_gpu >= 2:
                self.logger.info("Using FP32 precision for dual-GPU training (BF16 not available)")
            else:
                self.logger.info("Using FP32 precision for single GPU training (BF16 not available)")
            use_fp16 = False
            use_bf16 = False

        # Calculate effective batch size based on configured GPU count
        per_device_batch_size = self.args.batch_size
        effective_gpu_count = min(self.args.num_gpu, gpu_count)
        total_batch_size = per_device_batch_size * effective_gpu_count * self.args.gradient_accumulation_steps
        self.logger.info(f"Batch size per device: {per_device_batch_size}")
        self.logger.info(f"Total effective batch size: {total_batch_size} (Configured GPUs: {self.args.num_gpu}, Available: {gpu_count})")

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
            max_grad_norm=1.0,  # Gradient clipping (safe with BF16/FP32)
            # Memory optimization
            save_safetensors=True,  # Use safer tensor format
            optim="adamw_torch",  # Use PyTorch AdamW for better memory
            dataloader_pin_memory=False,
            remove_unused_columns=False,
            report_to=None,  # Disable wandb for now
            # GPU settings - force single device when num_gpu=1
            dataloader_num_workers=0,  # Avoid multiprocessing issues
            dataloader_drop_last=True,  # Ensure even batch distribution
            # Critical fix: prevent DataParallel when using single GPU
            no_cuda=False if torch.cuda.is_available() else True,
            # Force single process for single GPU to avoid DataParallel
            local_rank=-1 if self.args.num_gpu == 1 else None,
        )

        # FINAL FIX: Force n_gpu=1 to completely disable DataParallel
        if self.args.num_gpu == 1:
            training_args.n_gpu = 1
            training_args._n_gpu = 1  # Also set private attribute

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

        # Double check: manually set trainer's n_gpu if needed
        if self.args.num_gpu == 1:
            trainer.args.n_gpu = 1
            trainer.args._n_gpu = 1

        # Add best model tracker callback
        best_model_tracker = BestModelTracker(self)
        trainer.add_callback(best_model_tracker)

        # Store reference for saving adapter weights
        self.trainer = trainer

        # Final device verification before training
        if self.args.num_gpu == 1:
            self.logger.info(f"Single GPU - Model device: {next(self.model.parameters()).device}")
            self.logger.info(f"Training device: {self.device}")
        else:
            self.logger.info(f"Multi-GPU training - Model initial device: {next(self.model.parameters()).device}")
            self.logger.info(f"Trainer will handle device distribution across {self.args.num_gpu} GPUs")

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
    # CRITICAL: Parse num_gpu argument first and set environment immediately
    import sys
    num_gpu = 2  # default
    for i, arg in enumerate(sys.argv):
        if arg == "--num_gpu" and i + 1 < len(sys.argv):
            num_gpu = int(sys.argv[i + 1])
            break

    # Force GPU configuration at the very beginning
    if num_gpu == 1:
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        print(f"[EARLY SETUP] Forced CUDA_VISIBLE_DEVICES=0 for single GPU training")
    elif num_gpu == 2:
        os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
        print(f"[EARLY SETUP] Set CUDA_VISIBLE_DEVICES=0,1 for dual GPU training")

    parser = argparse.ArgumentParser(description="Train MixLoRA on cultural datasets")

    # Dataset configuration
    parser.add_argument("--data_id", type=int, default=2, help="Dataset ID (2=culturalbench, 3=normad)")
    parser.add_argument("--backbone", type=str, default="llama", help="Model backbone (llama, qwen)")
    parser.add_argument("--base_model", type=str,
                       default="/root/autodl-tmp/CultureMoE/Culture_Alignment/Meta-Llama-3.1-8B-Instruct",
                       help="Base model path")
    parser.add_argument("--output_dir", type=str, default="./mixlora_output", help="Output directory")
    parser.add_argument("--num_gpu", type=int, default=2, help="Number of GPUs to use (1=single, 2=dual)")

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