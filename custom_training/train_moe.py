"""
Custom MoE training script with shared expert support for cultural datasets.
Based on MixLoRA training but adds shared expert functionality.
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
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True,max_split_size_mb:128"
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"  # For debugging
os.environ["TORCH_USE_CUDA_DSA"] = "1"     # Additional debugging

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

# Add parent directory to path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mixlora.config import MixLoraConfig
from mixlora.model import load_adapter_weights
from moe.moe_config import MoEConfig
from moe.moe_model import inject_moe_adapter_in_model
from custom_training.dataset import ChoiceQuestionDataset, ChoiceQuestionCollator


# Configure logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


@dataclass
class MoETrainingArguments:
    """Custom training arguments for MoE with shared experts."""

    # Data arguments
    data_id: int = field(default=2, metadata={"help": "Dataset ID (0-4)"})
    backbone: str = field(default="llama", metadata={"help": "Model backbone (llama/qwen)"})
    base_model: str = field(default="", metadata={"help": "Path to base model"})
    pretrained_lora_path: Optional[str] = field(default=None, metadata={"help": "Path to pretrained LoRA weights"})

    # MoE configuration (optimized for memory efficiency)
    num_experts: int = field(default=2, metadata={"help": "Number of experts"})
    top_k: int = field(default=2, metadata={"help": "Top-k experts to activate"})
    routing_strategy: str = field(default="mixlora", metadata={"help": "Routing strategy"})
    router_aux_loss_coef: float = field(default=0.01, metadata={"help": "Router auxiliary loss coefficient"})
    router_init_range: float = field(default=0.02, metadata={"help": "Router initialization range"})
    jitter_noise: float = field(default=0.0, metadata={"help": "Jitter noise for routing"})

    # Shared expert configuration
    use_shared_expert: bool = field(default=True, metadata={"help": "Whether to use shared expert"})

    # Parameter freezing configuration
    train_moe_only: bool = field(default=True, metadata={"help": "Only train MoE components (router + shared experts + routing experts)"})

    # LoRA configuration (optimized for memory efficiency)
    lora_r: int = field(default=2, metadata={"help": "LoRA rank"})
    lora_alpha: int = field(default=16, metadata={"help": "LoRA alpha"})
    lora_dropout: float = field(default=0.05, metadata={"help": "LoRA dropout"})
    use_dora: bool = field(default=False, metadata={"help": "Use DoRA"})
    use_rslora: bool = field(default=False, metadata={"help": "Use RSLoRA"})

    # Training configuration
    output_dir: str = field(default="./moe_output", metadata={"help": "Output directory"})
    max_length: int = field(default=512, metadata={"help": "Maximum sequence length"})
    batch_size: int = field(default=1, metadata={"help": "Training batch size"})
    gradient_accumulation_steps: int = field(default=16, metadata={"help": "Gradient accumulation steps"})
    learning_rate: float = field(default=1e-4, metadata={"help": "Learning rate"})
    num_epochs: int = field(default=3, metadata={"help": "Number of training epochs"})
    warmup_ratio: float = field(default=0.1, metadata={"help": "Warmup ratio"})
    weight_decay: float = field(default=0.01, metadata={"help": "Weight decay"})

    # Evaluation and saving
    eval_interval: int = field(default=1, metadata={"help": "Evaluation interval (epochs)"})
    save_steps: int = field(default=500, metadata={"help": "Save steps"})
    logging_steps: int = field(default=10, metadata={"help": "Logging steps"})

    # GPU configuration
    num_gpu: int = field(default=2, metadata={"help": "Number of GPUs"})

    # Experiment tracking
    wandb_project: Optional[str] = field(default=None, metadata={"help": "Wandb project name"})
    run_name: str = field(default="", metadata={"help": "Run name"})


class BestMoEModelTracker(TrainerCallback):
    """Callback to track and save the best MoE model based on validation accuracy."""

    def __init__(self, trainer_instance):
        self.best_accuracy = 0.0
        self.best_model_path = None
        self.trainer_instance = trainer_instance
        self.eval_results = []

    def on_evaluate(self, args, state, control, model, eval_dataloader, **kwargs):
        """Called after evaluation."""
        logger.info("Performing custom MoE evaluation with choice answer extraction...")

        # Get validation metrics using detailed evaluation
        val_metrics, val_results = self.trainer_instance.evaluate_on_dataset(
            self.trainer_instance.eval_dataset, "Validation"
        )
        current_accuracy = val_metrics['accuracy']

        # Save evaluation result
        eval_result = {
            'epoch': state.epoch,
            'step': state.global_step,
            'eval_accuracy': current_accuracy,
            'eval_precision': val_metrics.get('precision', 0.0),
            'eval_recall': val_metrics.get('recall', 0.0),
            'eval_f1': val_metrics.get('f1', 0.0),
        }
        self.eval_results.append(eval_result)

        # Log metrics for trainer (this will be used for best model selection)
        kwargs['logs'] = kwargs.get('logs', {})
        kwargs['logs']['eval_accuracy'] = current_accuracy
        kwargs['logs']['eval_precision'] = val_metrics.get('precision', 0.0)
        kwargs['logs']['eval_recall'] = val_metrics.get('recall', 0.0)
        kwargs['logs']['eval_f1'] = val_metrics.get('f1', 0.0)

        # Check if this is the best model so far
        if current_accuracy > self.best_accuracy:
            self.best_accuracy = current_accuracy

            # Save the best model (only adapter weights)
            best_model_dir = os.path.join(args.output_dir, "best_model")
            os.makedirs(best_model_dir, exist_ok=True)

            # Save adapter weights
            self.trainer_instance.save_adapter_weights(best_model_dir)
            self.best_model_path = best_model_dir

            logger.info(f"New best MoE model saved with accuracy: {current_accuracy:.4f}")

        # Save evaluation results
        eval_results_path = os.path.join(args.output_dir, "validation_results.json")
        with open(eval_results_path, 'w') as f:
            json.dump(self.eval_results, f, indent=2)


class MoETrainer:
    """MoE trainer with shared expert support."""

    def __init__(self, args: MoETrainingArguments):
        self.args = args
        self.model = None
        self.tokenizer = None
        self.train_dataset = None
        self.eval_dataset = None
        self.test_dataset = None

        # Setup logging
        self.setup_logging()

        # Get dataset configuration
        self.dataset_config = self._get_dataset_config()

    def setup_logging(self):
        """Setup logging configuration."""
        logging.basicConfig(
            format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
            datefmt="%m/%d/%Y %H:%M:%S",
            level=logging.INFO,
        )

    def _get_dataset_config(self) -> Dict[str, str]:
        """Get dataset configuration based on data_id."""
        configs = {
            0: {
                "path": "/root/autodl-fs/unified_all_datasets_small_merge_gen.json",
                "tag": "unified_small",
                "name": "unified_all_datasets_small"
            },
            1: {
                "path": "/root/autodl-fs/unified_all_datasets_merge_gen.json",
                "tag": "unified",
                "name": "unified_all_datasets"
            },
            2: {
                "path": "/root/autodl-fs/CulturalBench_merge_gen.json",
                "tag": "CulturalBench",
                "name": "CulturalBench"
            },
            3: {
                "path": "/root/autodl-fs/normad_merge_gen.json",
                "tag": "normad",
                "name": "normad"
            },
            4: {
                "path": "/root/autodl-fs/cultureLLM_merge_gen.json",
                "tag": "cultureLLM",
                "name": "cultureLLM"
            }
        }

        if self.args.data_id not in configs:
            raise ValueError(f"Unsupported data_id: {self.args.data_id}. Supported values: {list(configs.keys())}")

        return configs[self.args.data_id]

    def setup_model_and_tokenizer(self):
        """Setup model and tokenizer."""
        logger.info(f"Loading base model: {self.args.base_model}")

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.args.base_model,
            trust_remote_code=True,
            use_fast=False
        )

        # Set pad token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Load base model with memory optimization
        # Determine model dtype
        if torch.cuda.is_bf16_supported():
            model_dtype = torch.bfloat16
            logger.info("Using bfloat16 for model weights")
        else:
            model_dtype = torch.float16
            logger.info("Using float16 for model weights")

        # Determine target device for model loading
        if "LOCAL_RANK" in os.environ:
            local_rank = int(os.environ["LOCAL_RANK"])
            target_device = f"cuda:{local_rank}"
            logger.info(f"DDP mode: loading model on {target_device}")
            # For DDP, don't use device_map as it conflicts
            self.model = AutoModelForCausalLM.from_pretrained(
                self.args.base_model,
                torch_dtype=model_dtype,
                trust_remote_code=True,
                low_cpu_mem_usage=True,
                device_map=None,  # Let DDP handle device placement
            )
        else:
            target_device = "cuda:0" if torch.cuda.is_available() else "cpu"
            logger.info(f"Single GPU mode: loading model on {target_device}")
            # For single GPU, use memory optimization
            self.model = AutoModelForCausalLM.from_pretrained(
                self.args.base_model,
                torch_dtype=model_dtype,
                trust_remote_code=True,
                low_cpu_mem_usage=True,
                device_map={"": target_device},
                max_memory={target_device: "46GiB"},
                offload_folder="./cpu_offload",
            )

        # Create MoE config
        moe_config = self._create_moe_config()

        # Create dummy weights for initialization
        weights = self._create_dummy_weights(moe_config)

        # Load pretrained LoRA weights if provided
        if self.args.pretrained_lora_path and os.path.exists(self.args.pretrained_lora_path):
            logger.info(f"Loading pretrained LoRA weights from: {self.args.pretrained_lora_path}")
            pretrained_weights = load_adapter_weights(self.args.pretrained_lora_path)
            # Merge pretrained weights with dummy weights
            weights.update(pretrained_weights)

        # Inject MoE adapters
        logger.info("Injecting MoE adapters with shared expert support...")
        self.model = inject_moe_adapter_in_model(self.model, moe_config, weights)

        # Configure parameter freezing based on training mode
        self._freeze_parameters()

        # Ensure model is on correct device for DDP
        if torch.cuda.is_available():
            local_rank = int(os.environ.get("LOCAL_RANK", 0))
            device = torch.device(f"cuda:{local_rank}")
            self.model = self.model.to(device)
            logger.info(f"Model moved to device: {device}")

        logger.info("Model setup completed successfully")

    def _create_moe_config(self) -> MoEConfig:
        """Create MoE configuration."""
        # Determine target modules based on backbone
        if self.args.backbone in ["llama", "qwen"]:
            target_modules = {
                "gate_proj": True,
                "up_proj": True,
                "down_proj": True,
                "q_proj": False,  # No MoE for attention
                "k_proj": False,
                "v_proj": False,
                "o_proj": False,
            }
        elif self.args.backbone == "phi":
            target_modules = {
                "fc1": True,
                "fc2": True,
                "q_proj": False,
                "k_proj": False,
                "v_proj": False,
                "o_proj": False,
            }
        else:
            # Default to LLaMA style
            target_modules = {
                "gate_proj": True,
                "up_proj": True,
                "down_proj": True,
                "q_proj": False,
                "k_proj": False,
                "v_proj": False,
                "o_proj": False,
            }

        config = MoEConfig(
            base_model_name_or_path=self.args.base_model,
            task_type="CAUSAL_LM",
            r=self.args.lora_r,
            lora_alpha=self.args.lora_alpha,
            lora_dropout=self.args.lora_dropout,
            target_modules=target_modules,
            use_dora=self.args.use_dora,
            use_rslora=self.args.use_rslora,
            routing_strategy=self.args.routing_strategy,
            num_experts=self.args.num_experts,
            top_k=self.args.top_k,
            router_aux_loss_coef=self.args.router_aux_loss_coef,
            router_init_range=self.args.router_init_range,
            jitter_noise=self.args.jitter_noise,
            use_shared_expert=self.args.use_shared_expert
        )

        return config

    def _create_dummy_weights(self, config: MoEConfig) -> Dict[str, torch.Tensor]:
        """Create dummy weights for MoE initialization."""
        weights = {}

        # Get model dimensions
        hidden_size = self.model.config.hidden_size
        intermediate_size = getattr(self.model.config, 'intermediate_size', 4 * hidden_size)

        # Create weights for each layer
        for layer_idx in range(self.model.config.num_hidden_layers):
            # Router gate weights
            weights[f"mixlora.layers.{layer_idx}.mlp.moe_gate.weight"] = torch.randn(
                config.num_experts_, hidden_size, dtype=torch.float16
            ) * config.router_init_range_

            # Expert weights for each projection
            target_modules = config.target_modules_

            if self.args.backbone in ["llama", "qwen"]:
                proj_configs = [
                    ("gate_proj", hidden_size, intermediate_size),
                    ("up_proj", hidden_size, intermediate_size),
                    ("down_proj", intermediate_size, hidden_size),
                ]
            elif self.args.backbone == "phi":
                proj_configs = [
                    ("fc1", hidden_size, intermediate_size),
                    ("fc2", intermediate_size, hidden_size),
                ]
            else:
                proj_configs = [
                    ("gate_proj", hidden_size, intermediate_size),
                    ("up_proj", hidden_size, intermediate_size),
                    ("down_proj", intermediate_size, hidden_size),
                ]

            for proj_name, in_features, out_features in proj_configs:
                if target_modules.get(proj_name, False):
                    # Expert LoRA weights
                    for expert_idx in range(config.num_experts_):
                        # LoRA A matrix
                        weights[f"mixlora.layers.{layer_idx}.mlp.experts.{expert_idx}.{proj_name}.lora_A.weight"] = torch.randn(
                            config.lora_r_, in_features, dtype=torch.float16
                        ) * 0.01

                        # LoRA B matrix (initialized to zero)
                        weights[f"mixlora.layers.{layer_idx}.mlp.experts.{expert_idx}.{proj_name}.lora_B.weight"] = torch.zeros(
                            out_features, config.lora_r_, dtype=torch.float16
                        )

                    # Shared expert weights (same initialization as experts)
                    if config.use_shared_expert:
                        weights[f"mixlora.layers.{layer_idx}.mlp.shared_expert.{proj_name}.lora_A.weight"] = torch.randn(
                            config.lora_r_, in_features, dtype=torch.float16
                        ) * 0.01

                        weights[f"mixlora.layers.{layer_idx}.mlp.shared_expert.{proj_name}.lora_B.weight"] = torch.zeros(
                            out_features, config.lora_r_, dtype=torch.float16
                        )

        return weights

    def _freeze_parameters(self):
        """Freeze specified parameters based on training mode."""
        frozen_count = 0
        trainable_count = 0

        if self.args.train_moe_only:
            logger.info("🔒 Training MoE components only (router + shared experts + routing experts), freezing all others")

            for name, param in self.model.named_parameters():
                # Only train MoE components: router, routing experts LoRA, shared experts LoRA
                # Keep trainable: moe_gate (router), experts.*.lora_A, experts.*.lora_B, shared_expert.*.lora_A, shared_expert.*.lora_B
                is_moe_component = (
                    ('moe_gate' in name) or  # Router
                    ('experts' in name and any(lora_part in name for lora_part in ['lora_A', 'lora_B'])) or  # Routing experts LoRA
                    ('shared_expert' in name and any(lora_part in name for lora_part in ['lora_A', 'lora_B']))  # Shared experts LoRA
                )

                if is_moe_component:
                    param.requires_grad = True
                    trainable_count += 1
                    logger.debug(f"  ✓ Trainable: {name}")
                else:
                    param.requires_grad = False
                    frozen_count += 1
                    logger.debug(f"  ❄️  Frozen: {name}")
        else:
            logger.info("🔓 Training all LoRA parameters and MoE components")

            for name, param in self.model.named_parameters():
                # Train all LoRA and MoE components
                if any(keyword in name for keyword in ["lora_A", "lora_B", "moe_gate", "shared_expert"]):
                    param.requires_grad = True
                    trainable_count += 1
                else:
                    param.requires_grad = False
                    frozen_count += 1

        logger.info(f"📊 Parameter status: {trainable_count} trainable, {frozen_count} frozen")

        # Log trainable parameters for verification
        trainable_params = []
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                trainable_params.append(name)

        if trainable_params:
            logger.info("🎯 Trainable parameters:")
            for param_name in trainable_params[:10]:  # Show first 10
                logger.info(f"  ✓ {param_name}")
            if len(trainable_params) > 10:
                logger.info(f"  ... and {len(trainable_params) - 10} more")

        # Calculate memory savings
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params_count = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        frozen_params_count = total_params - trainable_params_count

        logger.info(f"💾 Memory savings: {frozen_params_count:,} / {total_params:,} parameters frozen ({frozen_params_count/total_params*100:.1f}%)")

        return trainable_count, frozen_count

    def save_adapter_weights(self, save_path: str):
        """Save only the adapter weights (not the full model)."""
        os.makedirs(save_path, exist_ok=True)

        # Extract adapter weights from the model
        adapter_weights = {}

        for name, param in self.model.named_parameters():
            if any(adapter_name in name for adapter_name in ['lora_A', 'lora_B', 'moe_gate', 'experts', 'shared_expert']):
                adapter_weights[name] = param.data.clone()

        # Save adapter weights
        torch.save(adapter_weights, os.path.join(save_path, "adapter_model.bin"))

        # Save adapter config (create MoE config)
        config_dict = {
            "base_model_name_or_path": self.args.base_model,
            "task_type": "CAUSAL_LM",
            "peft_type": "MOE",
            "r": self.args.lora_r,
            "lora_alpha": self.args.lora_alpha,
            "lora_dropout": self.args.lora_dropout,
            "use_dora": self.args.use_dora,
            "use_rslora": self.args.use_rslora,
            "routing_strategy": self.args.routing_strategy,
            "num_experts": self.args.num_experts,
            "top_k": self.args.top_k,
            "router_aux_loss_coef": self.args.router_aux_loss_coef,
            "router_init_range": self.args.router_init_range,
            "jitter_noise": self.args.jitter_noise,
            "use_shared_expert": self.args.use_shared_expert,
        }

        config_path = os.path.join(save_path, "adapter_config.json")
        with open(config_path, 'w') as f:
            json.dump(config_dict, f, indent=2)

        logger.info(f"MoE adapter weights saved to {save_path}")

    def setup_datasets(self):
        """Setup datasets for training."""
        logger.info(f"Loading dataset with ID: {self.args.data_id}")
        logger.info(f"Dataset path: {self.dataset_config['path']}")

        # Create dataset
        dataset = ChoiceQuestionDataset(
            data_path=self.dataset_config['path'],
            tokenizer=self.tokenizer,
            max_length=self.args.max_length,
            choice_range=None  # Auto-detect
        )

        # Split dataset (8:1:1 for train:val:test)
        total_size = len(dataset)
        train_size = int(0.8 * total_size)
        val_size = int(0.1 * total_size)
        test_size = total_size - train_size - val_size

        self.train_dataset, self.eval_dataset, self.test_dataset = random_split(
            dataset, [train_size, val_size, test_size],
            generator=torch.Generator().manual_seed(42)
        )

        logger.info(f"Dataset split - Train: {len(self.train_dataset)}, Val: {len(self.eval_dataset)}, Test: {len(self.test_dataset)}")

        # Get choice range for later use
        self.choice_range = dataset.choice_range
        logger.info(f"Detected choice range: {self.choice_range}")

    def extract_answer_from_generation(self, generated_text: str) -> Optional[str]:
        """Extract answer choice from generated text."""
        # Clean the text
        generated_text = generated_text.strip()

        # First try exact match with choice range
        for choice in self.choice_range:
            if generated_text == choice:
                return choice

        # Try to find digits in the text
        import re
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
            collate_fn=ChoiceQuestionCollator(self.tokenizer)
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
                    ).to(self.model.device)

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
        from sklearn.metrics import accuracy_score, precision_recall_fscore_support

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

        logger.info(f"{dataset_name} Results - Accuracy: {accuracy:.4f}, F1: {f1:.4f}")

        return metrics, detailed_results

    def train(self):
        """Train the MoE model."""
        logger.info("Starting MoE training...")

        # Setup training arguments
        training_args = TrainingArguments(
            output_dir=self.args.output_dir,
            num_train_epochs=self.args.num_epochs,
            per_device_train_batch_size=self.args.batch_size,
            per_device_eval_batch_size=self.args.batch_size,
            gradient_accumulation_steps=self.args.gradient_accumulation_steps,
            learning_rate=self.args.learning_rate,
            weight_decay=self.args.weight_decay,
            warmup_ratio=self.args.warmup_ratio,
            logging_steps=self.args.logging_steps,
            evaluation_strategy="epoch" if self.args.eval_interval == 1 else "steps",
            eval_steps=self.args.save_steps if self.args.eval_interval != 1 else None,
            save_steps=self.args.save_steps,
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="eval_accuracy",
            greater_is_better=True,
            report_to=["wandb"] if self.args.wandb_project else [],
            run_name=self.args.run_name,
            # Memory optimization settings (same as train_mixlora_custom.py)
            bf16=torch.cuda.is_bf16_supported(),
            fp16=not torch.cuda.is_bf16_supported(),
            gradient_checkpointing=True,
            dataloader_pin_memory=False,
            remove_unused_columns=False,
            prediction_loss_only=True,
            skip_memory_metrics=True,
            eval_accumulation_steps=1,
            optim="adamw_8bit",
            save_safetensors=True,
            dataloader_num_workers=0,
            max_grad_norm=1.0,
            ddp_find_unused_parameters=False,
            ddp_backend="nccl",
            ddp_bucket_cap_mb=25,
            ddp_broadcast_buffers=False,
        )

        # Create data collator
        data_collator = ChoiceQuestionCollator(self.tokenizer)

        # Create trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.eval_dataset,
            data_collator=data_collator,
            compute_metrics=self.compute_metrics,
        )

        # Add best model tracker callback
        best_model_tracker = BestMoEModelTracker(self)
        trainer.add_callback(best_model_tracker)

        # Store reference for saving adapter weights
        self.trainer = trainer

        # Aggressive memory cleanup before training
        import gc
        gc.collect()
        torch.cuda.empty_cache()
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                with torch.cuda.device(i):
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
                    torch.cuda.reset_peak_memory_stats(i)

        # Log memory usage before training
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                allocated = torch.cuda.memory_allocated(i) / 1024**3
                cached = torch.cuda.memory_reserved(i) / 1024**3
                total = torch.cuda.get_device_properties(i).total_memory / 1024**3
                logger.info(f"GPU {i} Memory - Allocated: {allocated:.2f}GB, Cached: {cached:.2f}GB, Total: {total:.2f}GB")

        # Train model
        trainer.train()

        # Save final training config
        config_path = os.path.join(self.args.output_dir, "training_config.json")
        with open(config_path, 'w') as f:
            json.dump(vars(self.args), f, indent=2)

        # Load best model and evaluate on test set
        logger.info("Loading best model for test evaluation...")

        if best_model_tracker.best_model_path:
            # Load the best adapter weights
            best_adapter_path = os.path.join(best_model_tracker.best_model_path, "adapter_model.bin")
            if os.path.exists(best_adapter_path):
                adapter_weights = torch.load(best_adapter_path, map_location=self.model.device)

                # Load adapter weights into model
                missing_keys, unexpected_keys = self.model.load_state_dict(adapter_weights, strict=False)
                logger.info(f"Loaded best MoE model adapter weights")

                # Evaluate on test set
                test_metrics, test_results = self.evaluate_on_dataset(self.test_dataset, "Test")

                # Save test results
                test_results_path = os.path.join(self.args.output_dir, "test_results.json")
                with open(test_results_path, 'w') as f:
                    json.dump({
                        'metrics': test_metrics,
                        'detailed_results': test_results
                    }, f, indent=2)

                # Save generated answers for validation set
                val_metrics, val_results = self.evaluate_on_dataset(self.eval_dataset, "Validation")
                generated_answers_path = os.path.join(self.args.output_dir, "generated_answers.json")
                with open(generated_answers_path, 'w') as f:
                    json.dump(val_results, f, indent=2)

                logger.info(f"Final test accuracy: {test_metrics['accuracy']:.4f}")
                logger.info(f"Final test precision: {test_metrics['precision']:.4f}")
                logger.info(f"Final test recall: {test_metrics['recall']:.4f}")
                logger.info(f"Final test F1: {test_metrics['f1']:.4f}")
                logger.info(f"Results saved to {self.args.output_dir}")

                return test_metrics
            else:
                logger.error("Best model adapter weights not found!")
                return {}
        else:
            logger.error("No best model was saved during training!")
            return {}

    def compute_metrics(self, eval_pred):
        """Compute evaluation metrics for Trainer (not used in our custom evaluation)."""
        # This method is required by Trainer but not used in our custom evaluation
        # Our custom evaluation is handled by BestMoEModelTracker callback
        predictions, labels = eval_pred

        # For compatibility, return dummy metrics since we use custom evaluation
        return {
            "eval_accuracy": 0.0,  # Will be overridden by callback
            "eval_precision": 0.0,
            "eval_recall": 0.0,
            "eval_f1": 0.0,
        }


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description="MoE Training with Shared Expert Support")

    # Add all arguments from MoETrainingArguments
    parser.add_argument("--data_id", type=int, default=2, help="Dataset ID (0-4)")
    parser.add_argument("--backbone", type=str, default="llama", help="Model backbone (llama/qwen)")
    parser.add_argument("--base_model", type=str, required=True, help="Path to base model")
    parser.add_argument("--pretrained_lora_path", type=str, default=None, help="Path to pretrained LoRA weights")

    # MoE configuration
    parser.add_argument("--num_experts", type=int, default=8, help="Number of experts")
    parser.add_argument("--top_k", type=int, default=2, help="Top-k experts to activate")
    parser.add_argument("--routing_strategy", type=str, default="mixlora", help="Routing strategy")
    parser.add_argument("--router_aux_loss_coef", type=float, default=0.01, help="Router auxiliary loss coefficient")
    parser.add_argument("--router_init_range", type=float, default=0.02, help="Router initialization range")
    parser.add_argument("--jitter_noise", type=float, default=0.0, help="Jitter noise for routing")

    # Shared expert configuration
    parser.add_argument("--use_shared_expert", type=lambda x: x.lower() in ['true', '1', 'yes'], default=True, help="Whether to use shared expert")

    # Parameter freezing configuration
    parser.add_argument("--train_moe_only", action="store_true", help="Only train MoE components (router + shared experts + routing experts)")

    # LoRA configuration
    parser.add_argument("--lora_r", type=int, default=8, help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=16, help="LoRA alpha")
    parser.add_argument("--lora_dropout", type=float, default=0.05, help="LoRA dropout")
    parser.add_argument("--use_dora", action="store_true", help="Use DoRA")
    parser.add_argument("--use_rslora", action="store_true", help="Use RSLoRA")

    # Training configuration
    parser.add_argument("--output_dir", type=str, default="./moe_output", help="Output directory")
    parser.add_argument("--max_length", type=int, default=512, help="Maximum sequence length")
    parser.add_argument("--batch_size", type=int, default=4, help="Training batch size")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8, help="Gradient accumulation steps")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--num_epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--warmup_ratio", type=float, default=0.1, help="Warmup ratio")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay")

    # Evaluation and saving
    parser.add_argument("--eval_interval", type=int, default=1, help="Evaluation interval (epochs)")
    parser.add_argument("--save_steps", type=int, default=500, help="Save steps")
    parser.add_argument("--logging_steps", type=int, default=10, help="Logging steps")

    # GPU configuration
    parser.add_argument("--num_gpu", type=int, default=2, help="Number of GPUs")

    # Experiment tracking
    parser.add_argument("--wandb_project", type=str, default=None, help="Wandb project name")
    parser.add_argument("--run_name", type=str, default="", help="Run name")

    args = parser.parse_args()

    # Create training arguments object
    training_args = MoETrainingArguments(**vars(args))

    # Create and run trainer
    trainer = MoETrainer(training_args)

    # Setup model and datasets
    trainer.setup_model_and_tokenizer()
    trainer.setup_datasets()

    # Train model
    results = trainer.train()

    print("Training completed!")
    print(f"Test results: {results}")


if __name__ == "__main__":
    main()