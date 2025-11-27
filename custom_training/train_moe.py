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
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# CRITICAL: Disable automatic DataParallel at the very beginning
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

    # MoE configuration
    num_experts: int = field(default=8, metadata={"help": "Number of experts"})
    top_k: int = field(default=2, metadata={"help": "Top-k experts to activate"})
    routing_strategy: str = field(default="mixlora", metadata={"help": "Routing strategy"})
    router_aux_loss_coef: float = field(default=0.01, metadata={"help": "Router auxiliary loss coefficient"})
    router_init_range: float = field(default=0.02, metadata={"help": "Router initialization range"})
    jitter_noise: float = field(default=0.0, metadata={"help": "Jitter noise for routing"})

    # Shared expert configuration
    use_shared_expert: bool = field(default=True, metadata={"help": "Whether to use shared expert"})

    # LoRA configuration
    lora_r: int = field(default=8, metadata={"help": "LoRA rank"})
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

        # Load base model
        # Note: Don't use device_map="auto" with torchrun/DDP as it conflicts
        self.model = AutoModelForCausalLM.from_pretrained(
            self.args.base_model,
            torch_dtype=torch.float16,
            device_map=None,  # Let DDP handle device placement
            trust_remote_code=True
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

        # Enable gradient computation for LoRA parameters
        for name, param in self.model.named_parameters():
            if any(keyword in name for keyword in ["lora_A", "lora_B", "gate", "shared_expert"]):
                param.requires_grad = True
            else:
                param.requires_grad = False

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
            fp16=True,
            gradient_checkpointing=True,
            dataloader_pin_memory=False,
            remove_unused_columns=False,
            ddp_find_unused_parameters=False,
            ddp_backend="nccl",
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

        # Train model
        trainer.train()

        # Save final model
        trainer.save_model(os.path.join(self.args.output_dir, "final_model"))

        # Evaluate on test set
        test_results = trainer.evaluate(eval_dataset=self.test_dataset)

        # Save test results
        with open(os.path.join(self.args.output_dir, "test_results.json"), "w") as f:
            json.dump(test_results, f, indent=2)

        logger.info("Training completed successfully!")
        return test_results

    def compute_metrics(self, eval_pred):
        """Compute evaluation metrics."""
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)

        # Calculate accuracy
        accuracy = accuracy_score(labels, predictions)

        # Calculate precision, recall, F1
        precision, recall, f1, _ = precision_recall_fscore_support(
            labels, predictions, average='weighted', zero_division=0
        )

        return {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
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