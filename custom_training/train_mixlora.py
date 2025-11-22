"""
Training script for MixLoRA on choice question datasets.
This script handles the full training pipeline including model setup, data loading, and training loop.
"""

import argparse
import json
import logging
import os
import sys
import math
from typing import Dict, Optional, List
from dataclasses import dataclass, field

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    get_linear_schedule_with_warmup
)
from transformers.trainer_utils import get_last_checkpoint
import wandb

# Add parent directory to path to import mixlora
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mixlora.config import MixLoraConfig
from mixlora.model import inject_adapter_in_model
from dataset import ChoiceQuestionDataset, ChoiceQuestionCollator


@dataclass
class MixLoRATrainingArguments:
    """Training arguments specific to MixLoRA."""

    # Model and data
    base_model: str = field(metadata={"help": "Base model name or path"})
    dataset_path: str = field(metadata={"help": "Path to training dataset"})
    validation_dataset_path: Optional[str] = field(default=None, metadata={"help": "Path to validation dataset"})
    output_dir: str = field(default="./mixlora_output", metadata={"help": "Output directory"})

    # MixLoRA specific
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

    # Target modules (for different model architectures)
    target_modules: Optional[List[str]] = field(
        default=None,
        metadata={"help": "Target modules for LoRA. If None, will use default based on model type"}
    )

    # Training parameters
    max_length: int = field(default=512, metadata={"help": "Maximum sequence length"})
    batch_size: int = field(default=4, metadata={"help": "Training batch size"})
    gradient_accumulation_steps: int = field(default=4, metadata={"help": "Gradient accumulation steps"})
    learning_rate: float = field(default=1e-4, metadata={"help": "Learning rate"})
    num_epochs: int = field(default=3, metadata={"help": "Number of training epochs"})
    warmup_ratio: float = field(default=0.1, metadata={"help": "Warmup ratio"})
    weight_decay: float = field(default=0.01, metadata={"help": "Weight decay"})

    # Validation and logging
    validation_split: float = field(default=0.1, metadata={"help": "Validation split ratio if no validation dataset provided"})
    eval_steps: int = field(default=100, metadata={"help": "Evaluation steps"})
    save_steps: int = field(default=500, metadata={"help": "Save steps"})
    logging_steps: int = field(default=10, metadata={"help": "Logging steps"})

    # Choice-specific parameters
    choice_range: Optional[List[str]] = field(default=None, metadata={"help": "Valid choice range (auto-detected if None)"})

    # Experiment tracking
    wandb_project: Optional[str] = field(default=None, metadata={"help": "Weights & Biases project name"})
    run_name: Optional[str] = field(default=None, metadata={"help": "Run name for logging"})


class MixLoRATrainer:
    """Custom trainer for MixLoRA on choice question datasets."""

    def __init__(self, args: MixLoRATrainingArguments):
        self.args = args
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Set up logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

        # Initialize wandb if specified
        if args.wandb_project:
            wandb.init(
                project=args.wandb_project,
                name=args.run_name,
                config=args.__dict__
            )

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

        # Load base model
        self.model = AutoModelForCausalLM.from_pretrained(
            self.args.base_model,
            torch_dtype=torch.float16,
            device_map="auto"
        )

        # Determine target modules
        if self.args.target_modules is None:
            target_modules = self._get_default_target_modules(self.model.config.model_type)
        else:
            target_modules = self.args.target_modules

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
        }

        self.mixlora_config = MixLoraConfig.from_config(mixlora_config_dict)
        self.mixlora_config.dtype_ = torch.float16

        # Initialize MixLoRA weights (this would normally be done during training)
        # For now, we'll create dummy weights to inject the architecture
        dummy_weights = self._create_dummy_weights()

        # Inject MixLoRA adapters
        inject_adapter_in_model(self.model, self.mixlora_config, dummy_weights)

        self.logger.info("MixLoRA adapters injected successfully")

    def _create_dummy_weights(self) -> Dict[str, torch.Tensor]:
        """Create dummy weights for MixLoRA injection."""
        weights = {}

        # Get model dimensions
        hidden_size = self.model.config.hidden_size
        num_layers = self.model.config.num_hidden_layers

        for layer_idx in range(num_layers):
            # Router gate weights
            weights[f"mixlora.layers.{layer_idx}.mlp.moe_gate.weight"] = torch.randn(
                self.args.num_experts, hidden_size, dtype=torch.float16
            ) * self.args.router_init_range

            # Expert LoRA weights for each target module
            for module_name, is_target in self.mixlora_config.target_modules_.items():
                if not is_target:
                    continue

                # Get the actual layer to determine dimensions
                try:
                    if module_name in ["gate_proj", "up_proj"]:
                        in_features = hidden_size
                        out_features = self.model.config.intermediate_size
                    elif module_name == "down_proj":
                        in_features = self.model.config.intermediate_size
                        out_features = hidden_size
                    elif module_name in ["q_proj", "k_proj", "v_proj", "o_proj"]:
                        in_features = hidden_size
                        out_features = hidden_size
                    elif module_name == "fc1":
                        in_features = hidden_size
                        out_features = self.model.config.intermediate_size
                    elif module_name == "fc2":
                        in_features = self.model.config.intermediate_size
                        out_features = hidden_size
                    else:
                        # Default dimensions
                        in_features = hidden_size
                        out_features = hidden_size

                    # Create LoRA weights for each expert
                    for expert_idx in range(self.args.num_experts):
                        prefix = f"mixlora.layers.{layer_idx}.mlp.{module_name}.experts.{expert_idx}"

                        # LoRA A matrix (in_features -> rank)
                        weights[f"{prefix}.lora_A.weight"] = torch.randn(
                            self.args.lora_r, in_features, dtype=torch.float16
                        ) * 0.01

                        # LoRA B matrix (rank -> out_features)
                        weights[f"{prefix}.lora_B.weight"] = torch.zeros(
                            out_features, self.args.lora_r, dtype=torch.float16
                        )

                except Exception as e:
                    self.logger.warning(f"Could not determine dimensions for {module_name}: {e}")
                    continue

            # Attention LoRA weights
            for module_name, is_target in self.mixlora_config.target_modules_.items():
                if not is_target or module_name in ["gate_proj", "up_proj", "down_proj", "fc1", "fc2"]:
                    continue

                prefix = f"mixlora.layers.{layer_idx}.self_attn.{module_name}"

                # LoRA A matrix
                weights[f"{prefix}.lora_A.weight"] = torch.randn(
                    self.args.lora_r, hidden_size, dtype=torch.float16
                ) * 0.01

                # LoRA B matrix
                weights[f"{prefix}.lora_B.weight"] = torch.zeros(
                    hidden_size, self.args.lora_r, dtype=torch.float16
                )

        return weights

    def setup_datasets(self):
        """Setup training and validation datasets."""
        self.logger.info("Loading datasets...")

        # Load training dataset
        self.train_dataset = ChoiceQuestionDataset(
            data_path=self.args.dataset_path,
            tokenizer=self.tokenizer,
            max_length=self.args.max_length,
            choice_range=self.args.choice_range
        )

        # Setup validation dataset
        if self.args.validation_dataset_path:
            self.eval_dataset = ChoiceQuestionDataset(
                data_path=self.args.validation_dataset_path,
                tokenizer=self.tokenizer,
                max_length=self.args.max_length,
                choice_range=self.train_dataset.choice_range
            )
        else:
            # Split training dataset
            train_size = int((1 - self.args.validation_split) * len(self.train_dataset))
            eval_size = len(self.train_dataset) - train_size
            self.train_dataset, self.eval_dataset = random_split(
                self.train_dataset, [train_size, eval_size]
            )

        self.logger.info(f"Training samples: {len(self.train_dataset)}")
        self.logger.info(f"Validation samples: {len(self.eval_dataset)}")

        # Setup data collator
        self.data_collator = ChoiceQuestionCollator(tokenizer=self.tokenizer)

    def train(self):
        """Run the training loop."""
        self.logger.info("Starting training...")

        # Setup training arguments
        training_args = TrainingArguments(
            output_dir=self.args.output_dir,
            per_device_train_batch_size=self.args.batch_size,
            per_device_eval_batch_size=self.args.batch_size,
            gradient_accumulation_steps=self.args.gradient_accumulation_steps,
            learning_rate=self.args.learning_rate,
            num_train_epochs=self.args.num_epochs,
            warmup_ratio=self.args.warmup_ratio,
            weight_decay=self.args.weight_decay,
            logging_steps=self.args.logging_steps,
            eval_steps=self.args.eval_steps,
            save_steps=self.args.save_steps,
            evaluation_strategy="steps",
            save_strategy="steps",
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            report_to="wandb" if self.args.wandb_project else None,
            run_name=self.args.run_name,
            fp16=True,
            dataloader_pin_memory=False,
            remove_unused_columns=False,
        )

        # Create trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.eval_dataset,
            data_collator=self.data_collator,
            tokenizer=self.tokenizer,
        )

        # Check for existing checkpoint
        checkpoint = None
        if os.path.isdir(self.args.output_dir):
            checkpoint = get_last_checkpoint(self.args.output_dir)

        # Train
        trainer.train(resume_from_checkpoint=checkpoint)

        # Save final model
        trainer.save_model()

        # Save MixLoRA config
        config_path = os.path.join(self.args.output_dir, "adapter_config.json")
        with open(config_path, 'w') as f:
            json.dump(self.mixlora_config.export(), f, indent=2)

        self.logger.info(f"Training completed. Model saved to {self.args.output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Train MixLoRA on choice question datasets")

    # Add arguments from MixLoRATrainingArguments
    parser.add_argument("--base_model", type=str, required=True, help="Base model name or path")
    parser.add_argument("--dataset_path", type=str, required=True, help="Path to training dataset")
    parser.add_argument("--validation_dataset_path", type=str, help="Path to validation dataset")
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

    # Validation and logging
    parser.add_argument("--validation_split", type=float, default=0.1, help="Validation split ratio")
    parser.add_argument("--eval_steps", type=int, default=100, help="Evaluation steps")
    parser.add_argument("--save_steps", type=int, default=500, help="Save steps")
    parser.add_argument("--logging_steps", type=int, default=10, help="Logging steps")

    # Experiment tracking
    parser.add_argument("--wandb_project", type=str, help="Weights & Biases project name")
    parser.add_argument("--run_name", type=str, help="Run name for logging")

    args = parser.parse_args()

    # Convert to dataclass
    training_args = MixLoRATrainingArguments(**vars(args))

    # Create trainer and run training
    trainer = MixLoRATrainer(training_args)
    trainer.setup_model_and_tokenizer()
    trainer.setup_datasets()
    trainer.train()


if __name__ == "__main__":
    main()