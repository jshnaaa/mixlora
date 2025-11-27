"""
MoE configuration for shared expert architecture.
Based on MixLoRA config but adds shared expert support.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mixlora.config import MixLoraConfig


class MoEConfig(MixLoraConfig):
    """
    Configuration for MoE with shared experts.
    Extends MixLoRA config to support shared experts.
    """

    def __init__(self, *args, **kwargs):
        # Extract shared expert specific parameters
        self.use_shared_expert = kwargs.pop('use_shared_expert', True)

        # Initialize parent MixLoRA config
        super().__init__(*args, **kwargs)

        # Set MoE specific attributes
        self.peft_type = "MOE"

    @classmethod
    def from_config(cls, config_dict):
        """Create MoEConfig from dictionary."""
        # Extract shared expert parameter
        use_shared_expert = config_dict.get('use_shared_expert', True)

        # Create base MixLoRA config
        base_config = super().from_config(config_dict)

        # Create MoE config with shared expert parameter
        moe_config = cls(
            base_model_name_or_path=base_config.base_model_name_or_path,
            task_type=base_config.task_type,
            peft_type="MOE",
            r=base_config.lora_r_,
            lora_alpha=base_config.lora_alpha_,
            lora_dropout=base_config.lora_dropout_,
            target_modules=base_config.target_modules_,
            use_dora=base_config.use_dora_,
            use_rslora=base_config.use_rslora_,
            routing_strategy=base_config.routing_strategy_,
            num_experts=base_config.num_experts_,
            top_k=base_config.top_k_,
            router_aux_loss_coef=base_config.router_aux_loss_coef_,
            router_init_range=base_config.router_init_range_,
            jitter_noise=base_config.jitter_noise_,
            act_fn=base_config.act_fn_,
            use_shared_expert=use_shared_expert
        )

        # Copy other attributes
        if hasattr(base_config, 'dtype_'):
            moe_config.dtype_ = base_config.dtype_

        return moe_config

    def export(self):
        """Export config to dictionary."""
        config_dict = super().export()
        config_dict['use_shared_expert'] = self.use_shared_expert
        config_dict['peft_type'] = "MOE"
        return config_dict