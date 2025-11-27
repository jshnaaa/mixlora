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

        # Map standard parameter names to internal names
        if 'base_model_name_or_path' in kwargs:
            kwargs['base_model_'] = kwargs.pop('base_model_name_or_path')
        if 'task_type' in kwargs:
            kwargs['task_type_'] = kwargs.pop('task_type')
        if 'r' in kwargs:
            kwargs['lora_r_'] = kwargs.pop('r')
        if 'lora_alpha' in kwargs:
            kwargs['lora_alpha_'] = kwargs.pop('lora_alpha')
        if 'lora_dropout' in kwargs:
            kwargs['lora_dropout_'] = kwargs.pop('lora_dropout')
        if 'target_modules' in kwargs:
            kwargs['target_modules_'] = kwargs.pop('target_modules')
        if 'use_dora' in kwargs:
            kwargs['use_dora_'] = kwargs.pop('use_dora')
        if 'use_rslora' in kwargs:
            kwargs['use_rslora_'] = kwargs.pop('use_rslora')
        if 'routing_strategy' in kwargs:
            kwargs['routing_strategy_'] = kwargs.pop('routing_strategy')
        if 'num_experts' in kwargs:
            kwargs['num_experts_'] = kwargs.pop('num_experts')
        if 'top_k' in kwargs:
            kwargs['top_k_'] = kwargs.pop('top_k')
        if 'router_aux_loss_coef' in kwargs:
            kwargs['router_aux_loss_coef_'] = kwargs.pop('router_aux_loss_coef')
        if 'router_init_range' in kwargs:
            kwargs['router_init_range_'] = kwargs.pop('router_init_range')
        if 'jitter_noise' in kwargs:
            kwargs['jitter_noise_'] = kwargs.pop('jitter_noise')
        if 'act_fn' in kwargs:
            kwargs['act_fn_'] = kwargs.pop('act_fn')

        # Initialize parent MixLoRA config
        super().__init__(*args, **kwargs)

        # Set MoE specific attributes
        self.peft_type_ = "MOE"

    @classmethod
    def from_config(cls, config_dict):
        """Create MoEConfig from dictionary."""
        # Extract shared expert parameter
        use_shared_expert = config_dict.get('use_shared_expert', True)

        # Create MoE config directly with proper parameter mapping
        moe_config = cls(
            base_model_name_or_path=config_dict.get('base_model_name_or_path', ''),
            task_type=config_dict.get('task_type', 'CAUSAL_LM'),
            r=config_dict.get('lora_r', 8),
            lora_alpha=config_dict.get('lora_alpha', 16),
            lora_dropout=config_dict.get('lora_dropout', 0.05),
            target_modules=config_dict.get('target_modules', {}),
            use_dora=config_dict.get('use_dora', False),
            use_rslora=config_dict.get('use_rslora', False),
            routing_strategy=config_dict.get('routing_strategy', 'mixlora'),
            num_experts=config_dict.get('num_experts', 8),
            top_k=config_dict.get('top_k', 2),
            router_aux_loss_coef=config_dict.get('router_aux_loss_coef', 0.01),
            router_init_range=config_dict.get('router_init_range', 0.02),
            jitter_noise=config_dict.get('jitter_noise', 0.0),
            act_fn=config_dict.get('act_fn', None),
            use_shared_expert=use_shared_expert
        )

        return moe_config

    def export(self):
        """Export config to dictionary."""
        config_dict = super().export()
        config_dict['use_shared_expert'] = self.use_shared_expert
        config_dict['peft_type'] = "MOE"
        return config_dict