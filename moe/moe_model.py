"""
MoE model implementation with shared experts.
Based on MixLoRA but adds shared experts that are always activated.
"""

import sys
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mixlora.model import (
    MixLoraSparseMoe,
    inject_adapter_in_model as mixlora_inject_adapter
)
from mixlora.lora_linear import LoraLinear
from moe.moe_config import MoEConfig


class SharedExpert(nn.Module):
    """
    Shared expert that is always activated (not routed).
    Has the same structure as routed experts but with weight=1.
    """

    def __init__(self, config: MoEConfig, in_features: int, out_features: int, device=None, dtype=None):
        super().__init__()
        self.config = config
        self.in_features = in_features
        self.out_features = out_features

        # Create LoRA layers for shared expert (same structure as routed experts)
        self.lora_A = nn.Linear(in_features, config.lora_r_, bias=False, device=device, dtype=dtype)
        self.lora_B = nn.Linear(config.lora_r_, out_features, bias=False, device=device, dtype=dtype)

        # Initialize weights same as routed experts
        nn.init.normal_(self.lora_A.weight, std=0.01)
        nn.init.zeros_(self.lora_B.weight)

        self.scaling = config.lora_alpha_ / config.lora_r_

    def forward(self, x):
        """Forward pass through shared expert."""
        # LoRA forward: x + scaling * B(A(x))
        lora_output = self.lora_B(self.lora_A(x))
        return x + self.scaling * lora_output


class MoEWithSharedExpert(MixLoraSparseMoe):
    """
    MoE layer with shared expert support.
    Extends MixLoraSparseMoe to add shared expert that is always activated.
    """

    def __init__(self, base_layer, config: MoEConfig):
        # Initialize parent MixLoRA Sparse MoE
        super().__init__(base_layer, config)

        self.use_shared_expert = config.use_shared_expert

        # Add shared experts for each projection type if enabled
        self.shared_experts = {}
        if self.use_shared_expert:
            # Get the model type to determine which projections to create shared experts for
            model_type = getattr(config, 'model_type_', 'llama')

            if model_type in ['llama', 'gemma', 'gemma2', 'qwen2', 'mistral']:
                # For LLaMA-style models: gate_proj, up_proj, down_proj
                for proj_name in ['gate_proj', 'up_proj', 'down_proj']:
                    if hasattr(base_layer, proj_name):
                        base_proj = getattr(base_layer, proj_name)
                        self.shared_experts[proj_name] = SharedExpert(
                            config,
                            base_proj.in_features,
                            base_proj.out_features,
                            device=base_proj.weight.device,
                            dtype=base_proj.weight.dtype
                        )
            elif model_type == 'phi':
                # For Phi models: fc1, fc2
                for proj_name in ['fc1', 'fc2']:
                    if hasattr(base_layer, proj_name):
                        base_proj = getattr(base_layer, proj_name)
                        self.shared_experts[proj_name] = SharedExpert(
                            config,
                            base_proj.in_features,
                            base_proj.out_features,
                            device=base_proj.weight.device,
                            dtype=base_proj.weight.dtype
                        )
            elif model_type == 'phi3':
                # For Phi3 models: gate_up_proj, down_proj
                for proj_name in ['gate_up_proj', 'down_proj']:
                    if hasattr(base_layer, proj_name):
                        base_proj = getattr(base_layer, proj_name)
                        self.shared_experts[proj_name] = SharedExpert(
                            config,
                            base_proj.in_features,
                            base_proj.out_features,
                            device=base_proj.weight.device,
                            dtype=base_proj.weight.dtype
                        )

    def _llama_forward(self, expert_mask, hidden_states, input_dtype):
        """LLaMA forward with shared expert support."""
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_dim)

        # Get base layer outputs
        common_gate = self.base_layer_.gate_proj(hidden_states)
        common_up = self.base_layer_.up_proj(hidden_states)

        # Process routed experts (original MixLoRA logic)
        final_expert_states = []
        for expert_idx in range(self.num_experts_):
            lora_gate = self.experts_.get(f"experts.{expert_idx}.gate_proj", None)
            lora_up = self.experts_.get(f"experts.{expert_idx}.up_proj", None)
            lora_down = self.experts_.get(f"experts.{expert_idx}.down_proj", None)

            gate_states = common_gate
            up_states = common_up

            if lora_gate is not None:
                gate_states = lora_gate.lora_forward(gate_states, hidden_states)
            if lora_up is not None:
                up_states = lora_up.lora_forward(up_states, hidden_states)

            act_result = self.act_fn_(gate_states) * up_states

            if lora_down is not None:
                expert_output = lora_down.lora_forward(
                    self.base_layer_.down_proj(act_result), act_result
                )
            else:
                expert_output = self.base_layer_.down_proj(act_result)

            final_expert_states.append(expert_output)

        # Combine routed experts with weights
        final_expert_states = torch.stack(final_expert_states, dim=-1)
        routed_output = torch.sum(final_expert_states * expert_mask, dim=-1)

        # Add shared expert output if enabled
        if self.use_shared_expert and self.shared_experts:
            # Compute shared expert output
            shared_gate = self.shared_experts.get('gate_proj', None)
            shared_up = self.shared_experts.get('up_proj', None)
            shared_down = self.shared_experts.get('down_proj', None)

            if shared_gate is not None and shared_up is not None and shared_down is not None:
                # Shared expert forward: same structure as routed experts
                gate_output = shared_gate.forward(hidden_states)
                up_output = shared_up.forward(hidden_states)
                act_result = self.act_fn_(gate_output) * up_output
                shared_output = shared_down.forward(act_result)

                # Weighted normalization: consider routing weights
                # Sum of routing weights for normalization
                routing_weights_sum = torch.sum(expert_mask, dim=-1, keepdim=True)
                # Shared expert weight is 1
                total_weight = routing_weights_sum + 1.0

                # Combine outputs with proper normalization
                combined_output = (routed_output + shared_output) / total_weight
                final_output = combined_output
            else:
                final_output = routed_output
        else:
            final_output = routed_output

        return final_output.view(batch_size, sequence_length, hidden_dim).to(input_dtype)

    def _phi_forward(self, expert_mask, hidden_states, input_dtype):
        """Phi forward with shared expert support."""
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_dim)

        # Get base layer output
        common_fc1 = self.base_layer_.fc1(hidden_states)

        # Process routed experts
        final_expert_states = []
        for expert_idx in range(self.num_experts_):
            lora_fc1 = self.experts_.get(f"experts.{expert_idx}.fc1", None)
            lora_fc2 = self.experts_.get(f"experts.{expert_idx}.fc2", None)

            fc1_states = common_fc1
            if lora_fc1 is not None:
                fc1_states = lora_fc1.lora_forward(fc1_states, hidden_states)

            act_result = self.act_fn_(fc1_states)

            if lora_fc2 is not None:
                expert_output = lora_fc2.lora_forward(
                    self.base_layer_.fc2(act_result), act_result
                )
            else:
                expert_output = self.base_layer_.fc2(act_result)

            final_expert_states.append(expert_output)

        # Combine routed experts
        final_expert_states = torch.stack(final_expert_states, dim=-1)
        routed_output = torch.sum(final_expert_states * expert_mask, dim=-1)

        # Add shared expert output if enabled
        if self.use_shared_expert and self.shared_experts:
            shared_fc1 = self.shared_experts.get('fc1', None)
            shared_fc2 = self.shared_experts.get('fc2', None)

            if shared_fc1 is not None and shared_fc2 is not None:
                fc1_output = shared_fc1.forward(hidden_states)
                act_result = self.act_fn_(fc1_output)
                shared_output = shared_fc2.forward(act_result)

                # Weighted normalization
                routing_weights_sum = torch.sum(expert_mask, dim=-1, keepdim=True)
                total_weight = routing_weights_sum + 1.0
                combined_output = (routed_output + shared_output) / total_weight
                final_output = combined_output
            else:
                final_output = routed_output
        else:
            final_output = routed_output

        return final_output.view(batch_size, sequence_length, hidden_dim).to(input_dtype)

    def _phi3_forward(self, expert_mask, hidden_states, input_dtype):
        """Phi3 forward with shared expert support."""
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_dim)

        # Get base layer outputs
        common_gate_up = self.base_layer_.gate_up_proj(hidden_states)
        gate_states, up_states = common_gate_up.chunk(2, dim=-1)

        # Process routed experts
        final_expert_states = []
        for expert_idx in range(self.num_experts_):
            lora_gate_up = self.experts_.get(f"experts.{expert_idx}.gate_up_proj", None)
            lora_down = self.experts_.get(f"experts.{expert_idx}.down_proj", None)

            current_gate_states = gate_states
            current_up_states = up_states

            if lora_gate_up is not None:
                gate_up_output = lora_gate_up.lora_forward(common_gate_up, hidden_states)
                current_gate_states, current_up_states = gate_up_output.chunk(2, dim=-1)

            act_result = current_up_states * self.act_fn_(current_gate_states)

            if lora_down is not None:
                expert_output = lora_down.lora_forward(
                    self.base_layer_.down_proj(act_result), act_result
                )
            else:
                expert_output = self.base_layer_.down_proj(act_result)

            final_expert_states.append(expert_output)

        # Combine routed experts
        final_expert_states = torch.stack(final_expert_states, dim=-1)
        routed_output = torch.sum(final_expert_states * expert_mask, dim=-1)

        # Add shared expert output if enabled
        if self.use_shared_expert and self.shared_experts:
            shared_gate_up = self.shared_experts.get('gate_up_proj', None)
            shared_down = self.shared_experts.get('down_proj', None)

            if shared_gate_up is not None and shared_down is not None:
                gate_up_output = shared_gate_up.forward(hidden_states)
                gate_output, up_output = gate_up_output.chunk(2, dim=-1)
                act_result = up_output * self.act_fn_(gate_output)
                shared_output = shared_down.forward(act_result)

                # Weighted normalization
                routing_weights_sum = torch.sum(expert_mask, dim=-1, keepdim=True)
                total_weight = routing_weights_sum + 1.0
                combined_output = (routed_output + shared_output) / total_weight
                final_output = combined_output
            else:
                final_output = routed_output
        else:
            final_output = routed_output

        return final_output.view(batch_size, sequence_length, hidden_dim).to(input_dtype)


def inject_moe_adapter_in_model(model, config: MoEConfig, weights: Dict[str, torch.Tensor]):
    """
    Inject MoE adapters with shared expert support into model.
    Based on MixLoRA injection but uses MoE components with shared experts.
    """
    # Set model type in config
    config.model_type_ = model.config.model_type
    model._mixlora_config = config

    # Inject MoE adapters into each layer
    for idx, layer in enumerate(model.model.layers):
        # Inject attention modules (regular LoRA, no MoE)
        _inject_attn_module(idx, layer.self_attn, config, weights)

        # Inject MLP modules (MoE with shared experts)
        _inject_moe_mlp_module(idx, layer.mlp, config, weights)

    return model


def _inject_attn_module(layer_idx: int, attn_layer, config: MoEConfig, weights: Dict[str, torch.Tensor]):
    """Inject LoRA adapters into attention module (no MoE for attention)."""
    from mixlora.model import _inject_attn_module as mixlora_inject_attn
    # Use the original MixLoRA attention injection
    mixlora_inject_attn(layer_idx, attn_layer, config, weights)


def _inject_moe_mlp_module(layer_idx: int, mlp_layer, config: MoEConfig, weights: Dict[str, torch.Tensor]):
    """Inject MoE adapters with shared expert support into MLP module."""
    # Create MoE layer with shared expert support
    moe_layer = MoEWithSharedExpert(mlp_layer, config)

    # Load router gate weights
    gate_weight_key = f"mixlora.layers.{layer_idx}.mlp.moe_gate.weight"
    if gate_weight_key in weights:
        moe_layer.gate_ = weights[gate_weight_key]

    # Load expert weights for each projection
    target_modules = config.target_modules_
    model_type = getattr(config, 'model_type_', 'llama')

    # Determine projection names based on model type
    if model_type in ['llama', 'gemma', 'gemma2', 'qwen2', 'mistral']:
        proj_names = ['gate_proj', 'up_proj', 'down_proj']
    elif model_type == 'phi':
        proj_names = ['fc1', 'fc2']
    elif model_type == 'phi3':
        proj_names = ['gate_up_proj', 'down_proj']
    else:
        proj_names = ['gate_proj', 'up_proj', 'down_proj']  # Default to LLaMA style

    # Create expert LoRA layers
    for proj_name in proj_names:
        if target_modules.get(proj_name, False) and hasattr(mlp_layer, proj_name):
            base_proj = getattr(mlp_layer, proj_name)

            # Create LoRA layers for each expert
            for expert_idx in range(config.num_experts_):
                expert_key = f"experts.{expert_idx}.{proj_name}"

                # Create LoRA layer
                lora_layer = LoraLinear(
                    base_proj.in_features,
                    base_proj.out_features,
                    r=config.lora_r_,
                    lora_alpha=config.lora_alpha_,
                    lora_dropout=config.lora_dropout_,
                    bias=base_proj.bias is not None,
                    use_dora=config.use_dora_,
                    use_rslora=config.use_rslora_,
                    device=base_proj.weight.device,
                    dtype=base_proj.weight.dtype
                )

                # Load weights if available
                lora_a_key = f"mixlora.layers.{layer_idx}.mlp.experts.{expert_idx}.{proj_name}.lora_A.weight"
                lora_b_key = f"mixlora.layers.{layer_idx}.mlp.experts.{expert_idx}.{proj_name}.lora_B.weight"

                if lora_a_key in weights and lora_b_key in weights:
                    lora_layer.reset_parameters((weights[lora_a_key], weights[lora_b_key]))
                else:
                    # Use default initialization (same as routed experts)
                    lora_layer.reset_parameters()

                # Add to MoE layer
                moe_layer.experts_[expert_key] = lora_layer

    # Load shared expert weights if they exist
    if moe_layer.use_shared_expert and moe_layer.shared_experts:
        for proj_name in proj_names:
            if proj_name in moe_layer.shared_experts:
                shared_expert = moe_layer.shared_experts[proj_name]

                # Load shared expert weights if available
                shared_a_key = f"mixlora.layers.{layer_idx}.mlp.shared_expert.{proj_name}.lora_A.weight"
                shared_b_key = f"mixlora.layers.{layer_idx}.mlp.shared_expert.{proj_name}.lora_B.weight"

                if shared_a_key in weights and shared_b_key in weights:
                    shared_expert.lora_A.weight.data.copy_(weights[shared_a_key])
                    shared_expert.lora_B.weight.data.copy_(weights[shared_b_key])
                # If no weights found, keep default initialization (same as routed experts)

    # Replace the original forward function
    mlp_layer.forward = moe_layer.forward