#!/usr/bin/env python3
"""
Quick test script to verify MoE configuration works correctly.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from moe.moe_config import MoEConfig

def test_moe_config():
    """Test MoE configuration creation."""
    print("Testing MoE configuration...")

    try:
        # Test 1: Basic configuration
        config = MoEConfig(
            base_model_name_or_path="test_model",
            task_type="CAUSAL_LM",
            r=8,
            lora_alpha=16,
            lora_dropout=0.05,
            target_modules={
                "gate_proj": True,
                "up_proj": True,
                "down_proj": True,
                "q_proj": False,
                "k_proj": False,
                "v_proj": False,
                "o_proj": False,
            },
            use_dora=False,
            use_rslora=False,
            routing_strategy="mixlora",
            num_experts=8,
            top_k=2,
            router_aux_loss_coef=0.01,
            router_init_range=0.02,
            jitter_noise=0.0,
            use_shared_expert=True
        )

        print("✓ Basic configuration created successfully")
        print(f"  - Base model: {config.base_model_}")
        print(f"  - Task type: {config.task_type_}")
        print(f"  - LoRA rank: {config.lora_r_}")
        print(f"  - Num experts: {config.num_experts_}")
        print(f"  - Use shared expert: {config.use_shared_expert}")
        print(f"  - PEFT type: {config.peft_type_}")

        # Test 2: Configuration with shared expert disabled
        config_no_shared = MoEConfig(
            base_model_name_or_path="test_model",
            task_type="CAUSAL_LM",
            r=8,
            lora_alpha=16,
            lora_dropout=0.05,
            target_modules={
                "gate_proj": True,
                "up_proj": True,
                "down_proj": True,
            },
            routing_strategy="mixlora",
            num_experts=8,
            top_k=2,
            use_shared_expert=False
        )

        print("✓ Configuration without shared expert created successfully")
        print(f"  - Use shared expert: {config_no_shared.use_shared_expert}")

        # Test 3: Export and import
        exported = config.export()
        print("✓ Configuration exported successfully")
        print(f"  - Exported keys: {list(exported.keys())}")

        imported = MoEConfig.from_config(exported)
        print("✓ Configuration imported successfully")
        print(f"  - Imported use_shared_expert: {imported.use_shared_expert}")

        return True

    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_moe_config()
    if success:
        print("\n🎉 All tests passed! MoE configuration is working correctly.")
        sys.exit(0)
    else:
        print("\n❌ Tests failed! Please check the configuration.")
        sys.exit(1)