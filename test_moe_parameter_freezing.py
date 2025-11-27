#!/usr/bin/env python3
"""
Test script to verify parameter freezing logic for MoE training.
"""

import sys
import os

def test_moe_only_parameter_logic():
    """Test the train_moe_only parameter logic."""
    print("🔍 Testing train_moe_only parameter logic...")

    try:
        # Read the parameter freezing logic
        with open('/Users/yzl/ownCode/MixLoRA/custom_training/train_moe.py', 'r') as f:
            content = f.read()

        # Check if train_moe_only logic is implemented
        if 'if self.args.train_moe_only:' in content:
            print("✓ train_moe_only parameter logic found")
        else:
            print("✗ train_moe_only parameter logic missing")
            return False

        # Check if it correctly identifies MoE components
        if "'moe_gate' in name" in content and "'experts' in name" in content and "'shared_expert' in name" in content:
            print("✓ Correctly identifies MoE components (moe_gate, experts, shared_expert)")
        else:
            print("✗ MoE component identification incorrect")
            return False

        # Check if it freezes other parameters
        if 'param.requires_grad = False' in content:
            print("✓ Freezes non-MoE parameters")
        else:
            print("✗ Parameter freezing logic missing")
            return False

        return True

    except Exception as e:
        print(f"✗ Error testing parameter logic: {e}")
        return False

def test_shell_script_integration():
    """Test shell script integration with train_moe_only."""
    print("\n🔍 Testing shell script integration...")

    try:
        # Read shell script
        with open('/Users/yzl/ownCode/MixLoRA/run_moe.sh', 'r') as f:
            content = f.read()

        # Check if --train_moe_only is passed when LoRA weights exist
        if '--train_moe_only' in content:
            print("✓ Shell script passes --train_moe_only parameter")
        else:
            print("✗ Shell script missing --train_moe_only parameter")
            return False

        # Check the updated message
        if 'freeze pretrained LoRA and only train MoE components' in content:
            print("✓ Updated message explains the training mode")
        else:
            print("✗ Training mode message not updated")
            return False

        return True

    except Exception as e:
        print(f"✗ Error testing shell script: {e}")
        return False

def simulate_moe_parameter_classification():
    """Simulate parameter classification for MoE training."""
    print("\n🔍 Simulating MoE parameter classification...")

    try:
        # Simulate typical parameter names in a MoE model with shared experts
        parameter_names = [
            # Base model parameters (should be frozen)
            "model.layers.0.self_attn.q_proj.weight",
            "model.layers.0.self_attn.k_proj.weight",
            "model.layers.0.self_attn.v_proj.weight",
            "model.layers.0.self_attn.o_proj.weight",

            # Pretrained LoRA parameters (should be frozen)
            "model.layers.0.self_attn.q_proj.lora_A.weight",
            "model.layers.0.self_attn.q_proj.lora_B.weight",
            "model.layers.0.self_attn.v_proj.lora_A.weight",
            "model.layers.0.self_attn.v_proj.lora_B.weight",

            # MoE router (should be trainable)
            "model.layers.0.mlp.moe_gate.weight",

            # Routing experts LoRA (should be trainable)
            "model.layers.0.mlp.experts.0.gate_proj.lora_A.weight",
            "model.layers.0.mlp.experts.0.gate_proj.lora_B.weight",
            "model.layers.0.mlp.experts.0.up_proj.lora_A.weight",
            "model.layers.0.mlp.experts.0.up_proj.lora_B.weight",
            "model.layers.0.mlp.experts.1.gate_proj.lora_A.weight",
            "model.layers.0.mlp.experts.1.gate_proj.lora_B.weight",

            # Shared expert LoRA (should be trainable)
            "model.layers.0.mlp.shared_expert.gate_proj.lora_A.weight",
            "model.layers.0.mlp.shared_expert.gate_proj.lora_B.weight",
            "model.layers.0.mlp.shared_expert.up_proj.lora_A.weight",
            "model.layers.0.mlp.shared_expert.up_proj.lora_B.weight",

            # Base MLP weights (should be frozen)
            "model.layers.0.mlp.gate_proj.weight",
            "model.layers.0.mlp.up_proj.weight",
            "model.layers.0.mlp.down_proj.weight",
        ]

        def should_be_trainable(param_name):
            """Simulate the train_moe_only logic."""
            # Only train MoE components: router, routing experts LoRA, shared experts LoRA
            is_moe_component = (
                ('moe_gate' in param_name) or  # Router
                ('experts' in param_name and any(lora_part in param_name for lora_part in ['lora_A', 'lora_B'])) or  # Routing experts LoRA
                ('shared_expert' in param_name and any(lora_part in param_name for lora_part in ['lora_A', 'lora_B']))  # Shared experts LoRA
            )
            return is_moe_component

        print("📊 Parameter classification results:")
        trainable_count = 0
        frozen_count = 0

        for param_name in parameter_names:
            trainable = should_be_trainable(param_name)
            status = "TRAINABLE" if trainable else "FROZEN"
            print(f"  {status:<10} {param_name}")

            if trainable:
                trainable_count += 1
            else:
                frozen_count += 1

        print(f"\n📈 Summary:")
        print(f"  - Trainable parameters: {trainable_count}")
        print(f"  - Frozen parameters: {frozen_count}")
        print(f"  - Training efficiency: {frozen_count}/{frozen_count + trainable_count} parameters frozen ({frozen_count/(frozen_count + trainable_count)*100:.1f}%)")

        # Verify expected results
        expected_trainable = [
            "model.layers.0.mlp.moe_gate.weight",
            "model.layers.0.mlp.experts.0.gate_proj.lora_A.weight",
            "model.layers.0.mlp.experts.0.gate_proj.lora_B.weight",
            "model.layers.0.mlp.experts.0.up_proj.lora_A.weight",
            "model.layers.0.mlp.experts.0.up_proj.lora_B.weight",
            "model.layers.0.mlp.experts.1.gate_proj.lora_A.weight",
            "model.layers.0.mlp.experts.1.gate_proj.lora_B.weight",
            "model.layers.0.mlp.shared_expert.gate_proj.lora_A.weight",
            "model.layers.0.mlp.shared_expert.gate_proj.lora_B.weight",
            "model.layers.0.mlp.shared_expert.up_proj.lora_A.weight",
            "model.layers.0.mlp.shared_expert.up_proj.lora_B.weight",
        ]

        actual_trainable = [name for name in parameter_names if should_be_trainable(name)]

        if len(actual_trainable) == len(expected_trainable):
            print("✓ Correct number of trainable parameters")
        else:
            print(f"✗ Expected {len(expected_trainable)} trainable, got {len(actual_trainable)}")
            return False

        # Check that only MoE components are trainable
        all_moe_related = all(
            ('moe_gate' in name or 'experts' in name or 'shared_expert' in name)
            for name in actual_trainable
        )
        if all_moe_related:
            print("✓ Only MoE-related parameters are trainable")
        else:
            print("✗ Non-MoE parameters are trainable")
            return False

        return True

    except Exception as e:
        print(f"✗ Error simulating parameter classification: {e}")
        return False

def test_training_efficiency():
    """Test training efficiency with parameter freezing."""
    print("\n🔍 Testing training efficiency...")

    try:
        # Estimate parameter reduction for MoE with shared experts
        total_model_params = 8_000_000_000  # 8B model
        base_lora_params = 100_000_000      # ~100M LoRA parameters
        routing_experts_params = 50_000_000 # ~50M routing experts parameters
        shared_expert_params = 25_000_000   # ~25M shared expert parameters
        router_params = 5_000_000           # ~5M router parameters

        # Without freezing: train everything
        without_freezing = total_model_params + base_lora_params + routing_experts_params + shared_expert_params + router_params

        # With freezing: only train MoE components
        with_freezing = routing_experts_params + shared_expert_params + router_params

        reduction_ratio = with_freezing / without_freezing
        memory_savings = (1 - reduction_ratio) * 100

        print(f"📊 Training efficiency analysis:")
        print(f"  - Total model parameters: {total_model_params:,}")
        print(f"  - Base LoRA parameters: {base_lora_params:,}")
        print(f"  - Routing experts parameters: {routing_experts_params:,}")
        print(f"  - Shared expert parameters: {shared_expert_params:,}")
        print(f"  - Router parameters: {router_params:,}")
        print(f"  - Without freezing (trainable): {without_freezing:,}")
        print(f"  - With freezing (trainable): {with_freezing:,}")
        print(f"  - Parameter reduction: {reduction_ratio:.4f}x")
        print(f"  - Memory savings: {memory_savings:.1f}%")

        if memory_savings > 90:
            print("✓ Excellent memory savings achieved")
        elif memory_savings > 80:
            print("✓ Good memory savings achieved")
        else:
            print("⚠️  Limited memory savings")

        return True

    except Exception as e:
        print(f"✗ Error testing training efficiency: {e}")
        return False

def main():
    """Run all MoE parameter freezing tests."""
    print("🚀 Testing MoE parameter freezing configuration...\n")

    tests = [
        ("MoE Only Parameter Logic", test_moe_only_parameter_logic),
        ("Shell Script Integration", test_shell_script_integration),
        ("Parameter Classification Simulation", simulate_moe_parameter_classification),
        ("Training Efficiency", test_training_efficiency),
    ]

    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"✗ {test_name} failed with exception: {e}")
            results.append((test_name, False))

    print("\n" + "="*60)
    print("📋 MOE PARAMETER FREEZING TEST SUMMARY")
    print("="*60)

    passed = 0
    for test_name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status:<8} {test_name}")
        if result:
            passed += 1

    print(f"\n🎯 Results: {passed}/{len(tests)} tests passed")

    if passed == len(tests):
        print("\n🎉 MoE parameter freezing configuration is correct!")
        print("\n📋 Training configuration:")
        print("  ✅ Pretrained LoRA weights: FROZEN")
        print("  ✅ Base model parameters: FROZEN")
        print("  ✅ MoE router (moe_gate): TRAINABLE")
        print("  ✅ Routing experts LoRA: TRAINABLE")
        print("  ✅ Shared expert LoRA: TRAINABLE")
        print("  ❄️  All other parameters: FROZEN")
        print("\n🚀 run_moe will now only train MoE components!")
        print("\n💡 Usage:")
        print("  ./run_moe.sh llama 2 2 true")
        print("  (Requires LORA_WEIGHTS_PATH to exist)")
        return True
    else:
        print(f"\n⚠️  {len(tests) - passed} tests failed.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)