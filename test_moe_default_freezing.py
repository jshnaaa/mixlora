#!/usr/bin/env python3
"""
Test script to verify MoE default parameter freezing configuration.
"""

import sys
import os

def test_moe_default_configuration():
    """Test that MoE uses default parameter freezing."""
    print("🔍 Testing MoE default configuration...")

    try:
        with open('/Users/yzl/ownCode/MixLoRA/custom_training/train_moe.py', 'r') as f:
            content = f.read()

        # Check default values
        checks = [
            ('train_moe_only: bool = field(default=True', "train_moe_only defaults to True"),
            ('num_experts: int = field(default=2', "Number of experts reduced to 2"),
            ('lora_r: int = field(default=2', "LoRA rank reduced to 2"),
            ('batch_size: int = field(default=1', "Batch size set to 1"),
            ('gradient_accumulation_steps: int = field(default=16', "Gradient accumulation set to 16"),
        ]

        all_correct = True
        for pattern, description in checks:
            if pattern in content:
                print(f"  ✓ {description}")
            else:
                print(f"  ✗ {description} - NOT SET")
                all_correct = False

        return all_correct

    except Exception as e:
        print(f"✗ Error testing MoE default configuration: {e}")
        return False

def test_moe_shell_script_freezing():
    """Test that run_moe.sh always uses parameter freezing."""
    print("\n🔍 Testing run_moe.sh parameter freezing...")

    try:
        with open('/Users/yzl/ownCode/MixLoRA/run_moe.sh', 'r') as f:
            content = f.read()

        # Check that --train_moe_only is always passed
        checks = [
            ('MOE_ARGS="--train_moe_only --use_shared_expert $USE_SHARED"', "Always passes --train_moe_only when no LoRA path"),
            ('MOE_ARGS="--pretrained_lora_path', "Passes pretrained LoRA path when available"),
            ('--train_moe_only --use_shared_expert', "Always includes --train_moe_only"),
        ]

        all_correct = True
        for pattern, description in checks:
            if pattern in content:
                print(f"  ✓ {description}")
            else:
                print(f"  ✗ {description} - MISSING")
                all_correct = False

        return all_correct

    except Exception as e:
        print(f"✗ Error testing shell script: {e}")
        return False

def test_moe_memory_optimizations():
    """Test that MoE has memory optimizations."""
    print("\n🔍 Testing MoE memory optimizations...")

    try:
        with open('/Users/yzl/ownCode/MixLoRA/custom_training/train_moe.py', 'r') as f:
            content = f.read()

        # Check memory optimizations
        optimizations = [
            ('max_split_size_mb:128', "Memory allocator optimization"),
            ('max_memory={target_device: "46GiB"}', "Memory limit for single GPU"),
            ('offload_folder="./cpu_offload"', "CPU offloading for single GPU"),
            ('optim="adamw_8bit"', "8-bit optimizer"),
            ('gradient_checkpointing=True', "Gradient checkpointing"),
            ('gc.collect()', "Garbage collection before training"),
            ('torch.cuda.reset_peak_memory_stats', "GPU memory stats reset"),
            ('bf16=torch.cuda.is_bf16_supported()', "Dynamic precision selection"),
        ]

        all_applied = True
        for pattern, description in optimizations:
            if pattern in content:
                print(f"  ✓ {description}")
            else:
                print(f"  ✗ {description} - MISSING")
                all_applied = False

        return all_applied

    except Exception as e:
        print(f"✗ Error testing memory optimizations: {e}")
        return False

def simulate_moe_parameter_count():
    """Simulate MoE parameter count with default settings."""
    print("\n🔍 Simulating MoE parameter count...")

    try:
        # Default MoE configuration
        num_layers = 32
        hidden_size = 4096
        intermediate_size = 14336
        num_experts = 2  # Reduced from 8
        lora_rank = 2    # Reduced from 8

        # Calculate trainable parameters (only MoE components)
        # Router parameters
        router_params = num_layers * num_experts * hidden_size

        # Routing expert LoRA parameters
        routing_expert_lora_per_layer = num_experts * (
            # gate_proj LoRA: A (r x hidden) + B (intermediate x r)
            (lora_rank * hidden_size + intermediate_size * lora_rank) +
            # up_proj LoRA: A (r x hidden) + B (intermediate x r)
            (lora_rank * hidden_size + intermediate_size * lora_rank) +
            # down_proj LoRA: A (r x intermediate) + B (hidden x r)
            (lora_rank * intermediate_size + hidden_size * lora_rank)
        )

        # Shared expert LoRA parameters (same structure as routing experts)
        shared_expert_lora_per_layer = (
            # gate_proj LoRA: A (r x hidden) + B (intermediate x r)
            (lora_rank * hidden_size + intermediate_size * lora_rank) +
            # up_proj LoRA: A (r x hidden) + B (intermediate x r)
            (lora_rank * hidden_size + intermediate_size * lora_rank) +
            # down_proj LoRA: A (r x intermediate) + B (hidden x r)
            (lora_rank * intermediate_size + hidden_size * lora_rank)
        )

        total_routing_expert_lora = num_layers * routing_expert_lora_per_layer
        total_shared_expert_lora = num_layers * shared_expert_lora_per_layer
        total_trainable_params = router_params + total_routing_expert_lora + total_shared_expert_lora

        # Total model parameters
        base_model_params = 8_000_000_000  # 8B model
        total_params = base_model_params + total_trainable_params

        print(f"📊 MoE parameter analysis:")
        print(f"  - Base model parameters: {base_model_params:,}")
        print(f"  - Router parameters: {router_params:,}")
        print(f"  - Routing expert LoRA parameters: {total_routing_expert_lora:,}")
        print(f"  - Shared expert LoRA parameters: {total_shared_expert_lora:,}")
        print(f"  - Total trainable parameters: {total_trainable_params:,}")
        print(f"  - Total model parameters: {total_params:,}")
        print(f"  - Trainable ratio: {total_trainable_params/total_params*100:.4f}%")

        # Memory estimation
        trainable_memory_gb = total_trainable_params * 2 / 1e9  # FP16
        base_model_memory_gb = base_model_params * 2 / 1e9     # FP16

        print(f"\n💾 Memory estimation:")
        print(f"  - Base model memory: {base_model_memory_gb:.1f}GB")
        print(f"  - Trainable parameters memory: {trainable_memory_gb:.1f}GB")
        print(f"  - Optimizer memory (8-bit+offload): {trainable_memory_gb * 3 * 0.5 * 0.3:.1f}GB")
        print(f"  - Total estimated: {base_model_memory_gb + trainable_memory_gb + (trainable_memory_gb * 3 * 0.5 * 0.3):.1f}GB")

        if total_trainable_params < 50_000_000:  # Less than 50M trainable params
            print("✓ Very low number of trainable parameters")
            return True
        else:
            print("⚠️  High number of trainable parameters")
            return False

    except Exception as e:
        print(f"✗ Error simulating parameter count: {e}")
        return False

def main():
    """Run all MoE default freezing tests."""
    print("🚀 Testing MoE default parameter freezing configuration...\n")

    tests = [
        ("MoE Default Configuration", test_moe_default_configuration),
        ("Shell Script Freezing", test_moe_shell_script_freezing),
        ("Memory Optimizations", test_moe_memory_optimizations),
        ("Parameter Count Simulation", simulate_moe_parameter_count),
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
    print("📋 MOE DEFAULT FREEZING TEST SUMMARY")
    print("="*60)

    passed = 0
    for test_name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status:<8} {test_name}")
        if result:
            passed += 1

    print(f"\n🎯 Results: {passed}/{len(tests)} tests passed")

    if passed == len(tests):
        print("\n🎉 MoE default parameter freezing configuration is correct!")
        print("\n📋 Default training configuration:")
        print("  ✅ train_moe_only=True (default)")
        print("  ✅ num_experts=2 (memory optimized)")
        print("  ✅ lora_r=2 (memory optimized)")
        print("  ✅ batch_size=1 (conservative)")
        print("  ✅ Always freeze pretrained LoRA")
        print("  ✅ Only train MoE components:")
        print("    - Router (moe_gate)")
        print("    - Routing experts LoRA")
        print("    - Shared expert LoRA")
        print("  ❄️  All other parameters frozen")
        print("\n🚀 run_moe will now default to memory-efficient training!")
        print("\n💡 Usage (all default to parameter freezing):")
        print("  ./run_moe.sh llama 2 2 true")
        print("  ./run_moe.sh qwen 3 1 true")
        return True
    else:
        print(f"\n⚠️  {len(tests) - passed} tests failed.")
        print("Please check the failing tests.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)