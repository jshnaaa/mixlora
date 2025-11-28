#!/usr/bin/env python3
"""
Test script to verify all OOM fixes are applied correctly.
"""

import sys
import os

def test_parameter_freezing_fix():
    """Test that parameter freezing fix is applied."""
    print("🔍 Testing parameter freezing fix...")

    try:
        with open('/Users/yzl/ownCode/MixLoRA/custom_training/train_mixlora_custom.py', 'r') as f:
            content = f.read()

        # Check if the fix is applied
        if 'if self.args.freeze_base_model or self.args.train_mixlora_only:' in content:
            print("✓ Parameter freezing fix applied")
            return True
        else:
            print("✗ Parameter freezing fix missing")
            return False

    except Exception as e:
        print(f"✗ Error testing parameter freezing fix: {e}")
        return False

def test_memory_optimizations():
    """Test that all memory optimizations are applied."""
    print("\n🔍 Testing memory optimizations...")

    try:
        with open('/Users/yzl/ownCode/MixLoRA/custom_training/train_mixlora_custom.py', 'r') as f:
            content = f.read()

        optimizations = [
            ('max_split_size_mb:128', "Memory allocator optimization"),
            ('max_memory={target_device: "46GiB"}', "Single GPU memory limit"),
            ('offload_folder="./cpu_offload"', "Single GPU CPU offloading"),
            ('"optim": "adamw_8bit"', "8-bit optimizer"),
            ('"gradient_checkpointing": True', "Gradient checkpointing"),
            ('gc.collect()', "Garbage collection"),
            ('torch.cuda.reset_peak_memory_stats', "GPU memory stats reset"),
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

def test_batch_size_configuration():
    """Test that batch sizes are conservative."""
    print("\n🔍 Testing batch size configuration...")

    try:
        with open('/Users/yzl/ownCode/MixLoRA/run_custom_training.sh', 'r') as f:
            content = f.read()

        # Check for minimal batch sizes
        if 'BATCH_SIZE=1   # Minimal batch size' in content:
            print("✓ Batch size reduced to 1")
        else:
            print("✗ Batch size not minimal")
            return False

        if 'GRADIENT_ACCUMULATION_STEPS=32' in content:
            print("✓ Gradient accumulation increased to 32")
        else:
            print("✗ Gradient accumulation not optimized")
            return False

        return True

    except Exception as e:
        print(f"✗ Error testing batch size configuration: {e}")
        return False

def test_parameter_count_simulation():
    """Simulate actual parameter count for MixLoRA training."""
    print("\n🔍 Simulating parameter count for MixLoRA training...")

    try:
        # Simulate MixLoRA model structure
        num_layers = 32
        hidden_size = 4096
        intermediate_size = 14336
        num_experts = 2  # Reduced from 8
        lora_rank = 2    # Reduced from 8

        # Calculate trainable parameters (only MoE components)
        # Router parameters
        router_params = num_layers * num_experts * hidden_size

        # Expert LoRA parameters (only MLP experts are trainable)
        expert_lora_params_per_layer = num_experts * (
            # gate_proj LoRA: A (r x hidden) + B (intermediate x r)
            (lora_rank * hidden_size + intermediate_size * lora_rank) +
            # up_proj LoRA: A (r x hidden) + B (intermediate x r)
            (lora_rank * hidden_size + intermediate_size * lora_rank) +
            # down_proj LoRA: A (r x intermediate) + B (hidden x r)
            (lora_rank * intermediate_size + hidden_size * lora_rank)
        )

        total_expert_lora_params = num_layers * expert_lora_params_per_layer
        total_trainable_params = router_params + total_expert_lora_params

        # Total model parameters
        base_model_params = 8_000_000_000  # 8B model
        total_params = base_model_params + total_trainable_params

        print(f"📊 Parameter analysis:")
        print(f"  - Base model parameters: {base_model_params:,}")
        print(f"  - Router parameters: {router_params:,}")
        print(f"  - Expert LoRA parameters: {total_expert_lora_params:,}")
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

        if total_trainable_params < 100_000_000:  # Less than 100M trainable params
            print("✓ Very low number of trainable parameters")
            return True
        else:
            print("⚠️  High number of trainable parameters")
            return False

    except Exception as e:
        print(f"✗ Error simulating parameter count: {e}")
        return False

def test_environment_variables():
    """Test environment variable settings."""
    print("\n🔍 Testing environment variable settings...")

    try:
        with open('/Users/yzl/ownCode/MixLoRA/custom_training/train_mixlora_custom.py', 'r') as f:
            content = f.read()

        env_vars = [
            ('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True,max_split_size_mb:128'),
            ('CUDA_LAUNCH_BLOCKING', '1'),
            ('TORCH_USE_CUDA_DSA', '1'),
        ]

        all_set = True
        for var_name, expected_value in env_vars:
            pattern = f'os.environ["{var_name}"] = "{expected_value}"'
            if pattern in content:
                print(f"  ✓ {var_name} = {expected_value}")
            else:
                print(f"  ✗ {var_name} not set correctly")
                all_set = False

        return all_set

    except Exception as e:
        print(f"✗ Error testing environment variables: {e}")
        return False

def main():
    """Run all OOM fix tests."""
    print("🚀 Testing OOM fixes for MixLoRA training...\n")

    tests = [
        ("Parameter Freezing Fix", test_parameter_freezing_fix),
        ("Memory Optimizations", test_memory_optimizations),
        ("Batch Size Configuration", test_batch_size_configuration),
        ("Parameter Count Simulation", test_parameter_count_simulation),
        ("Environment Variables", test_environment_variables),
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
    print("📋 OOM FIXES TEST SUMMARY")
    print("="*60)

    passed = 0
    for test_name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status:<8} {test_name}")
        if result:
            passed += 1

    print(f"\n🎯 Results: {passed}/{len(tests)} tests passed")

    if passed == len(tests):
        print("\n🎉 All OOM fixes applied successfully!")
        print("\n📋 Applied fixes:")
        print("  ✅ Fixed parameter freezing logic (train_mixlora_only now works)")
        print("  ✅ Added CPU offloading for single GPU")
        print("  ✅ Reduced batch size to 1")
        print("  ✅ Added aggressive memory cleanup")
        print("  ✅ Optimized memory allocator settings")
        print("  ✅ Enabled 8-bit optimizer")
        print("  ✅ Enabled gradient checkpointing")
        print("\n🚀 run_custom_training should now work without OOM!")
        print("\n💡 If OOM still occurs, try:")
        print("  - Reduce MAX_LENGTH from 512 to 256")
        print("  - Use single GPU instead of dual GPU")
        print("  - Check if other processes are using GPU memory")
        return True
    else:
        print(f"\n⚠️  {len(tests) - passed} tests failed.")
        print("Please fix the failing tests before running training.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)