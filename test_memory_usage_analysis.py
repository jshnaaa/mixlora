#!/usr/bin/env python3
"""
Test script to analyze memory usage for MixLoRA training.
"""

import sys
import os

def analyze_mixlora_memory_requirements():
    """Analyze MixLoRA memory requirements."""
    print("🔍 Analyzing MixLoRA memory requirements...")

    try:
        # Read current configuration
        with open('/Users/yzl/ownCode/MixLoRA/custom_training/train_mixlora_custom.py', 'r') as f:
            content = f.read()

        # Extract key parameters
        import re

        # Extract parameters
        num_experts = re.search(r'num_experts: int = field\(default=(\d+)', content)
        lora_r = re.search(r'lora_r: int = field\(default=(\d+)', content)
        batch_size = re.search(r'batch_size: int = field\(default=(\d+)', content)
        gradient_accumulation = re.search(r'gradient_accumulation_steps: int = field\(default=(\d+)', content)
        max_memory = re.search(r'max_memory=\{target_device: "(\d+)GiB"\}', content)

        if all([num_experts, lora_r, batch_size, gradient_accumulation, max_memory]):
            experts = int(num_experts.group(1))
            rank = int(lora_r.group(1))
            bs = int(batch_size.group(1))
            grad_acc = int(gradient_accumulation.group(1))
            mem_limit = int(max_memory.group(1))

            print(f"✓ Current configuration:")
            print(f"  - Number of experts: {experts}")
            print(f"  - LoRA rank: {rank}")
            print(f"  - Batch size: {bs}")
            print(f"  - Gradient accumulation: {grad_acc}")
            print(f"  - Memory limit: {mem_limit}GB")
            print(f"  - Effective batch size: {bs * grad_acc}")

            # Estimate memory usage
            base_model_memory = 16  # LLaMA-3.1-8B in FP16

            # LoRA parameters per expert (rough estimate)
            # For LLaMA: gate_proj, up_proj, down_proj
            # Each has LoRA A (hidden_size x rank) + LoRA B (rank x intermediate_size)
            hidden_size = 4096
            intermediate_size = 14336  # LLaMA-3.1-8B

            lora_params_per_expert = (
                2 * (hidden_size * rank + rank * intermediate_size) +  # gate_proj + up_proj
                (intermediate_size * rank + rank * hidden_size)        # down_proj
            ) * 2 / 1e9  # Convert to GB (FP16)

            total_lora_memory = experts * lora_params_per_expert

            # Activation memory (rough estimate)
            seq_len = 512
            activation_memory = bs * seq_len * hidden_size * 4 / 1e9  # FP16, multiple layers

            # Check if using optimizations
            use_8bit_optimizer = '"optim": "adamw_8bit"' in content
            use_cpu_offload = 'offload_folder="./cpu_offload"' in content

            # Optimizer memory (AdamW: gradients + momentum + variance)
            optimizer_multiplier = 3
            if use_8bit_optimizer:
                optimizer_multiplier *= 0.5  # 8-bit reduces optimizer memory by ~50%
            if use_cpu_offload:
                optimizer_multiplier *= 0.3  # CPU offloading reduces GPU optimizer memory by ~70%

            optimizer_memory = (base_model_memory + total_lora_memory) * optimizer_multiplier

            print(f"\n🔧 Applied optimizations:")
            if use_8bit_optimizer:
                print(f"  ✓ 8-bit optimizer enabled (50% memory reduction)")
            if use_cpu_offload:
                print(f"  ✓ CPU offloading enabled (70% GPU memory reduction)")

            total_estimated = base_model_memory + total_lora_memory + activation_memory + optimizer_memory

            print(f"\n📊 Memory estimation:")
            print(f"  - Base model: {base_model_memory:.1f}GB")
            print(f"  - LoRA parameters: {total_lora_memory:.1f}GB")
            print(f"  - Activations: {activation_memory:.1f}GB")
            print(f"  - Optimizer states: {optimizer_memory:.1f}GB")
            print(f"  - Total estimated: {total_estimated:.1f}GB")
            print(f"  - Memory limit: {mem_limit}GB")

            if total_estimated > mem_limit:
                print(f"⚠️  Estimated usage ({total_estimated:.1f}GB) exceeds limit ({mem_limit}GB)")
                return False
            else:
                print(f"✓ Estimated usage within limit")
                return True
        else:
            print("✗ Could not extract configuration parameters")
            return False

    except Exception as e:
        print(f"✗ Error analyzing memory requirements: {e}")
        return False

def suggest_memory_optimizations():
    """Suggest further memory optimizations."""
    print("\n🔧 Suggested memory optimizations...")

    optimizations = [
        ("Reduce experts", "4 → 2", "~50% LoRA memory reduction"),
        ("Reduce LoRA rank", "4 → 2", "~75% LoRA memory reduction"),
        ("Reduce sequence length", "512 → 256", "~50% activation memory reduction"),
        ("Use CPU offloading", "offload_folder='./offload'", "Move optimizer to CPU"),
        ("Use 8-bit optimizer", "optim='adamw_8bit'", "~50% optimizer memory reduction"),
        ("Increase memory limit", "46GB → 47GB", "Use more available GPU memory"),
    ]

    for opt, change, benefit in optimizations:
        print(f"  • {opt}: {change} → {benefit}")

    print("\n💡 Recommended immediate fixes:")
    print("  1. Reduce experts to 2")
    print("  2. Reduce LoRA rank to 2")
    print("  3. Increase memory limit to 47GB")
    print("  4. Consider using CPU offloading")

def test_optimized_configuration():
    """Test an optimized configuration."""
    print("\n🧪 Testing optimized configuration...")

    try:
        # Simulate optimized parameters
        optimized_config = {
            "num_experts": 2,
            "lora_r": 2,
            "batch_size": 1,
            "gradient_accumulation_steps": 32,  # Maintain effective batch size
            "max_memory_gb": 47,
            "sequence_length": 512,
        }

        # Estimate memory with optimized config
        base_model_memory = 16
        hidden_size = 4096
        intermediate_size = 14336

        lora_params_per_expert = (
            2 * (hidden_size * optimized_config["lora_r"] + optimized_config["lora_r"] * intermediate_size) +
            (intermediate_size * optimized_config["lora_r"] + optimized_config["lora_r"] * hidden_size)
        ) * 2 / 1e9

        total_lora_memory = optimized_config["num_experts"] * lora_params_per_expert
        activation_memory = optimized_config["batch_size"] * optimized_config["sequence_length"] * hidden_size * 4 / 1e9
        optimizer_memory = (base_model_memory + total_lora_memory) * 3

        total_optimized = base_model_memory + total_lora_memory + activation_memory + optimizer_memory

        print(f"📊 Optimized configuration memory:")
        print(f"  - Base model: {base_model_memory:.1f}GB")
        print(f"  - LoRA parameters: {total_lora_memory:.1f}GB")
        print(f"  - Activations: {activation_memory:.1f}GB")
        print(f"  - Optimizer states: {optimizer_memory:.1f}GB")
        print(f"  - Total estimated: {total_optimized:.1f}GB")
        print(f"  - Memory limit: {optimized_config['max_memory_gb']}GB")

        if total_optimized <= optimized_config['max_memory_gb']:
            print("✓ Optimized configuration should fit in memory")
            return True
        else:
            print("✗ Even optimized configuration may not fit")
            return False

    except Exception as e:
        print(f"✗ Error testing optimized configuration: {e}")
        return False

def main():
    """Run all memory analysis."""
    print("🚀 Memory usage analysis for MixLoRA training...\n")

    tests = [
        ("Current Memory Requirements", analyze_mixlora_memory_requirements),
        ("Memory Optimization Suggestions", suggest_memory_optimizations),
        ("Optimized Configuration Test", test_optimized_configuration),
    ]

    results = []
    for test_name, test_func in tests:
        try:
            if test_func == suggest_memory_optimizations:
                test_func()  # This function doesn't return a result
                results.append((test_name, True))
            else:
                result = test_func()
                results.append((test_name, result))
        except Exception as e:
            print(f"✗ {test_name} failed with exception: {e}")
            results.append((test_name, False))

    print("\n" + "="*60)
    print("📋 MEMORY ANALYSIS SUMMARY")
    print("="*60)

    passed = 0
    for test_name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status:<8} {test_name}")
        if result:
            passed += 1

    print(f"\n🎯 Results: {passed}/{len(tests)} tests passed")

    if passed < len(tests):
        print("\n⚠️  Current configuration may cause OOM!")
        print("🔧 Apply suggested optimizations to fix memory issues.")
    else:
        print("\n🎉 Memory configuration looks good!")

    return passed == len(tests)

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)