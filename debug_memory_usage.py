#!/usr/bin/env python3
"""
Debug script to analyze memory usage during MixLoRA training.
"""

import sys
import os

def analyze_current_configuration():
    """Analyze current memory configuration."""
    print("🔍 Analyzing current memory configuration...")

    try:
        # Read current configuration
        with open('/Users/yzl/ownCode/MixLoRA/custom_training/train_mixlora_custom.py', 'r') as f:
            content = f.read()

        # Check memory optimizations
        optimizations = {
            "CPU offloading (single GPU)": 'offload_folder="./cpu_offload"' in content and "Single GPU loading" in content,
            "Memory limit (single GPU)": 'max_memory={target_device: "46GiB"}' in content,
            "8-bit optimizer": '"optim": "adamw_8bit"' in content,
            "Gradient checkpointing": '"gradient_checkpointing": True' in content,
            "Parameter freezing fix": 'if self.args.freeze_base_model or self.args.train_mixlora_only:' in content,
        }

        print("✓ Memory optimizations status:")
        all_applied = True
        for opt, applied in optimizations.items():
            status = "✓" if applied else "✗"
            print(f"  {status} {opt}")
            if not applied:
                all_applied = False

        return all_applied

    except Exception as e:
        print(f"✗ Error analyzing configuration: {e}")
        return False

def analyze_batch_size_configuration():
    """Analyze batch size configuration."""
    print("\n🔍 Analyzing batch size configuration...")

    try:
        # Read shell script
        with open('/Users/yzl/ownCode/MixLoRA/run_custom_training.sh', 'r') as f:
            content = f.read()

        import re

        # Extract batch sizes
        single_gpu_batch = re.search(r'Single-GPU \(MixLoRA-only\): batch_size=(\d+)', content)
        dual_gpu_batch = re.search(r'Dual-GPU \(MixLoRA-only\): batch_size=(\d+)', content)

        if single_gpu_batch and dual_gpu_batch:
            single_batch = int(single_gpu_batch.group(1))
            dual_batch = int(dual_gpu_batch.group(1))

            print(f"✓ Current batch sizes:")
            print(f"  - Single GPU: {single_batch}")
            print(f"  - Dual GPU: {dual_batch}")

            if single_batch <= 2 and dual_batch <= 2:
                print("✓ Batch sizes are conservative")
                return True
            else:
                print("⚠️  Batch sizes may be too large")
                return False
        else:
            print("✗ Could not extract batch sizes")
            return False

    except Exception as e:
        print(f"✗ Error analyzing batch sizes: {e}")
        return False

def estimate_memory_breakdown():
    """Estimate detailed memory breakdown."""
    print("\n🔍 Estimating memory breakdown...")

    try:
        # Model parameters (LLaMA-3.1-8B)
        base_model_params = 8_000_000_000
        base_model_memory_fp16 = base_model_params * 2 / 1e9  # FP16 = 2 bytes per param

        # LoRA parameters (reduced configuration)
        num_layers = 32
        hidden_size = 4096
        intermediate_size = 14336
        lora_rank = 2  # Reduced from 8
        num_experts = 2  # Reduced from 8

        # Calculate LoRA memory per layer
        # Attention LoRA: q_proj, v_proj (frozen in train_mixlora_only mode)
        attention_lora_per_layer = 2 * (hidden_size * lora_rank + lora_rank * hidden_size) * 2 / 1e9  # FP16

        # MLP LoRA (only experts are trainable)
        mlp_lora_per_expert = (
            (hidden_size * lora_rank + lora_rank * intermediate_size) * 2 +  # gate_proj + up_proj
            (intermediate_size * lora_rank + lora_rank * hidden_size)        # down_proj
        ) * 2 / 1e9  # FP16

        total_attention_lora = num_layers * attention_lora_per_layer
        total_expert_lora = num_layers * num_experts * mlp_lora_per_expert
        router_memory = num_layers * num_experts * hidden_size * 2 / 1e9  # FP16

        # Activations (batch_size=2, seq_len=512)
        batch_size = 2
        seq_len = 512
        activation_memory = batch_size * seq_len * hidden_size * num_layers * 4 / 1e9  # Conservative estimate

        # Optimizer states (8-bit AdamW with CPU offloading)
        # Only for trainable parameters (experts LoRA + router)
        trainable_memory = total_expert_lora + router_memory
        optimizer_memory = trainable_memory * 3 * 0.5 * 0.3  # 8-bit (50%) + CPU offload (30% on GPU)

        # Buffers and misc
        buffer_memory = 2.0

        print(f"📊 Memory breakdown estimate:")
        print(f"  - Base model (FP16): {base_model_memory_fp16:.1f}GB")
        print(f"  - Attention LoRA (frozen): {total_attention_lora:.1f}GB")
        print(f"  - Expert LoRA (trainable): {total_expert_lora:.1f}GB")
        print(f"  - Router weights: {router_memory:.1f}GB")
        print(f"  - Activations (batch=2): {activation_memory:.1f}GB")
        print(f"  - Optimizer states (8-bit+offload): {optimizer_memory:.1f}GB")
        print(f"  - Buffers and misc: {buffer_memory:.1f}GB")

        total_memory = (base_model_memory_fp16 + total_attention_lora +
                       total_expert_lora + router_memory + activation_memory +
                       optimizer_memory + buffer_memory)

        print(f"\n🎯 Total estimated memory: {total_memory:.1f}GB")
        print(f"🎯 Available GPU memory: ~47GB")

        if total_memory <= 47:
            print("✓ Should fit in GPU memory")
            return True
        else:
            print("⚠️  May still exceed GPU memory")
            return False

    except Exception as e:
        print(f"✗ Error estimating memory: {e}")
        return False

def suggest_additional_optimizations():
    """Suggest additional optimizations if needed."""
    print("\n🔧 Additional optimization suggestions...")

    suggestions = [
        ("Reduce batch size to 1", "BATCH_SIZE=1, GRADIENT_ACCUMULATION_STEPS=32"),
        ("Reduce sequence length", "MAX_LENGTH=256 instead of 512"),
        ("Use FP16 instead of BF16", "More memory efficient on some GPUs"),
        ("Enable ZeRO optimizer", "Distribute optimizer states across devices"),
        ("Use gradient accumulation", "Reduce memory per forward pass"),
        ("Clear cache before training", "torch.cuda.empty_cache()"),
    ]

    print("💡 Try these if OOM persists:")
    for i, (suggestion, description) in enumerate(suggestions, 1):
        print(f"  {i}. {suggestion}: {description}")

def main():
    """Run memory analysis."""
    print("🚀 MixLoRA Memory Usage Analysis...\n")

    tests = [
        ("Current Configuration", analyze_current_configuration),
        ("Batch Size Configuration", analyze_batch_size_configuration),
        ("Memory Breakdown Estimate", estimate_memory_breakdown),
    ]

    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"✗ {test_name} failed: {e}")
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
        suggest_additional_optimizations()
    else:
        print("\n🎉 Memory configuration should work!")

    return passed == len(tests)

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)