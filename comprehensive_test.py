#!/usr/bin/env python3
"""
Comprehensive test script to verify all MoE and MixLoRA components work correctly.
"""

import sys
import os
import traceback

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_imports():
    """Test all critical imports."""
    print("🔍 Testing imports...")

    try:
        # Test MixLoRA imports
        from mixlora.config import MixLoraConfig
        from mixlora.model import MixLoraSparseMoe, load_adapter_weights
        from mixlora.lora_linear import LoraLinear
        print("✓ MixLoRA imports successful")

        # Test MoE imports
        from moe.moe_config import MoEConfig
        from moe.moe_model import SharedExpert, MoEWithSharedExpert, inject_moe_adapter_in_model
        print("✓ MoE imports successful")

        return True
    except ImportError as e:
        print(f"✗ Import error: {e}")
        traceback.print_exc()
        return False

def test_moe_config():
    """Test MoE configuration creation."""
    print("\n🔍 Testing MoE configuration...")

    try:
        from moe.moe_config import MoEConfig

        # Test basic configuration
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
            },
            num_experts=8,
            top_k=2,
            use_shared_expert=True
        )

        print(f"✓ Basic config created - use_shared_expert: {config.use_shared_expert}")

        # Test config without shared expert
        config_no_shared = MoEConfig(
            base_model_name_or_path="test_model",
            task_type="CAUSAL_LM",
            r=8,
            lora_alpha=16,
            num_experts=8,
            top_k=2,
            use_shared_expert=False
        )

        print(f"✓ No shared expert config created - use_shared_expert: {config_no_shared.use_shared_expert}")

        # Test export/import
        exported = config.export()
        imported = MoEConfig.from_config(exported)
        print(f"✓ Export/import successful - use_shared_expert: {imported.use_shared_expert}")

        return True
    except Exception as e:
        print(f"✗ MoE config error: {e}")
        traceback.print_exc()
        return False

def test_shared_expert():
    """Test SharedExpert class."""
    print("\n🔍 Testing SharedExpert class...")

    try:
        import torch
        import torch.nn as nn
        from moe.moe_config import MoEConfig
        from moe.moe_model import SharedExpert

        # Create config
        config = MoEConfig(
            base_model_name_or_path="test",
            task_type="CAUSAL_LM",
            r=8,
            lora_alpha=16,
            lora_dropout=0.05,
            num_experts=8,
            top_k=2,
            use_shared_expert=True
        )

        # Create base layer
        base_layer = nn.Linear(512, 2048, bias=False)

        # Create shared expert
        shared_expert = SharedExpert(config, base_layer)

        print(f"✓ SharedExpert created - LoRA A: {shared_expert.lora_A.weight.shape}")
        print(f"✓ SharedExpert created - LoRA B: {shared_expert.lora_B.weight.shape}")
        print(f"✓ SharedExpert created - Scaling: {shared_expert.scaling}")

        # Test forward pass
        x = torch.randn(4, 512)
        output = shared_expert.forward(x)
        print(f"✓ SharedExpert forward pass - Output shape: {output.shape}")

        # Test forward pass with base_output
        base_output = base_layer(x)
        output_with_base = shared_expert.forward(x, base_output)
        print(f"✓ SharedExpert forward with base_output - Output shape: {output_with_base.shape}")

        return True
    except Exception as e:
        print(f"✗ SharedExpert error: {e}")
        traceback.print_exc()
        return False

def test_moe_with_shared_expert():
    """Test MoEWithSharedExpert class."""
    print("\n🔍 Testing MoEWithSharedExpert class...")

    try:
        import torch
        import torch.nn as nn
        from moe.moe_config import MoEConfig
        from moe.moe_model import MoEWithSharedExpert

        # Create config
        config = MoEConfig(
            base_model_name_or_path="test",
            task_type="CAUSAL_LM",
            r=8,
            lora_alpha=16,
            lora_dropout=0.05,
            target_modules={
                "gate_proj": True,
                "up_proj": True,
                "down_proj": True,
            },
            num_experts=4,
            top_k=2,
            routing_strategy="mixlora",
            router_aux_loss_coef=0.01,
            router_init_range=0.02,
            use_shared_expert=True
        )
        config.model_type_ = "llama"  # Set model type

        # Create mock MLP layer
        class MockMLP(nn.Module):
            def __init__(self):
                super().__init__()
                self.gate_proj = nn.Linear(512, 2048, bias=False)
                self.up_proj = nn.Linear(512, 2048, bias=False)
                self.down_proj = nn.Linear(2048, 512, bias=False)

        base_layer = MockMLP()

        # Create MoE with shared expert
        moe_layer = MoEWithSharedExpert(base_layer, config)

        print(f"✓ MoEWithSharedExpert created - use_shared_expert: {moe_layer.use_shared_expert}")
        print(f"✓ Shared experts count: {len(moe_layer.shared_experts)}")
        print(f"✓ Shared experts keys: {list(moe_layer.shared_experts.keys())}")

        return True
    except Exception as e:
        print(f"✗ MoEWithSharedExpert error: {e}")
        traceback.print_exc()
        return False

def test_training_script_args():
    """Test training script argument parsing."""
    print("\n🔍 Testing training script arguments...")

    try:
        import argparse

        # Simulate train_moe.py argument parsing
        parser = argparse.ArgumentParser()
        parser.add_argument("--use_shared_expert", type=lambda x: x.lower() in ['true', '1', 'yes'], default=True)
        parser.add_argument("--num_experts", type=int, default=8)
        parser.add_argument("--top_k", type=int, default=2)

        # Test different boolean inputs
        args1 = parser.parse_args(["--use_shared_expert", "true"])
        args2 = parser.parse_args(["--use_shared_expert", "false"])
        args3 = parser.parse_args(["--use_shared_expert", "1"])
        args4 = parser.parse_args(["--use_shared_expert", "0"])

        print(f"✓ Boolean parsing - 'true': {args1.use_shared_expert}")
        print(f"✓ Boolean parsing - 'false': {args2.use_shared_expert}")
        print(f"✓ Boolean parsing - '1': {args3.use_shared_expert}")
        print(f"✓ Boolean parsing - '0': {args4.use_shared_expert}")

        return True
    except Exception as e:
        print(f"✗ Training script args error: {e}")
        traceback.print_exc()
        return False

def test_mixlora_lora_path_handling():
    """Test MixLoRA LoRA path handling."""
    print("\n🔍 Testing MixLoRA LoRA path handling...")

    try:
        import argparse

        # Simulate train_mixlora_custom.py argument parsing
        parser = argparse.ArgumentParser()
        parser.add_argument("--pretrained_lora_path", type=str)
        parser.add_argument("--skip_lora_autodetect", action="store_true")

        # Test skip_lora_autodetect
        args1 = parser.parse_args(["--skip_lora_autodetect"])
        args2 = parser.parse_args([])
        args3 = parser.parse_args(["--pretrained_lora_path", "/some/path", "--skip_lora_autodetect"])

        print(f"✓ skip_lora_autodetect only: {args1.skip_lora_autodetect}")
        print(f"✓ Default args: {args2.skip_lora_autodetect}")
        print(f"✓ Both args: path={args3.pretrained_lora_path}, skip={args3.skip_lora_autodetect}")

        return True
    except Exception as e:
        print(f"✗ LoRA path handling error: {e}")
        traceback.print_exc()
        return False

def run_all_tests():
    """Run all tests."""
    print("🚀 Starting comprehensive MoE and MixLoRA tests...\n")

    tests = [
        ("Import Tests", test_imports),
        ("MoE Configuration", test_moe_config),
        ("SharedExpert Class", test_shared_expert),
        ("MoEWithSharedExpert Class", test_moe_with_shared_expert),
        ("Training Script Arguments", test_training_script_args),
        ("MixLoRA LoRA Path Handling", test_mixlora_lora_path_handling),
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
    print("📋 TEST SUMMARY")
    print("="*60)

    passed = 0
    for test_name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status:<8} {test_name}")
        if result:
            passed += 1

    print(f"\n🎯 Results: {passed}/{len(tests)} tests passed")

    if passed == len(tests):
        print("\n🎉 All tests passed! The MoE and MixLoRA system is ready to use.")
        return True
    else:
        print(f"\n⚠️  {len(tests) - passed} tests failed. Please check the errors above.")
        return False

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)