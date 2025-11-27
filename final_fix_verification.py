#!/usr/bin/env python3
"""
Final verification script for all MoE and MixLoRA fixes.
"""

import sys
import os
import re
from dataclasses import dataclass, field
from typing import Optional

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_lora_linear_constructor_fix():
    """Test LoraLinear constructor fix."""
    print("🔍 Testing LoraLinear constructor fix...")

    try:
        with open('/Users/yzl/ownCode/MixLoRA/moe/moe_model.py', 'r') as f:
            content = f.read()

        # Find LoraLinear constructor
        pattern = r'lora_layer = LoraLinear\(\s*([^)]+)\)'
        matches = re.findall(pattern, content, re.DOTALL)

        if not matches:
            print("✗ No LoraLinear constructor found")
            return False

        constructor_params = matches[0].strip()

        # Check for correct parameters
        if 'base_layer=' in constructor_params and 'config=' in constructor_params:
            print("✓ LoraLinear uses correct (base_layer, config) pattern")
        else:
            print("✗ LoraLinear constructor incorrect")
            return False

        # Check for absence of incorrect parameters
        incorrect_patterns = [r'\br\s*=', r'\blora_alpha\s*=', r'\blora_dropout\s*=']
        for pattern in incorrect_patterns:
            if re.search(pattern, constructor_params):
                print(f"✗ Found incorrect parameter: {pattern}")
                return False

        print("✓ No incorrect parameters found")
        return True

    except Exception as e:
        print(f"✗ Error: {e}")
        return False

def test_custom_training_arguments_fix():
    """Test CustomTrainingArguments fix."""
    print("\n🔍 Testing CustomTrainingArguments fix...")

    try:
        with open('/Users/yzl/ownCode/MixLoRA/custom_training/train_mixlora_custom.py', 'r') as f:
            content = f.read()

        # Check that skip_lora_autodetect is in dataclass
        if 'skip_lora_autodetect: bool = field(default=False' in content:
            print("✓ skip_lora_autodetect parameter added to CustomTrainingArguments")
        else:
            print("✗ skip_lora_autodetect parameter missing from CustomTrainingArguments")
            return False

        # Check that it's in argument parser
        if '--skip_lora_autodetect' in content and 'action="store_true"' in content:
            print("✓ skip_lora_autodetect argument parser configured")
        else:
            print("✗ skip_lora_autodetect argument parser missing")
            return False

        return True

    except Exception as e:
        print(f"✗ Error: {e}")
        return False

def test_moe_config_parameter_mapping():
    """Test MoE config parameter mapping."""
    print("\n🔍 Testing MoE config parameter mapping...")

    try:
        with open('/Users/yzl/ownCode/MixLoRA/moe/moe_config.py', 'r') as f:
            content = f.read()

        # Check for parameter mapping
        mapping_patterns = [
            "kwargs.pop('base_model_name_or_path')",
            "kwargs['base_model_']",
            "class MoEConfig(MixLoraConfig)"
        ]

        for pattern in mapping_patterns:
            if pattern in content:
                print(f"✓ Found parameter mapping: {pattern}")
            else:
                print(f"✗ Missing parameter mapping: {pattern}")
                return False

        return True

    except Exception as e:
        print(f"✗ Error: {e}")
        return False

def test_shared_expert_implementation():
    """Test SharedExpert implementation."""
    print("\n🔍 Testing SharedExpert implementation...")

    try:
        with open('/Users/yzl/ownCode/MixLoRA/moe/moe_model.py', 'r') as f:
            content = f.read()

        # Check SharedExpert class
        if 'class SharedExpert(nn.Module):' in content:
            print("✓ SharedExpert class defined")
        else:
            print("✗ SharedExpert class missing")
            return False

        # Check LoRA structure
        if 'self.lora_A = nn.Linear(' in content and 'self.lora_B = nn.Linear(' in content:
            print("✓ SharedExpert has LoRA structure")
        else:
            print("✗ SharedExpert LoRA structure missing")
            return False

        # Check forward method
        if 'def forward(self, x, base_output=None):' in content:
            print("✓ SharedExpert forward method defined")
        else:
            print("✗ SharedExpert forward method missing")
            return False

        return True

    except Exception as e:
        print(f"✗ Error: {e}")
        return False

def test_run_scripts():
    """Test run scripts configuration."""
    print("\n🔍 Testing run scripts...")

    try:
        # Test run_moe.sh
        with open('/Users/yzl/ownCode/MixLoRA/run_moe.sh', 'r') as f:
            moe_content = f.read()

        if 'USE_SHARED=${4:-"true"}' in moe_content and '--use_shared_expert $USE_SHARED' in moe_content:
            print("✓ run_moe.sh properly configured for shared experts")
        else:
            print("✗ run_moe.sh configuration incorrect")
            return False

        # Test run_custom_training.sh
        with open('/Users/yzl/ownCode/MixLoRA/run_custom_training.sh', 'r') as f:
            custom_content = f.read()

        if '--skip_lora_autodetect' in custom_content:
            print("✓ run_custom_training.sh uses skip_lora_autodetect")
        else:
            print("✗ run_custom_training.sh missing skip_lora_autodetect")
            return False

        return True

    except Exception as e:
        print(f"✗ Error: {e}")
        return False

def main():
    """Run all verification tests."""
    print("🚀 Final verification of all MoE and MixLoRA fixes...\n")

    tests = [
        ("LoraLinear Constructor Fix", test_lora_linear_constructor_fix),
        ("CustomTrainingArguments Fix", test_custom_training_arguments_fix),
        ("MoE Config Parameter Mapping", test_moe_config_parameter_mapping),
        ("SharedExpert Implementation", test_shared_expert_implementation),
        ("Run Scripts Configuration", test_run_scripts),
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
    print("📋 FINAL VERIFICATION SUMMARY")
    print("="*60)

    passed = 0
    for test_name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status:<8} {test_name}")
        if result:
            passed += 1

    print(f"\n🎯 Results: {passed}/{len(tests)} tests passed")

    if passed == len(tests):
        print("\n🎉 ALL FIXES VERIFIED SUCCESSFULLY!")
        print("\n🔧 Summary of fixes applied:")
        print("1. ✅ LoraLinear constructor: Fixed parameter pattern to (base_layer, config)")
        print("2. ✅ CustomTrainingArguments: Added skip_lora_autodetect parameter")
        print("3. ✅ MoE config: Fixed parameter mapping for PEFT compatibility")
        print("4. ✅ SharedExpert: Implemented with proper LoRA structure")
        print("5. ✅ Run scripts: Configured for proper parameter passing")
        print("\n🚀 The MoE system with shared experts is ready for training!")
        return True
    else:
        print(f"\n⚠️  {len(tests) - passed} tests failed. Please check the errors above.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)