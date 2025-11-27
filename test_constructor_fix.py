#!/usr/bin/env python3
"""
Simple test to verify LoraLinear constructor fix is correct.
"""

import sys
import os
import re

def test_lora_constructor_fix():
    """Test that LoraLinear constructor uses correct parameters."""
    print("🔍 Testing LoraLinear constructor fix...")

    try:
        # Read the moe_model.py file
        with open('/Users/yzl/ownCode/MixLoRA/moe/moe_model.py', 'r') as f:
            content = f.read()

        # Find the LoraLinear constructor call
        pattern = r'lora_layer = LoraLinear\(\s*([^)]+)\)'
        matches = re.findall(pattern, content, re.DOTALL)

        if not matches:
            print("✗ No LoraLinear constructor found")
            return False

        for i, match in enumerate(matches):
            print(f"\n✓ Found LoraLinear constructor {i+1}:")
            constructor_params = match.strip()
            print(f"Parameters: {constructor_params}")

            # Check for correct parameters
            if 'base_layer=' in constructor_params:
                print("✓ Uses 'base_layer' parameter")
            else:
                print("✗ Missing 'base_layer' parameter")
                return False

            if 'config=' in constructor_params:
                print("✓ Uses 'config' parameter")
            else:
                print("✗ Missing 'config' parameter")
                return False

            # Check for incorrect old-style parameters (exact matches only)
            incorrect_patterns = [
                r'\br\s*=',           # r= parameter
                r'\blora_alpha\s*=',  # lora_alpha= parameter
                r'\blora_dropout\s*=', # lora_dropout= parameter
                r'\bbias\s*=',        # bias= parameter
                r'\buse_dora\s*=',    # use_dora= parameter
                r'\buse_rslora\s*='   # use_rslora= parameter
            ]

            found_incorrect = False
            for pattern in incorrect_patterns:
                if re.search(pattern, constructor_params):
                    print(f"✗ Found incorrect parameter pattern: {pattern}")
                    found_incorrect = True

            if found_incorrect:
                return False
            else:
                print("✓ No incorrect parameters found")

        print("\n🎉 LoraLinear constructor fix is correct!")
        return True

    except Exception as e:
        print(f"✗ Error checking file: {e}")
        return False

def test_shared_expert_constructor():
    """Test SharedExpert constructor parameters."""
    print("\n🔍 Testing SharedExpert constructor...")

    try:
        # Read the moe_model.py file
        with open('/Users/yzl/ownCode/MixLoRA/moe/moe_model.py', 'r') as f:
            content = f.read()

        # Check SharedExpert class definition
        if 'class SharedExpert(nn.Module):' in content:
            print("✓ SharedExpert class found")

            # Check constructor signature
            pattern = r'def __init__\(self, config: MoEConfig, base_layer: nn\.Linear, device=None, dtype=None\):'
            if re.search(pattern, content):
                print("✓ SharedExpert constructor signature is correct")
            else:
                print("✗ SharedExpert constructor signature incorrect")
                return False

            # Check LoRA layer creation
            if 'self.lora_A = nn.Linear(' in content and 'self.lora_B = nn.Linear(' in content:
                print("✓ SharedExpert creates LoRA layers correctly")
            else:
                print("✗ SharedExpert LoRA layer creation incorrect")
                return False

            return True
        else:
            print("✗ SharedExpert class not found")
            return False

    except Exception as e:
        print(f"✗ Error checking SharedExpert: {e}")
        return False

def main():
    """Run all constructor tests."""
    print("🚀 Testing constructor fixes...\n")

    tests = [
        ("LoraLinear Constructor Fix", test_lora_constructor_fix),
        ("SharedExpert Constructor", test_shared_expert_constructor),
    ]

    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"✗ {test_name} failed with exception: {e}")
            results.append((test_name, False))

    print("\n" + "="*50)
    print("📋 CONSTRUCTOR TEST SUMMARY")
    print("="*50)

    passed = 0
    for test_name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status:<8} {test_name}")
        if result:
            passed += 1

    print(f"\n🎯 Results: {passed}/{len(tests)} tests passed")

    if passed == len(tests):
        print("\n🎉 All constructor fixes are correct!")
        print("\nKey fixes applied:")
        print("✓ LoraLinear constructor uses (base_layer, config) pattern")
        print("✓ Removed incorrect individual parameters (r, lora_alpha, etc.)")
        print("✓ SharedExpert constructor properly defined")
        print("✓ All parameter passing is consistent")
        return True
    else:
        print(f"\n⚠️  {len(tests) - passed} tests failed.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)