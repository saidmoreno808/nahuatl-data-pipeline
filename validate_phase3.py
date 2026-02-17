"""
Phase 3 Validation Script

Validates the unified pipeline by running shadow mode tests.

This ensures that:
1. New pipeline runs successfully
2. Deduplication works correctly
3. Layer priority is respected (Diamond > Silver)
4. Unicode preservation works (macrons)
5. Split ratios are correct
6. Results are reproducible

Run with:
    python validate_phase3.py
"""

import subprocess
import sys
from pathlib import Path


def print_header(text):
    """Print section header."""
    print("\n" + "=" * 70)
    print(f"  {text}")
    print("=" * 70)


def print_test_result(test_name, passed):
    """Print test result."""
    status = "[PASS]" if passed else "[FAIL]"
    symbol = "+" if passed else "X"
    print(f"  {symbol} {status} {test_name}")


def run_pytest(test_path, marker=None):
    """
    Run pytest and return success status.

    Args:
        test_path: Path to test file or directory
        marker: Optional pytest marker to filter tests

    Returns:
        bool: True if all tests passed
    """
    cmd = [sys.executable, "-m", "pytest", str(test_path), "-v", "--tb=short"]

    if marker:
        cmd.extend(["-m", marker])

    result = subprocess.run(cmd, capture_output=True, text=True)

    return result.returncode == 0


def main():
    """Run Phase 3 validation."""
    print_header("Phase 3: Shadow Mode Testing - Validation")

    all_passed = True

    # Test Suite 1: Integration Tests
    print_header("Test Suite 1: Integration Tests")
    print("Running shadow mode integration tests...")

    test_file = Path("tests/integration/test_shadow_mode.py")

    if not test_file.exists():
        print(f"  [FAIL] Test file not found: {test_file}")
        all_passed = False
    else:
        passed = run_pytest(test_file)
        print_test_result("Shadow mode integration tests", passed)
        all_passed = all_passed and passed

    # Test Suite 2: Pipeline Module
    print_header("Test Suite 2: Pipeline Module")
    print("Checking pipeline module structure...")

    checks = [
        ("src/pipeline/__init__.py", "Pipeline __init__.py exists"),
        ("src/pipeline/unify.py", "UnifiedPipeline module exists"),
    ]

    for file_path, description in checks:
        exists = Path(file_path).exists()
        print_test_result(description, exists)
        all_passed = all_passed and exists

    # Test Suite 3: Core Functionality
    print_header("Test Suite 3: Core Functionality Tests")
    print("Testing deduplication, layer priority, and splits...")

    # Run specific test classes
    test_classes = [
        "TestShadowMode",
        "TestLegacyFormatSupport",
    ]

    for test_class in test_classes:
        test_selector = f"{test_file}::{test_class}"
        passed = run_pytest(test_selector)
        print_test_result(f"{test_class} tests", passed)
        all_passed = all_passed and passed

    # Final Summary
    print_header("Validation Summary")

    if all_passed:
        print("\n  [SUCCESS] All Phase 3 validations passed!")
        print("\n  Next steps:")
        print("  1. Run: make test-integration")
        print("  2. Test with real data (if available)")
        print("  3. Compare outputs with legacy pipeline (make parity)")
        print("  4. Continue to Phase 4: Production-Ready Features")
        return 0
    else:
        print("\n  [FAILED] Some validations failed.")
        print("\n  Please fix the errors and run validation again.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
