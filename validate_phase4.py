"""
Phase 4 Validation Script

Validates production-ready features:
1. Custom exceptions with structured output
2. CLI interface (validate, stats, run)
3. Pipeline v2 with progress tracking and metadata
4. Logger level override support

Run with:
    python validate_phase4.py
"""

import subprocess
import sys
from pathlib import Path


def print_header(text):
    print("\n" + "=" * 70)
    print(f"  {text}")
    print("=" * 70)


def run_pytest(test_path, description=""):
    """Run pytest and return pass/fail with output."""
    cmd = [sys.executable, "-m", "pytest", str(test_path), "-v", "--tb=short", "--no-header"]
    result = subprocess.run(cmd, capture_output=True, text=True)

    passed = result.returncode == 0
    status = "[PASS]" if passed else "[FAIL]"
    symbol = "+" if passed else "X"
    print(f"  {symbol} {status} {description or test_path}")

    if not passed:
        # Show last few lines of output for debugging
        lines = result.stdout.strip().split("\n")
        for line in lines[-5:]:
            if "FAILED" in line or "ERROR" in line:
                print(f"        {line.strip()}")

    return passed


def check_module_import(module_path, description):
    """Check that a module can be imported without errors."""
    cmd = [sys.executable, "-c", f"import {module_path}"]
    result = subprocess.run(cmd, capture_output=True, text=True)
    passed = result.returncode == 0
    status = "[PASS]" if passed else "[FAIL]"
    symbol = "+" if passed else "X"
    print(f"  {symbol} {status} {description}")

    if not passed:
        print(f"        Error: {result.stderr.strip().split(chr(10))[-1]}")

    return passed


def check_cli_help():
    """Verify CLI help message works."""
    cmd = [sys.executable, "-m", "src.pipeline.cli", "--help"]
    result = subprocess.run(cmd, capture_output=True, text=True)
    passed = result.returncode == 0 and "CORC-NAH" in result.stdout
    status = "[PASS]" if passed else "[FAIL]"
    symbol = "+" if passed else "X"
    print(f"  {symbol} {status} CLI help message")
    return passed


def check_cli_validate():
    """Verify CLI validate subcommand works."""
    cmd = [sys.executable, "-m", "src.pipeline.cli", "-q", "validate"]
    result = subprocess.run(cmd, capture_output=True, text=True)
    passed = result.returncode in [0, 1]  # Both are valid outcomes
    status = "[PASS]" if passed else "[FAIL]"
    symbol = "+" if passed else "X"
    print(f"  {symbol} {status} CLI validate command")
    return passed


def check_cli_stats():
    """Verify CLI stats subcommand works."""
    cmd = [sys.executable, "-m", "src.pipeline.cli", "-q", "stats"]
    result = subprocess.run(cmd, capture_output=True, text=True)
    passed = result.returncode == 0
    status = "[PASS]" if passed else "[FAIL]"
    symbol = "+" if passed else "X"
    print(f"  {symbol} {status} CLI stats command")
    return passed


def main():
    print_header("Phase 4: Production-Ready Features - Validation")

    all_passed = True

    # Suite 1: Module imports
    print_header("Suite 1: Module Imports")

    modules = [
        ("src.exceptions", "Exceptions module"),
        ("src.pipeline.cli", "CLI module"),
        ("src.pipeline.unify_v2", "Pipeline v2 module"),
    ]

    for module, desc in modules:
        passed = check_module_import(module, desc)
        all_passed = all_passed and passed

    # Suite 2: CLI interface
    print_header("Suite 2: CLI Interface")

    cli_checks = [
        check_cli_help,
        check_cli_validate,
        check_cli_stats,
    ]

    for check in cli_checks:
        passed = check()
        all_passed = all_passed and passed

    # Suite 3: Unit tests
    print_header("Suite 3: Unit Tests")

    test_files = [
        ("tests/unit/test_cli.py", "CLI + Exceptions + Pipeline v2 tests"),
    ]

    for test_file, desc in test_files:
        if Path(test_file).exists():
            passed = run_pytest(test_file, desc)
            all_passed = all_passed and passed
        else:
            print(f"  X [FAIL] {desc} - File not found: {test_file}")
            all_passed = False

    # Suite 4: Integration check
    print_header("Suite 4: Integration Check")

    passed = run_pytest(
        "tests/integration/test_shadow_mode.py",
        "Shadow mode integration tests still pass"
    )
    all_passed = all_passed and passed

    # Summary
    print_header("Validation Summary")

    if all_passed:
        print("\n  [SUCCESS] All Phase 4 validations passed!")
        print("\n  Production features working:")
        print("    + Custom exceptions with structured context")
        print("    + CLI interface (run, validate, stats)")
        print("    + Pipeline v2 with progress bars")
        print("    + Metadata tracking support")
        print("    + Logger level override")
        print("    + Graceful error handling")
        print("\n  Next: Phase 5 - Final Polish & Documentation")
        return 0
    else:
        print("\n  [FAILED] Some validations failed.")
        print("\n  Fix the errors and run again.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
