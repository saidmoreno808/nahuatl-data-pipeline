#!/usr/bin/env python3
"""
Environment Validation Script

Verifies that the development environment is correctly configured
for CORC-NAH project development.

Usage:
    python tests/validate_environment.py
"""

import os
import platform
import subprocess
import sys
from pathlib import Path


class Colors:
    """ANSI color codes for terminal output."""

    GREEN = "\033[92m"
    RED = "\033[91m"
    YELLOW = "\033[93m"
    BLUE = "\033[94m"
    RESET = "\033[0m"


def print_check(message: str, passed: bool):
    """Print a check result with color."""
    symbol = f"{Colors.GREEN}✓{Colors.RESET}" if passed else f"{Colors.RED}✗{Colors.RESET}"
    print(f"{symbol} {message}")


def print_warning(message: str):
    """Print a warning message."""
    print(f"{Colors.YELLOW}⚠️  {message}{Colors.RESET}")


def print_header(message: str):
    """Print a section header."""
    print(f"\n{Colors.BLUE}{'=' * 60}{Colors.RESET}")
    print(f"{Colors.BLUE}{message}{Colors.RESET}")
    print(f"{Colors.BLUE}{'=' * 60}{Colors.RESET}\n")


def check_python_version():
    """Verify Python version >= 3.10."""
    version = sys.version_info
    passed = version.major == 3 and version.minor >= 10

    print_check(
        f"Python {version.major}.{version.minor}.{version.micro}",
        passed
    )

    if not passed:
        print_warning("Python 3.10+ required. Please upgrade.")

    return passed


def check_virtual_env():
    """Verify virtual environment is active."""
    in_venv = hasattr(sys, 'real_prefix') or (
        hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix
    )

    print_check("Virtual environment active", in_venv)

    if not in_venv:
        print_warning("Virtual environment not detected. Activate with: source .venv/bin/activate")

    return in_venv


def check_git_config():
    """Verify Git is configured correctly."""
    try:
        # Check autocrlf
        result = subprocess.run(
            ["git", "config", "--get", "core.autocrlf"],
            capture_output=True,
            text=True,
            check=False
        )
        autocrlf = result.stdout.strip()
        autocrlf_ok = autocrlf == "input"

        # Check eol
        result = subprocess.run(
            ["git", "config", "--get", "core.eol"],
            capture_output=True,
            text=True,
            check=False
        )
        eol = result.stdout.strip()
        eol_ok = eol == "lf"

        passed = autocrlf_ok and eol_ok

        print_check(
            f"Git line endings (autocrlf={autocrlf}, eol={eol})",
            passed
        )

        if not passed:
            print_warning("Configure with: git config --global core.autocrlf input")
            print_warning("Configure with: git config --global core.eol lf")

        return passed

    except FileNotFoundError:
        print_check("Git installed", False)
        return False


def check_encoding():
    """Verify UTF-8 encoding."""
    encoding = sys.stdout.encoding
    passed = encoding.lower() in ['utf-8', 'utf8']

    print_check(f"UTF-8 encoding ({encoding})", passed)

    if not passed:
        print_warning("Set LANG=en_US.UTF-8 in your shell")

    return passed


def check_wsl2():
    """Verify running in WSL2 (not Windows native)."""
    is_wsl = 'microsoft' in platform.uname().release.lower()

    if platform.system() == 'Windows':
        print_check("WSL2 environment", False)
        print_warning("Must run in WSL2, not Windows native!")
        print_warning("Open Ubuntu terminal and navigate to ~/projects/corc-nah")
        return False
    elif is_wsl:
        print_check("WSL2 environment", True)
        return True
    else:
        # Linux native is also OK
        print_check("Linux environment", True)
        return True


def check_packages():
    """Verify required packages are installed."""
    required_packages = [
        'pandas',
        'requests',
        'yt_dlp',
        'youtube_transcript_api',
    ]

    dev_packages = [
        'pytest',
        'black',
        'isort',
        'mypy',
    ]

    all_passed = True

    for package in required_packages:
        try:
            __import__(package)
            print_check(f"Package: {package}", True)
        except ImportError:
            print_check(f"Package: {package}", False)
            all_passed = False

    print()

    for package in dev_packages:
        try:
            __import__(package)
            print_check(f"Dev package: {package}", True)
        except ImportError:
            print_check(f"Dev package: {package}", False)
            print_warning(f"Install with: pip install {package}")

    return all_passed


def check_docker():
    """Verify Docker is accessible."""
    try:
        result = subprocess.run(
            ["docker", "--version"],
            capture_output=True,
            text=True,
            check=False
        )
        passed = result.returncode == 0

        print_check(f"Docker accessible", passed)

        if not passed:
            print_warning("Docker Desktop must be running with WSL2 integration enabled")

        return passed

    except FileNotFoundError:
        print_check("Docker installed", False)
        return False


def check_filesystem():
    """Verify running on WSL2 native filesystem."""
    cwd = Path.cwd()

    # Check if on /mnt/c (Windows filesystem)
    is_windows_fs = str(cwd).startswith('/mnt/c') or str(cwd).startswith('C:')

    passed = not is_windows_fs

    if is_windows_fs:
        print_check("WSL2 native filesystem", False)
        print_warning("Project is on Windows filesystem (/mnt/c)")
        print_warning("For best performance, move to ~/projects/")
        print_warning("Example: mv /mnt/c/path/to/project ~/projects/")
    else:
        print_check("WSL2 native filesystem", True)

    return passed


def check_project_structure():
    """Verify project structure is correct."""
    required_dirs = [
        'benchmark',
        'config',
        'data',
        'docs',
        'scripts',
        'tests',
    ]

    required_files = [
        '.gitattributes',
        '.editorconfig',
        'README.md',
        'requirements.txt',
        'pytest.ini',
        'Makefile',
    ]

    all_passed = True

    for dir_name in required_dirs:
        path = Path(dir_name)
        exists = path.exists() and path.is_dir()
        print_check(f"Directory: {dir_name}/", exists)
        all_passed = all_passed and exists

    print()

    for file_name in required_files:
        path = Path(file_name)
        exists = path.exists() and path.is_file()
        print_check(f"File: {file_name}", exists)
        all_passed = all_passed and exists

    return all_passed


def main():
    """Run all validation checks."""
    print_header("CORC-NAH Environment Validation")

    checks = [
        ("Python Version", check_python_version),
        ("Virtual Environment", check_virtual_env),
        ("Git Configuration", check_git_config),
        ("UTF-8 Encoding", check_encoding),
        ("WSL2/Linux Environment", check_wsl2),
        ("Python Packages", check_packages),
        ("Docker", check_docker),
        ("Filesystem Location", check_filesystem),
        ("Project Structure", check_project_structure),
    ]

    results = []

    for check_name, check_func in checks:
        print_header(check_name)
        try:
            result = check_func()
            results.append((check_name, result))
        except Exception as e:
            print_check(f"Check failed with error: {e}", False)
            results.append((check_name, False))

    # Summary
    print_header("Summary")

    passed_count = sum(1 for _, passed in results if passed)
    total_count = len(results)

    for check_name, passed in results:
        symbol = f"{Colors.GREEN}✓{Colors.RESET}" if passed else f"{Colors.RED}✗{Colors.RESET}"
        print(f"{symbol} {check_name}")

    print(f"\n{passed_count}/{total_count} checks passed")

    if passed_count == total_count:
        print(f"\n{Colors.GREEN}✅ Environment is ready for development!{Colors.RESET}\n")
        return 0
    else:
        print(f"\n{Colors.RED}❌ Please fix the issues above before continuing.{Colors.RESET}\n")
        return 1


if __name__ == "__main__":
    sys.exit(main())
