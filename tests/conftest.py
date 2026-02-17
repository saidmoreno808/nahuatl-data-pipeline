"""
Pytest Configuration and Shared Fixtures

This file contains test fixtures and configuration shared across all tests.
"""

import json
import tempfile
from pathlib import Path
from typing import Dict

import pandas as pd
import pytest


# ============================================================================
# Session-scoped fixtures (run once per test session)
# ============================================================================


@pytest.fixture(scope="session")
def project_root() -> Path:
    """Returns the project root directory."""
    return Path(__file__).parent.parent


@pytest.fixture(scope="session")
def benchmark_dir(project_root) -> Path:
    """Returns the benchmark directory path."""
    return project_root / "benchmark"


@pytest.fixture(scope="session")
def data_dir(project_root) -> Path:
    """Returns the data directory path."""
    return project_root / "data"


# ============================================================================
# Sample data fixtures
# ============================================================================


@pytest.fixture
def sample_nahuatl_record() -> Dict:
    """Returns a sample Náhuatl record with Unicode characters."""
    return {
        "es": "Hola, ¿cómo estás?",
        "nah": "Piyali, ¿quēn timotlaneltoquia?",  # Note: macron on 'ē'
        "source": "test_source",
        "layer": "diamond",
        "origin_file": "test.jsonl",
    }


@pytest.fixture
def sample_maya_record() -> Dict:
    """Returns a sample Maya record."""
    return {
        "es": "Buenos días",
        "myn": "Ma'alob k'iin",
        "source": "test_source",
        "layer": "diamond",
        "origin_file": "test.jsonl",
    }


@pytest.fixture
def sample_multilingual_records() -> list:
    """Returns a list of sample records with both Náhuatl and Maya."""
    return [
        {
            "es": "Gracias",
            "nah": "Tlazohcamati",
            "source": "test_1",
        },
        {
            "es": "Adiós",
            "nah": "Cualli tonalli",
            "source": "test_1",
        },
        {
            "es": "Buenos días",
            "myn": "Ma'alob k'iin",
            "source": "test_2",
        },
        {
            "es": "Buenas noches",
            "myn": "Ma'alob áak'ab",
            "source": "test_2",
        },
    ]


# ============================================================================
# Temporary file fixtures
# ============================================================================


@pytest.fixture
def temp_jsonl_file(tmp_path) -> Path:
    """Creates a temporary JSONL file for testing."""
    temp_file = tmp_path / "test.jsonl"
    return temp_file


@pytest.fixture
def temp_output_dir(tmp_path) -> Path:
    """Creates a temporary output directory."""
    output_dir = tmp_path / "output"
    output_dir.mkdir(exist_ok=True)
    return output_dir


# ============================================================================
# Data loading fixtures
# ============================================================================


@pytest.fixture
def sample_df() -> pd.DataFrame:
    """Returns a sample DataFrame with test data."""
    return pd.DataFrame([
        {
            "es": "Hola",
            "nah": "Piyali",
            "myn": None,
            "source": "test",
        },
        {
            "es": "Gracias",
            "nah": "Tlazohcamati",
            "myn": None,
            "source": "test",
        },
        {
            "es": "Buenos días",
            "nah": None,
            "myn": "Ma'alob k'iin",
            "source": "test",
        },
    ])


# ============================================================================
# Configuration fixtures
# ============================================================================


@pytest.fixture
def test_config() -> Dict:
    """Returns test configuration settings."""
    return {
        "seed": 42,
        "train_ratio": 0.9,
        "val_ratio": 0.05,
        "test_ratio": 0.05,
        "encoding": "utf-8",
    }


# ============================================================================
# Cleanup fixtures
# ============================================================================


@pytest.fixture(autouse=True)
def cleanup_temp_files():
    """Automatically cleanup temporary files after each test."""
    yield
    # Cleanup code runs after test
    # (Currently no-op, but can be extended)


# ============================================================================
# Markers
# ============================================================================


def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line(
        "markers", "slow: mark test as slow to run"
    )
    config.addinivalue_line(
        "markers", "integration: mark test as integration test"
    )
    config.addinivalue_line(
        "markers", "unit: mark test as unit test"
    )
    config.addinivalue_line(
        "markers", "parity: mark test as parity test with legacy code"
    )
    config.addinivalue_line(
        "markers", "unicode: mark test as Unicode handling test"
    )
    config.addinivalue_line(
        "markers", "quality: mark test as data quality test"
    )


# ============================================================================
# Custom assertions
# ============================================================================


def assert_unicode_preserved(text: str, expected_chars: list):
    """
    Custom assertion to verify Unicode characters are preserved.

    Args:
        text: Text to check
        expected_chars: List of expected Unicode characters
    """
    for char in expected_chars:
        assert char in text, f"Unicode character '{char}' not found in text"


# Add to pytest namespace for easy access
pytest.assert_unicode_preserved = assert_unicode_preserved
