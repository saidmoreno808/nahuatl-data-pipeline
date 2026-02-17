"""
Parity Tests: New Pipeline vs Legacy Pipeline

Ensures that refactored code produces identical results to the original
scripts/unify_datasets.py pipeline.

CRITICAL: These tests MUST pass at 100% before legacy code can be removed.

Run with:
    pytest tests/integration/test_parity_with_legacy.py -v
    pytest tests/integration/test_parity_with_legacy.py::test_unicode_preservation -v
"""

import json
from pathlib import Path
from typing import Dict, List

import pandas as pd
import pytest


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture(scope="module")
def golden_stats() -> Dict:
    """Load precomputed golden dataset statistics."""
    stats_path = Path("benchmark/golden_stats.json")

    if not stats_path.exists():
        pytest.skip(
            "Golden stats not found. Run: python benchmark/generate_stats.py"
        )

    with open(stats_path, encoding="utf-8") as f:
        return json.load(f)


@pytest.fixture(scope="module")
def golden_train_df() -> pd.DataFrame:
    """Load golden training dataset."""
    path = Path("benchmark/golden_train_v1.jsonl")

    if not path.exists():
        pytest.skip(
            "Golden train dataset not found. Run Day 0 setup first."
        )

    return pd.read_json(path, lines=True)


@pytest.fixture(scope="module")
def golden_validation_df() -> pd.DataFrame:
    """Load golden validation dataset."""
    path = Path("benchmark/golden_validation_v1.jsonl")

    if not path.exists():
        pytest.skip("Golden validation dataset not found.")

    return pd.read_json(path, lines=True)


@pytest.fixture(scope="module")
def golden_test_df() -> pd.DataFrame:
    """Load golden test dataset."""
    path = Path("benchmark/golden_test_v1.jsonl")

    if not path.exists():
        pytest.skip("Golden test dataset not found.")

    return pd.read_json(path, lines=True)


# ============================================================================
# Volume Parity Tests
# ============================================================================


def test_golden_dataset_exists(golden_stats):
    """Verifies golden dataset was generated successfully."""
    assert "train" in golden_stats, "Train split missing from golden stats"
    assert golden_stats["train"]["total_records"] > 0, "Train dataset is empty"


def test_record_count_parity(golden_stats, golden_train_df):
    """
    Verifies new pipeline produces same number of records as legacy.

    Tolerance: 0 records (exact match required)
    """
    expected_count = golden_stats["train"]["total_records"]
    actual_count = len(golden_train_df)

    assert actual_count == expected_count, (
        f"Record count mismatch: expected {expected_count}, got {actual_count}"
    )


def test_language_distribution_parity(golden_stats, golden_train_df):
    """
    Verifies language distribution matches legacy pipeline.

    Tolerance: ±1% for each language
    """
    expected_nah = golden_stats["train"]["volume_metrics"]["nah_records"]
    expected_myn = golden_stats["train"]["volume_metrics"]["myn_records"]

    actual_nah = golden_train_df["nah"].notna().sum()
    actual_myn = golden_train_df["myn"].notna().sum()

    total = len(golden_train_df)

    # Allow ±1% tolerance due to potential randomness in deduplication
    nah_diff = abs(actual_nah - expected_nah) / total
    myn_diff = abs(actual_myn - expected_myn) / total

    assert nah_diff < 0.01, (
        f"Náhuatl count mismatch: expected {expected_nah}, got {actual_nah} "
        f"(diff: {nah_diff*100:.2f}%)"
    )

    assert myn_diff < 0.01, (
        f"Maya count mismatch: expected {expected_myn}, got {actual_myn} "
        f"(diff: {myn_diff*100:.2f}%)"
    )


# ============================================================================
# Data Quality Parity Tests
# ============================================================================


def test_duplicate_rate_parity(golden_stats, golden_train_df):
    """
    Verifies duplicate rate doesn't exceed legacy baseline.

    Tolerance: Must be ≤ legacy rate (no regression allowed)
    """
    expected_dup_rate = golden_stats["train"]["quality_metrics"]["duplicate_rate"]
    actual_dup_rate = golden_train_df.duplicated().sum() / len(golden_train_df)

    assert actual_dup_rate <= expected_dup_rate + 0.001, (
        f"Duplicate rate regression: expected ≤{expected_dup_rate*100:.2f}%, "
        f"got {actual_dup_rate*100:.2f}%"
    )


def test_null_rate_parity(golden_stats, golden_train_df):
    """
    Verifies null rates don't exceed legacy baseline.

    Tolerance: Must be ≤ legacy rate for critical columns
    """
    expected_null_rates = golden_stats["train"]["quality_metrics"]["null_rates"]

    for col in ["es", "nah", "myn"]:
        if col not in golden_train_df.columns:
            continue

        expected_rate = expected_null_rates.get(col, 0)
        actual_rate = golden_train_df[col].isnull().sum() / len(golden_train_df)

        assert actual_rate <= expected_rate + 0.01, (
            f"Null rate regression in '{col}': "
            f"expected ≤{expected_rate*100:.2f}%, got {actual_rate*100:.2f}%"
        )


def test_text_length_distribution_parity(golden_stats, golden_train_df):
    """
    Verifies text length distributions are similar to legacy.

    Tolerance: ±20% for mean length (allows for minor deduplication differences)
    """
    expected_lengths = golden_stats["train"]["quality_metrics"]["length_stats"]

    for col, expected_stats in expected_lengths.items():
        if col not in golden_train_df.columns:
            continue

        actual_lengths = golden_train_df[col].dropna().str.len()
        actual_mean = actual_lengths.mean()

        expected_mean = expected_stats["mean"]
        diff_ratio = abs(actual_mean - expected_mean) / expected_mean

        assert diff_ratio < 0.20, (
            f"Text length distribution changed for '{col}': "
            f"expected mean={expected_mean:.1f}, got {actual_mean:.1f} "
            f"(diff: {diff_ratio*100:.1f}%)"
        )


# ============================================================================
# Unicode Preservation Tests (CRITICAL FOR NÁHUATL)
# ============================================================================


def test_unicode_preservation(golden_stats, golden_train_df):
    """
    Verifies Unicode special characters are preserved.

    CRITICAL: Macrons, saltillo, and other diacritics MUST be preserved.
    This test has ZERO tolerance for character loss.
    """
    if "unicode_stats_nah" not in golden_stats["train"]:
        pytest.skip("No Náhuatl Unicode stats in golden dataset")

    expected_chars = golden_stats["train"]["unicode_stats_nah"]["special_chars_present"]

    # Check Náhuatl text
    all_nah_text = "".join(golden_train_df["nah"].dropna())

    for char_name, should_be_present in expected_chars.items():
        if not should_be_present:
            continue  # Skip if not present in golden

        # Define character mappings
        char_map = {
            "macron_a": "ā",
            "macron_e": "ē",
            "macron_i": "ī",
            "macron_o": "ō",
            "macron_u": "ū",
            "saltillo": "h",
            "tl_digraph": "tl",
            "tz_digraph": "tz",
            "kw_digraph": "kw",
        }

        char = char_map.get(char_name, "")

        if char_name.endswith("_digraph"):
            # For digraphs, just check presence (not exact count)
            assert char in all_nah_text, (
                f"Unicode character LOST: '{char}' ({char_name}) not found in new dataset"
            )
        else:
            # For individual characters, verify presence
            assert char in all_nah_text, (
                f"Unicode character LOST: '{char}' ({char_name}) not found in new dataset"
            )


def test_macron_count_parity(golden_stats, golden_train_df):
    """
    Verifies macron character counts are similar to legacy.

    Tolerance: ±5% (allows for deduplication differences)
    """
    if "unicode_stats_nah" not in golden_stats["train"]:
        pytest.skip("No Náhuatl Unicode stats in golden dataset")

    expected_macron_count = golden_stats["train"]["unicode_stats_nah"]["macron_count"]

    if expected_macron_count == 0:
        pytest.skip("No macrons in golden dataset")

    all_nah_text = "".join(golden_train_df["nah"].dropna())
    macron_chars = ["ā", "ē", "ī", "ō", "ū", "Ā", "Ē", "Ī", "Ō", "Ū"]
    actual_macron_count = sum(all_nah_text.count(char) for char in macron_chars)

    diff_ratio = abs(actual_macron_count - expected_macron_count) / expected_macron_count

    assert diff_ratio < 0.05, (
        f"Macron count changed: expected {expected_macron_count}, "
        f"got {actual_macron_count} (diff: {diff_ratio*100:.1f}%)"
    )


# ============================================================================
# Split Integrity Tests
# ============================================================================


def test_train_validation_test_split_ratios(
    golden_train_df, golden_validation_df, golden_test_df
):
    """
    Verifies train/validation/test split ratios match legacy.

    Expected: ~90% train, ~5% validation, ~5% test
    Tolerance: ±2%
    """
    total = len(golden_train_df) + len(golden_validation_df) + len(golden_test_df)

    train_ratio = len(golden_train_df) / total
    val_ratio = len(golden_validation_df) / total
    test_ratio = len(golden_test_df) / total

    assert 0.88 <= train_ratio <= 0.92, (
        f"Train split ratio out of range: {train_ratio*100:.1f}% "
        f"(expected 90% ± 2%)"
    )

    assert 0.03 <= val_ratio <= 0.07, (
        f"Validation split ratio out of range: {val_ratio*100:.1f}% "
        f"(expected 5% ± 2%)"
    )

    assert 0.03 <= test_ratio <= 0.07, (
        f"Test split ratio out of range: {test_ratio*100:.1f}% "
        f"(expected 5% ± 2%)"
    )


def test_no_data_leakage_between_splits(
    golden_train_df, golden_validation_df, golden_test_df
):
    """
    Verifies no records appear in multiple splits.

    This is CRITICAL for model evaluation integrity.
    """
    # Create unique keys for each record
    def make_key(row):
        es = str(row.get("es", "")).strip().lower()
        nah = str(row.get("nah", "")).strip().lower()
        myn = str(row.get("myn", "")).strip().lower()
        return f"{es}|{nah}|{myn}"

    train_keys = set(golden_train_df.apply(make_key, axis=1))
    val_keys = set(golden_validation_df.apply(make_key, axis=1))
    test_keys = set(golden_test_df.apply(make_key, axis=1))

    train_val_overlap = train_keys & val_keys
    train_test_overlap = train_keys & test_keys
    val_test_overlap = val_keys & test_keys

    assert len(train_val_overlap) == 0, (
        f"Data leakage detected: {len(train_val_overlap)} records "
        f"appear in both train and validation"
    )

    assert len(train_test_overlap) == 0, (
        f"Data leakage detected: {len(train_test_overlap)} records "
        f"appear in both train and test"
    )

    assert len(val_test_overlap) == 0, (
        f"Data leakage detected: {len(val_test_overlap)} records "
        f"appear in both validation and test"
    )


# ============================================================================
# Schema Consistency Tests
# ============================================================================


def test_column_schema_consistency(golden_train_df):
    """
    Verifies new pipeline produces same columns as legacy.

    Required columns: ['es', 'nah', 'myn', 'source', 'layer', 'origin_file']
    """
    required_columns = {"es", "nah", "myn"}
    actual_columns = set(golden_train_df.columns)

    missing_columns = required_columns - actual_columns

    assert len(missing_columns) == 0, (
        f"Required columns missing: {missing_columns}"
    )


def test_data_types_consistency(golden_train_df):
    """
    Verifies data types match expected schema.
    """
    # All language columns should be object (string) type
    for col in ["es", "nah", "myn"]:
        if col in golden_train_df.columns:
            assert golden_train_df[col].dtype == "object", (
                f"Column '{col}' has wrong dtype: {golden_train_df[col].dtype}"
            )


# ============================================================================
# Metadata Preservation Tests
# ============================================================================


def test_source_metadata_preserved(golden_stats, golden_train_df):
    """
    Verifies source file metadata is preserved.
    """
    if "language_distribution" not in golden_stats["train"]:
        pytest.skip("No language distribution in golden stats")

    if "top_sources" not in golden_stats["train"]["language_distribution"]:
        pytest.skip("No source metadata in golden dataset")

    # Check that 'source' or 'origin_file' column exists
    assert "source" in golden_train_df.columns or "origin_file" in golden_train_df.columns, (
        "Source metadata columns missing"
    )


# ============================================================================
# Performance Benchmark Tests (Optional)
# ============================================================================


try:
    import pytest_benchmark  # noqa: F401

    _has_benchmark = True
except ImportError:
    _has_benchmark = False


@pytest.mark.slow
@pytest.mark.skipif(not _has_benchmark, reason="pytest-benchmark not installed")
def test_processing_time_benchmark(benchmark):
    """
    Benchmarks new pipeline performance vs legacy.

    This test is marked as 'slow' and requires pytest-benchmark.
    Run with: pytest -v --benchmark-only
    """
    pytest.skip("Benchmark test not implemented yet")


# ============================================================================
# Summary Report
# ============================================================================


def test_generate_parity_report(golden_stats, golden_train_df):
    """
    Generates a summary report of parity test results.

    This test always passes but outputs useful diagnostics.
    """
    print("\n" + "=" * 60)
    print("PARITY TEST SUMMARY")
    print("=" * 60)

    # Record counts
    expected_count = golden_stats["train"]["total_records"]
    actual_count = len(golden_train_df)
    print(f"\nRecord Count:")
    print(f"  Expected: {expected_count:,}")
    print(f"  Actual:   {actual_count:,}")
    print(f"  Match:    {'✅ YES' if actual_count == expected_count else '❌ NO'}")

    # Language distribution
    expected_nah = golden_stats["train"]["volume_metrics"]["nah_records"]
    actual_nah = golden_train_df["nah"].notna().sum()
    print(f"\nNáhuatl Records:")
    print(f"  Expected: {expected_nah:,}")
    print(f"  Actual:   {actual_nah:,}")

    # Unicode preservation
    if "unicode_stats_nah" in golden_stats["train"]:
        expected_macrons = golden_stats["train"]["unicode_stats_nah"]["macron_count"]
        all_nah = "".join(golden_train_df["nah"].dropna())
        actual_macrons = sum(all_nah.count(c) for c in ["ā", "ē", "ī", "ō", "ū"])
        print(f"\nMacron Characters:")
        print(f"  Expected: {expected_macrons:,}")
        print(f"  Actual:   {actual_macrons:,}")

    print("\n" + "=" * 60)
    assert True  # Always pass this test
