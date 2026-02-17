"""
Unit tests for deduplicators.

Run with:
    pytest tests/unit/test_deduplicators.py -v
"""

import pytest

from src.models.enums import DataLayer, DataSource
from src.models.schemas import Record
from src.transforms.deduplicators import Deduplicator, deduplicate_records


class TestDeduplicator:
    """Test Deduplicator class."""

    def test_create_deduplicator(self):
        """Test creating deduplicator."""
        dedup = Deduplicator()
        assert dedup.case_sensitive is False

    def test_create_case_sensitive_deduplicator(self):
        """Test creating case-sensitive deduplicator."""
        dedup = Deduplicator(case_sensitive=True)
        assert dedup.case_sensitive is True

    def test_deduplicate_no_duplicates(self):
        """Test deduplication with no duplicates."""
        records = [
            Record(es="Hola", nah="Piyali", source=DataSource.MANUAL, layer=DataLayer.SILVER),
            Record(es="Adiós", nah="Cualli", source=DataSource.MANUAL, layer=DataLayer.SILVER),
        ]

        dedup = Deduplicator()
        unique = dedup.deduplicate(records)

        assert len(unique) == 2

    def test_deduplicate_exact_duplicates(self):
        """Test deduplication with exact duplicates."""
        records = [
            Record(es="Hola", nah="Piyali", source=DataSource.MANUAL, layer=DataLayer.SILVER),
            Record(es="Hola", nah="Piyali", source=DataSource.MANUAL, layer=DataLayer.SILVER),
        ]

        dedup = Deduplicator()
        unique = dedup.deduplicate(records, keep="last")

        assert len(unique) == 1

    def test_deduplicate_case_insensitive(self):
        """Test case-insensitive deduplication (default)."""
        records = [
            Record(es="HOLA", nah="Piyali", source=DataSource.MANUAL, layer=DataLayer.SILVER),
            Record(es="hola", nah="Piyali", source=DataSource.MANUAL, layer=DataLayer.DIAMOND),
        ]

        dedup = Deduplicator(case_sensitive=False)
        unique = dedup.deduplicate(records, keep="best")

        assert len(unique) == 1
        # Should keep Diamond version
        assert unique[0].layer == DataLayer.DIAMOND

    def test_deduplicate_case_sensitive(self):
        """Test case-sensitive deduplication."""
        records = [
            Record(es="HOLA", nah="Piyali", source=DataSource.MANUAL, layer=DataLayer.SILVER),
            Record(es="hola", nah="Piyali", source=DataSource.MANUAL, layer=DataLayer.DIAMOND),
        ]

        dedup = Deduplicator(case_sensitive=True)
        unique = dedup.deduplicate(records)

        # Should keep both (different cases)
        assert len(unique) == 2

    def test_deduplicate_keep_first(self):
        """Test keeping first occurrence."""
        records = [
            Record(es="Hola", nah="Piyali", source=DataSource.MANUAL, layer=DataLayer.SILVER),
            Record(es="Hola", nah="Piyali", source=DataSource.MANUAL, layer=DataLayer.DIAMOND),
        ]

        dedup = Deduplicator()
        unique = dedup.deduplicate(records, keep="first")

        assert len(unique) == 1
        assert unique[0].layer == DataLayer.SILVER  # First

    def test_deduplicate_keep_last(self):
        """Test keeping last occurrence."""
        records = [
            Record(es="Hola", nah="Piyali", source=DataSource.MANUAL, layer=DataLayer.SILVER),
            Record(es="Hola", nah="Piyali", source=DataSource.MANUAL, layer=DataLayer.DIAMOND),
        ]

        dedup = Deduplicator()
        unique = dedup.deduplicate(records, keep="last")

        assert len(unique) == 1
        assert unique[0].layer == DataLayer.DIAMOND  # Last

    def test_deduplicate_keep_best(self):
        """Test keeping best quality record."""
        records = [
            Record(es="Hola", nah="Piyali", source=DataSource.MANUAL, layer=DataLayer.BRONZE),
            Record(es="Hola", nah="Piyali", source=DataSource.MANUAL, layer=DataLayer.SILVER),
            Record(es="Hola", nah="Piyali", source=DataSource.MANUAL, layer=DataLayer.DIAMOND),
        ]

        dedup = Deduplicator()
        unique = dedup.deduplicate(records, keep="best")

        assert len(unique) == 1
        # Should keep highest-priority layer (Diamond)
        assert unique[0].layer == DataLayer.DIAMOND

    def test_deduplicate_whitespace_normalized(self):
        """Test that whitespace differences are handled."""
        records = [
            Record(es="Hola", nah="Piyali", source=DataSource.MANUAL, layer=DataLayer.SILVER),
            Record(es="  Hola  ", nah="  Piyali  ", source=DataSource.MANUAL, layer=DataLayer.DIAMOND),
        ]

        dedup = Deduplicator()
        unique = dedup.deduplicate(records, keep="best")

        # Should be considered duplicates despite whitespace
        assert len(unique) == 1

    def test_get_duplicate_stats(self):
        """Test getting duplicate statistics."""
        records = [
            Record(es="Hola", nah="Piyali", source=DataSource.MANUAL, layer=DataLayer.SILVER),
            Record(es="Hola", nah="Piyali", source=DataSource.MANUAL, layer=DataLayer.DIAMOND),
            Record(es="Adiós", nah="Cualli", source=DataSource.MANUAL, layer=DataLayer.SILVER),
        ]

        dedup = Deduplicator()
        stats = dedup.get_duplicate_stats(records)

        assert stats["total_records"] == 3
        assert stats["unique_records"] == 2
        assert stats["duplicate_records"] == 1
        assert stats["duplicate_rate"] == 1 / 3
        assert stats["duplicate_groups"] == 1

    def test_empty_records_list(self):
        """Test deduplication with empty list."""
        dedup = Deduplicator()
        unique = dedup.deduplicate([])

        assert unique == []

    def test_invalid_keep_strategy(self):
        """Test that invalid keep strategy raises error."""
        records = [
            Record(es="Hola", nah="Piyali", source=DataSource.MANUAL, layer=DataLayer.SILVER),
        ]

        dedup = Deduplicator()

        with pytest.raises(ValueError, match="Invalid keep strategy"):
            dedup.deduplicate(records, keep="invalid")


class TestSelectBestRecord:
    """Test _select_best_record method."""

    def test_select_by_layer_priority(self):
        """Test selection prioritizes layer."""
        dedup = Deduplicator()

        records = [
            Record(es="Test", nah="Test", source=DataSource.MANUAL, layer=DataLayer.BRONZE),
            Record(es="Test", nah="Test", source=DataSource.MANUAL, layer=DataLayer.DIAMOND),
            Record(es="Test", nah="Test", source=DataSource.MANUAL, layer=DataLayer.SILVER),
        ]

        best = dedup._select_best_record(records)
        assert best.layer == DataLayer.DIAMOND

    def test_select_by_completeness(self):
        """Test selection prioritizes completeness."""
        dedup = Deduplicator()

        records = [
            Record(es="Test", nah=None, source=DataSource.MANUAL, layer=DataLayer.SILVER),
            Record(
                es="Test",
                nah="Test",
                myn="Test",
                source=DataSource.MANUAL,
                layer=DataLayer.SILVER,
            ),
        ]

        best = dedup._select_best_record(records)
        # Should prefer the more complete record
        assert best.has_maya()

    def test_select_by_length(self):
        """Test selection prioritizes longer text."""
        dedup = Deduplicator()

        records = [
            Record(es="Hi", nah="Hi", source=DataSource.MANUAL, layer=DataLayer.SILVER),
            Record(
                es="Hello world",
                nah="Hello world",
                source=DataSource.MANUAL,
                layer=DataLayer.SILVER,
            ),
        ]

        best = dedup._select_best_record(records)
        # Should prefer longer text
        assert len(best.es) > 5


class TestDeduplicateRecordsFunction:
    """Test convenience function."""

    def test_deduplicate_records_function(self):
        """Test deduplicate_records convenience function."""
        records = [
            Record(es="Hola", nah="Piyali", source=DataSource.MANUAL, layer=DataLayer.SILVER),
            Record(es="Hola", nah="Piyali", source=DataSource.MANUAL, layer=DataLayer.DIAMOND),
        ]

        unique = deduplicate_records(records, keep="best")

        assert len(unique) == 1
        assert unique[0].layer == DataLayer.DIAMOND


class TestDeduplicationWithRealData:
    """Test deduplication with realistic scenarios."""

    def test_multilingual_duplicates(self):
        """Test deduplication with multilingual records."""
        records = [
            # Same Spanish + Náhuatl
            Record(
                es="Buenos días",
                nah="Cualli tonalli",
                source=DataSource.MANUAL,
                layer=DataLayer.SILVER,
            ),
            Record(
                es="Buenos días",
                nah="Cualli tonalli",
                source=DataSource.MANUAL,
                layer=DataLayer.DIAMOND,
            ),
            # Same Spanish, different Náhuatl (not duplicates)
            Record(
                es="Buenos días",
                nah="Piyali",
                source=DataSource.MANUAL,
                layer=DataLayer.SILVER,
            ),
            # Completely different
            Record(
                es="Adiós",
                nah="Ximopanolti",
                source=DataSource.MANUAL,
                layer=DataLayer.SILVER,
            ),
        ]

        dedup = Deduplicator()
        unique = dedup.deduplicate(records, keep="best")

        # Should have 3 unique records
        assert len(unique) == 3

    def test_partial_content_duplicates(self):
        """Test deduplication with partial content matches."""
        records = [
            # Full record
            Record(
                es="Hola",
                nah="Piyali",
                myn="Ma'alob",
                source=DataSource.MANUAL,
                layer=DataLayer.DIAMOND,
            ),
            # Missing Maya (not a duplicate)
            Record(
                es="Hola",
                nah="Piyali",
                myn=None,
                source=DataSource.MANUAL,
                layer=DataLayer.SILVER,
            ),
        ]

        dedup = Deduplicator()
        unique = dedup.deduplicate(records, keep="best")

        # Should keep both (different content)
        assert len(unique) == 2
