"""
Unit tests for data schemas.

Run with:
    pytest tests/unit/test_schemas.py -v
"""

import pytest

from src.models.enums import DataLayer, DataSource, Language
from src.models.schemas import Record, RecordMetadata


class TestRecord:
    """Test Record model."""

    def test_create_basic_record(self):
        """Test creating a basic record."""
        record = Record(
            es="Hola",
            nah="Piyali",
            source=DataSource.MANUAL,
            layer=DataLayer.DIAMOND,
        )

        assert record.es == "Hola"
        assert record.nah == "Piyali"
        assert record.source == DataSource.MANUAL
        assert record.layer == DataLayer.DIAMOND

    def test_strip_whitespace(self):
        """Test that whitespace is automatically stripped."""
        record = Record(
            es="  Hola  ",
            nah="  Piyali  ",
            source="manual",
            layer="diamond",
        )

        assert record.es == "Hola"
        assert record.nah == "Piyali"

    def test_empty_string_to_none(self):
        """Test that empty strings become None."""
        record = Record(
            es="",
            nah="Piyali",
            source="manual",
            layer="diamond",
        )

        assert record.es is None
        assert record.nah == "Piyali"

    def test_has_methods(self):
        """Test has_* convenience methods."""
        record = Record(
            es="Hola",
            nah="Piyali",
            myn=None,
            source="manual",
            layer="diamond",
        )

        assert record.has_spanish()
        assert record.has_nahuatl()
        assert not record.has_maya()
        assert record.has_translation_pair()

    def test_get_dedup_key(self):
        """Test deduplication key generation."""
        record = Record(
            es="HOLA",
            nah="Piyali",
            source="manual",
            layer="diamond",
        )

        key = record.get_dedup_key()
        assert key == "hola|piyali|"
        assert key.islower()

    def test_to_dict(self):
        """Test conversion to dictionary."""
        record = Record(
            es="Hola",
            nah="Piyali",
            source="manual",
            layer="diamond",
            origin_file="test.jsonl",
        )

        data = record.to_dict()

        assert data["es"] == "Hola"
        assert data["nah"] == "Piyali"
        assert data["source"] == "manual"
        assert data["layer"] == "diamond"
        assert data["origin_file"] == "test.jsonl"

    def test_from_dict(self):
        """Test creating record from dictionary."""
        data = {
            "es": "Hola",
            "nah": "Piyali",
            "source": "manual",
            "layer": "diamond",
        }

        record = Record.from_dict(data)

        assert record.es == "Hola"
        assert record.nah == "Piyali"

    def test_from_legacy_format_basic(self):
        """Test conversion from legacy format."""
        legacy_data = {
            "es_translation": "Hola",
            "nah_translation": "Piyali",
            "source_file": "legacy.json",
            "layer": "silver",
        }

        record = Record.from_legacy_format(legacy_data)

        assert record.es == "Hola"
        assert record.nah == "Piyali"
        assert record.origin_file == "legacy.json"
        assert record.layer == DataLayer.SILVER

    def test_from_legacy_format_audio(self):
        """Test conversion from audio transcript format."""
        legacy_data = {
            "original_audio_text": "Piyali",
            "detected_language": "nah",
            "original_es": "Hola",
        }

        record = Record.from_legacy_format(legacy_data)

        assert record.es == "Hola"
        assert record.nah == "Piyali"

    def test_from_legacy_format_dpo(self):
        """Test conversion from DPO format."""
        legacy_data = {
            "prompt": "¿Cómo estás?",
            "chosen": "Quēnin timotlaneltoquia?",
        }

        record = Record.from_legacy_format(legacy_data)

        assert record.es == "¿Cómo estás?"
        assert record.nah == "Quēnin timotlaneltoquia?"

    def test_to_jsonl(self):
        """Test JSONL serialization."""
        record = Record(
            es="Hola",
            nah="Piyali",
            source="manual",
            layer="diamond",
        )

        jsonl = record.to_jsonl()

        assert '"es": "Hola"' in jsonl or '"es":"Hola"' in jsonl
        assert '"nah": "Piyali"' in jsonl or '"nah":"Piyali"' in jsonl


class TestRecordMetadata:
    """Test RecordMetadata model."""

    def test_create_metadata(self):
        """Test creating metadata."""
        metadata = RecordMetadata(
            source=DataSource.MANUAL,
            layer=DataLayer.DIAMOND,
            is_validated=True,
            quality_score=0.95,
        )

        assert metadata.source == DataSource.MANUAL
        assert metadata.layer == DataLayer.DIAMOND
        assert metadata.is_validated is True
        assert metadata.quality_score == 0.95

    def test_quality_score_validation(self):
        """Test quality score must be between 0 and 1."""
        # Valid score
        metadata = RecordMetadata(quality_score=0.5)
        assert metadata.quality_score == 0.5

        # Invalid score (too high)
        with pytest.raises(ValueError):
            RecordMetadata(quality_score=1.5)

        # Invalid score (negative)
        with pytest.raises(ValueError):
            RecordMetadata(quality_score=-0.1)


class TestEnums:
    """Test enum types."""

    def test_language_enum(self):
        """Test Language enum."""
        assert Language.SPANISH == "es"
        assert Language.NAHUATL == "nah"
        assert Language.MAYA == "myn"

        assert str(Language.SPANISH) == "es"

    def test_data_layer_enum(self):
        """Test DataLayer enum."""
        assert DataLayer.BRONZE == "bronze"
        assert DataLayer.SILVER == "silver"
        assert DataLayer.DIAMOND == "diamond"
        assert DataLayer.GOLD == "gold"

    def test_data_layer_priority(self):
        """Test DataLayer priority property."""
        assert DataLayer.DIAMOND.priority > DataLayer.SILVER.priority
        assert DataLayer.SILVER.priority > DataLayer.BRONZE.priority
        assert DataLayer.GOLD.priority > DataLayer.DIAMOND.priority

    def test_data_source_enum(self):
        """Test DataSource enum."""
        assert DataSource.HUGGINGFACE == "huggingface"
        assert DataSource.YOUTUBE == "youtube"
        assert DataSource.MANUAL == "manual"
