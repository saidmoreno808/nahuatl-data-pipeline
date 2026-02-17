"""
Unit tests for text normalizers.

Run with:
    pytest tests/unit/test_normalizers.py -v
"""

import pytest

from src.models.enums import DataSource, DataLayer
from src.models.schemas import Record
from src.transforms.normalizers import (
    TextNormalizer,
    detect_language_by_characters,
    normalize_record,
)


class TestTextNormalizer:
    """Test TextNormalizer class."""

    def test_create_normalizer(self):
        """Test creating normalizer with default settings."""
        normalizer = TextNormalizer()
        assert normalizer.form == "NFC"

    def test_create_normalizer_custom_form(self):
        """Test creating normalizer with custom form."""
        normalizer = TextNormalizer(form="NFD")
        assert normalizer.form == "NFD"

    def test_invalid_normalization_form(self):
        """Test that invalid form raises error."""
        with pytest.raises(ValueError, match="Invalid normalization form"):
            TextNormalizer(form="INVALID")

    def test_normalize_basic(self):
        """Test basic text normalization."""
        normalizer = TextNormalizer()
        text = "  Hola  mundo  "
        normalized = normalizer.normalize(text)

        assert normalized == "Hola mundo"
        assert normalized.strip() == normalized

    def test_normalize_whitespace_collapse(self):
        """Test that multiple spaces are collapsed."""
        normalizer = TextNormalizer()
        text = "Hola    mundo"
        normalized = normalizer.normalize(text)

        assert normalized == "Hola mundo"
        assert "  " not in normalized

    def test_normalize_nahuatl_preserves_macrons(self):
        """Test that Náhuatl macrons are preserved (CRITICAL)."""
        normalizer = TextNormalizer(form="NFC")
        text = "Quēnin timotlaneltoquia?"  # Note: ē has macron

        normalized = normalizer.normalize(text, language="nah")

        # CRITICAL: Macron MUST be preserved
        assert "ē" in normalized
        assert normalized == "Quēnin timotlaneltoquia?"

    def test_normalize_all_nahuatl_macrons(self):
        """Test that all Náhuatl macrons are preserved."""
        normalizer = TextNormalizer(form="NFC")
        text = "ā ē ī ō ū"  # All macron vowels

        normalized = normalizer.normalize(text, language="nah")

        assert "ā" in normalized
        assert "ē" in normalized
        assert "ī" in normalized
        assert "ō" in normalized
        assert "ū" in normalized

    def test_normalize_spanish_preserves_accents(self):
        """Test that Spanish accents are preserved."""
        normalizer = TextNormalizer(form="NFC")
        text = "¿Cómo estás?"

        normalized = normalizer.normalize(text, language="es")

        assert "ó" in normalized
        assert "á" in normalized
        assert "¿" in normalized

    def test_normalize_maya_preserves_glottal_stops(self):
        """Test that Maya glottal stops are preserved."""
        normalizer = TextNormalizer(form="NFC")
        text = "Ma'alob k'iin"  # Maya greeting

        normalized = normalizer.normalize(text, language="myn")

        assert "'" in normalized
        assert normalized == "Ma'alob k'iin"

    def test_normalize_batch(self):
        """Test batch normalization."""
        normalizer = TextNormalizer()
        texts = ["  Hola  ", "  Mundo  ", "  Test  "]

        normalized = normalizer.normalize_batch(texts)

        assert len(normalized) == 3
        assert normalized[0] == "Hola"
        assert normalized[1] == "Mundo"
        assert normalized[2] == "Test"


class TestNormalizeRecord:
    """Test normalize_record function."""

    def test_normalize_record_basic(self):
        """Test normalizing a basic record."""
        record = Record(
            es="  Hola  ",
            nah="  Piyali  ",
            source=DataSource.MANUAL,
            layer=DataLayer.BRONZE,
        )

        normalized = normalize_record(record)

        assert normalized.es == "Hola"
        assert normalized.nah == "Piyali"

    def test_normalize_record_preserves_macrons(self):
        """Test that record normalization preserves macrons."""
        record = Record(
            es="¿Cómo estás?",
            nah="Quēnin timotlaneltoquia?",
            source=DataSource.MANUAL,
            layer=DataLayer.BRONZE,
        )

        normalized = normalize_record(record)

        # CRITICAL: Macron must be preserved
        assert "ē" in normalized.nah
        assert "ó" in normalized.es
        assert "á" in normalized.es

    def test_normalize_record_with_none_values(self):
        """Test normalizing record with None values."""
        record = Record(
            es="Hola",
            nah=None,
            myn=None,
            source=DataSource.MANUAL,
            layer=DataLayer.BRONZE,
        )

        normalized = normalize_record(record)

        assert normalized.es == "Hola"
        assert normalized.nah is None
        assert normalized.myn is None


class TestDetectLanguageByCharacters:
    """Test language detection function."""

    def test_detect_nahuatl_by_macron(self):
        """Test detecting Náhuatl by macron presence."""
        text = "Quēnin timotlaneltoquia?"
        detected = detect_language_by_characters(text)

        assert detected == "nah"

    def test_detect_nahuatl_by_digraphs(self):
        """Test detecting Náhuatl by digraph frequency."""
        text = "Cualli tonalli tlazohcamati"  # Contains 'tl' digraphs
        detected = detect_language_by_characters(text)

        assert detected == "nah"

    def test_detect_maya_by_glottal_stops(self):
        """Test detecting Maya by glottal stops."""
        text = "Ma'alob k'iin"
        detected = detect_language_by_characters(text)

        assert detected == "myn"

    def test_detect_spanish_by_punctuation(self):
        """Test detecting Spanish by inverted punctuation."""
        text = "¿Cómo estás?"
        detected = detect_language_by_characters(text)

        assert detected == "es"

    def test_detect_spanish_by_ñ(self):
        """Test detecting Spanish by ñ."""
        text = "Mañana"
        detected = detect_language_by_characters(text)

        assert detected == "es"

    def test_detect_unknown(self):
        """Test that ambiguous text returns None."""
        text = "Hello world"  # English - not supported
        detected = detect_language_by_characters(text)

        assert detected is None


class TestUnicodePreservation:
    """Test that Unicode normalization preserves critical characters."""

    def test_nfc_preserves_composed_characters(self):
        """Test that NFC preserves composed characters."""
        normalizer = TextNormalizer(form="NFC")

        # Test with composed characters
        text = "ā ē ī ō ū"  # Composed (single codepoint each)
        normalized = normalizer.normalize(text)

        # Should remain composed
        assert "ā" in normalized
        assert len("ā") == 1  # Single codepoint

    def test_nfd_decomposes_characters(self):
        """Test that NFD decomposes characters (for comparison)."""
        normalizer = TextNormalizer(form="NFD")

        # Test with composed characters
        text = "ā"  # Composed
        normalized = normalizer.normalize(text)

        # NFD decomposes into base + combining mark
        # Length will be > 1 if decomposed
        assert len(normalized) > len(text)  # Decomposed

    def test_all_nahuatl_special_chars(self):
        """Test comprehensive Náhuatl character preservation."""
        normalizer = TextNormalizer(form="NFC")

        # All special Náhuatl characters
        text = "ā ē ī ō ū Ā Ē Ī Ō Ū tl tz kw h"
        normalized = normalizer.normalize(text, language="nah")

        # All should be preserved
        for char in ["ā", "ē", "ī", "ō", "ū", "tl", "tz", "kw"]:
            assert char in normalized.lower()
