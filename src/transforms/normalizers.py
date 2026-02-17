"""
Text Normalization

Unicode normalization for multilingual text, with special handling
for Náhuatl macrons and Maya diacritics.

CRITICAL: This module preserves linguistic features that are easily lost
in naive normalization (e.g., macrons indicating vowel length in Náhuatl).

Example:
    >>> normalizer = TextNormalizer()
    >>> text = "Quēnin timotlaneltoquia?"  # Note macron on 'ē'
    >>> normalized = normalizer.normalize(text, language="nah")
    >>> assert 'ē' in normalized  # Macron preserved!
"""

import re
import unicodedata
from typing import Optional

from src.models.enums import Language
from src.models.schemas import Record
from src.utils.config import get_settings
from src.utils.logger import get_logger

logger = get_logger(__name__)


class TextNormalizer:
    """
    Text normalizer with language-specific handling.

    Preserves linguistic features critical for indigenous languages:
    - Náhuatl: Macrons (ā, ē, ī, ō, ū) for vowel length
    - Maya: Glottal stops ('), ejectives
    - Both: Special orthographic conventions

    Example:
        >>> normalizer = TextNormalizer(form="NFC")
        >>> text = normalizer.normalize("Piyali", language="nah")
    """

    def __init__(self, form: Optional[str] = None):
        """
        Initialize normalizer.

        Args:
            form: Unicode normalization form (NFC, NFD, NFKC, NFKD).
                  Defaults to settings.unicode_normalization.
                  Use NFC to preserve composed characters (e.g., macrons).
        """
        settings = get_settings()
        self.form = form or settings.unicode_normalization

        # Validate form
        valid_forms = ["NFC", "NFD", "NFKC", "NFKD"]
        if self.form not in valid_forms:
            raise ValueError(
                f"Invalid normalization form '{self.form}'. "
                f"Must be one of {valid_forms}"
            )

        logger.debug(f"TextNormalizer initialized with form={self.form}")

    def normalize(
        self,
        text: str,
        language: Optional[str] = None,
        strip_whitespace: bool = True,
    ) -> str:
        """
        Normalize text with language-specific handling.

        Args:
            text: Text to normalize
            language: Language code (for language-specific handling)
            strip_whitespace: Whether to strip leading/trailing whitespace

        Returns:
            str: Normalized text

        Example:
            >>> normalizer = TextNormalizer()
            >>> normalizer.normalize("  Cualli tonalli  ", language="nah")
            'Cualli tonalli'
        """
        if not text:
            return text

        # Strip whitespace
        if strip_whitespace:
            text = text.strip()

        # Apply Unicode normalization
        # CRITICAL: NFC preserves composed characters (macrons)
        #           NFD decomposes them (loses macrons!)
        text = unicodedata.normalize(self.form, text)

        # Language-specific normalization
        if language == Language.NAHUATL or language == "nah":
            text = self._normalize_nahuatl(text)
        elif language == Language.MAYA or language == "myn":
            text = self._normalize_maya(text)
        elif language == Language.SPANISH or language == "es":
            text = self._normalize_spanish(text)

        # Normalize whitespace (collapse multiple spaces)
        text = re.sub(r"\s+", " ", text).strip()

        return text

    def _normalize_nahuatl(self, text: str) -> str:
        """
        Náhuatl-specific normalization.

        Preserves:
        - Macrons (ā, ē, ī, ō, ū) - indicate vowel length
        - Saltillo (h, H, ') - glottal stop
        - Digraphs (tl, tz, kw, etc.)

        Args:
            text: Náhuatl text

        Returns:
            str: Normalized Náhuatl text
        """
        # Preserve macrons (already handled by NFC normalization)
        # No additional processing needed if using NFC

        # Normalize saltillo variations
        # Some orthographies use different representations
        # Common variations: h, H, ', ʔ
        # We keep original for now (don't force one convention)

        # Normalize common alternative spellings
        # (Only if explicitly needed for deduplication)

        return text

    def _normalize_maya(self, text: str) -> str:
        """
        Maya-specific normalization.

        Preserves:
        - Glottal stops (')
        - Ejectives (marked with ')
        - Special orthography

        Args:
            text: Maya text

        Returns:
            str: Normalized Maya text
        """
        # Preserve glottal stops and ejectives
        # Common in Maya languages: k', ch', t', p', etc.

        return text

    def _normalize_spanish(self, text: str) -> str:
        """
        Spanish-specific normalization.

        Preserves:
        - Accents (á, é, í, ó, ú, ñ)
        - Inverted punctuation (¿, ¡)

        Args:
            text: Spanish text

        Returns:
            str: Normalized Spanish text
        """
        # Spanish accents are already preserved by NFC normalization

        return text

    def normalize_batch(
        self,
        texts: list[str],
        language: Optional[str] = None,
    ) -> list[str]:
        """
        Normalize multiple texts.

        Args:
            texts: List of texts to normalize
            language: Language code (applied to all texts)

        Returns:
            list: Normalized texts

        Example:
            >>> normalizer = TextNormalizer()
            >>> texts = ["Piyali", "Cualli tonalli"]
            >>> normalizer.normalize_batch(texts, language="nah")
            ['Piyali', 'Cualli tonalli']
        """
        return [self.normalize(text, language=language) for text in texts]


def normalize_record(
    record: Record,
    normalizer: Optional[TextNormalizer] = None,
) -> Record:
    """
    Normalize all text fields in a record.

    Args:
        record: Record to normalize
        normalizer: TextNormalizer instance (creates new one if None)

    Returns:
        Record: Record with normalized text

    Example:
        >>> record = Record(es="  Hola  ", nah="  Piyali  ")
        >>> normalized = normalize_record(record)
        >>> normalized.es
        'Hola'
    """
    if normalizer is None:
        normalizer = TextNormalizer()

    # Normalize Spanish
    if record.es:
        record.es = normalizer.normalize(record.es, language=Language.SPANISH)

    # Normalize Náhuatl
    if record.nah:
        record.nah = normalizer.normalize(record.nah, language=Language.NAHUATL)

    # Normalize Maya
    if record.myn:
        record.myn = normalizer.normalize(record.myn, language=Language.MAYA)

    return record


def detect_language_by_characters(text: str) -> Optional[str]:
    """
    Detect language based on character patterns.

    Heuristic detection based on:
    - Náhuatl: macrons (ā, ē, ī, ō, ū), digraphs (tl, tz, kw)
    - Maya: glottal stops ('), ejectives (k', ch', t')
    - Spanish: inverted punctuation (¿, ¡), accents without macrons

    Args:
        text: Text to analyze

    Returns:
        str: Detected language code or None

    Example:
        >>> detect_language_by_characters("Quēnin timotlaneltoquia?")
        'nah'  # Detected macron 'ē'
    """
    text_lower = text.lower()

    # Check for Náhuatl macrons
    nahuatl_macrons = ["ā", "ē", "ī", "ō", "ū"]
    if any(macron in text for macron in nahuatl_macrons):
        return "nah"

    # Check for Náhuatl digraphs
    nahuatl_digraphs = ["tl", "tz", "kw", "ku"]
    nahuatl_score = sum(
        text_lower.count(digraph) for digraph in nahuatl_digraphs
    )

    # Check for Maya glottal stops and ejectives
    maya_patterns = ["k'", "ch'", "t'", "p'", "ts'", "ʔ"]
    maya_score = sum(text.count(pattern) for pattern in maya_patterns)

    # Threshold-based decision (lowered thresholds for better detection)
    if nahuatl_score >= 1:
        return "nah"
    elif maya_score >= 1:
        return "myn"

    # Check for Spanish patterns
    spanish_patterns = ["¿", "¡", "ñ"]
    if any(pattern in text for pattern in spanish_patterns):
        return "es"

    # Default: unable to detect
    return None


# Example usage
if __name__ == "__main__":
    from src.utils.logger import configure_root_logger

    configure_root_logger()

    # Test normalizer
    normalizer = TextNormalizer(form="NFC")

    print("=== Text Normalization Tests ===\n")

    # Test 1: Náhuatl with macrons
    print("Test 1: Náhuatl with macrons")
    nah_text = "Quēnin timotlaneltoquia?"  # Note: ē has macron
    normalized = normalizer.normalize(nah_text, language="nah")
    print(f"  Original:   {nah_text}")
    print(f"  Normalized: {normalized}")
    print(f"  Macron preserved: {'ē' in normalized}")

    # Test 2: Whitespace normalization
    print("\nTest 2: Whitespace normalization")
    text_with_spaces = "  Cualli    tonalli  "
    normalized = normalizer.normalize(text_with_spaces, language="nah")
    print(f"  Original:   '{text_with_spaces}'")
    print(f"  Normalized: '{normalized}'")

    # Test 3: Spanish accents
    print("\nTest 3: Spanish accents")
    es_text = "¿Cómo estás?"
    normalized = normalizer.normalize(es_text, language="es")
    print(f"  Original:   {es_text}")
    print(f"  Normalized: {normalized}")
    print(f"  Accents preserved: {'ó' in normalized and 'á' in normalized}")

    # Test 4: Record normalization
    print("\nTest 4: Record normalization")
    record = Record(
        es="  ¿Cómo  estás?  ",
        nah="  Quēnin timotlaneltoquia?  ",
        source="test",
        layer="bronze",
    )
    normalized_record = normalize_record(record)
    print(f"  Original ES:    '{record.es}'")
    print(f"  Normalized ES:  '{normalized_record.es}'")
    print(f"  Original NAH:   '{record.nah}'")
    print(f"  Normalized NAH: '{normalized_record.nah}'")

    # Test 5: Language detection
    print("\nTest 5: Language detection")
    texts = [
        ("Quēnin timotlaneltoquia?", "nah (macron detected)"),
        ("Ma'alob k'iin", "myn (glottal stop detected)"),
        ("¿Cómo estás?", "es (Spanish punctuation)"),
        ("Cualli tonalli", "nah (tl digraph)"),
    ]
    for text, expected in texts:
        detected = detect_language_by_characters(text)
        print(f"  '{text}' → {detected} (expected: {expected})")

    print("\n✓ Normalization tests complete!")
