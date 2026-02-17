"""
Enumerations for CORC-NAH data models.

Provides type-safe enums for languages, data sources, and pipeline layers.
"""

from enum import Enum


class Language(str, Enum):
    """Supported languages in the pipeline."""

    SPANISH = "es"
    NAHUATL = "nah"
    MAYA = "myn"

    def __str__(self) -> str:
        return self.value


class DataSource(str, Enum):
    """Data source types."""

    HUGGINGFACE = "huggingface"
    YOUTUBE = "youtube"
    PDF = "pdf"
    MANUAL = "manual"
    SYNTHETIC = "synthetic"
    BIBLE = "bible"
    UNKNOWN = "unknown"

    def __str__(self) -> str:
        return self.value


class DataLayer(str, Enum):
    """Pipeline data layers (lakehouse architecture)."""

    BRONZE = "bronze"  # Raw ingestion
    SILVER = "silver"  # Cleaned + normalized
    DIAMOND = "diamond"  # Human-validated + synthetic
    GOLD = "gold"  # Training-ready

    def __str__(self) -> str:
        return self.value

    @property
    def priority(self) -> int:
        """
        Priority for deduplication (higher = keep this record).

        Diamond layer has highest priority, Bronze lowest.
        """
        priorities = {
            DataLayer.BRONZE: 0,
            DataLayer.SILVER: 1,
            DataLayer.DIAMOND: 2,
            DataLayer.GOLD: 3,
        }
        return priorities[self]


class NahuatlDialect(str, Enum):
    """
    Náhuatl dialect variants.

    See: https://www.inali.gob.mx/pdf/catalogo_lenguas_indigenas.pdf
    """

    CLASSICAL = "classical"  # Náhuatl clásico (siglos XVI-XVII)
    CENTRAL = "central"  # Náhuatl del centro
    HUASTECA = "huasteca"  # Náhuatl de la Huasteca
    GUERRERO = "guerrero"  # Náhuatl de Guerrero
    PUEBLA = "puebla"  # Náhuatl de Puebla
    TLAXCALA = "tlaxcala"  # Náhuatl de Tlaxcala
    MORELOS = "morelos"  # Náhuatl de Morelos
    UNKNOWN = "unknown"  # No identificado

    def __str__(self) -> str:
        return self.value


class MayaVariant(str, Enum):
    """
    Maya language variants.

    See: https://www.inali.gob.mx/pdf/catalogo_lenguas_indigenas.pdf
    """

    YUCATEC = "yucatec"  # Maya yucateco
    TZOTZIL = "tzotzil"  # Tsotsil
    TZELTAL = "tzeltal"  # Tseltal
    CHOL = "chol"  # Ch'ol
    TOJOLABAL = "tojolabal"  # Tojolabal
    MAME = "mame"  # Mam
    QEQCHI = "qeqchi"  # Q'eqchi'
    UNKNOWN = "unknown"  # No identificado

    def __str__(self) -> str:
        return self.value
