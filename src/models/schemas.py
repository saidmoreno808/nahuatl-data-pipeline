"""
Pydantic data models for CORC-NAH records.

Provides type-safe, validated data structures for pipeline processing.

Example:
    >>> record = Record(
    ...     es="Hola",
    ...     nah="Piyali",
    ...     source="manual",
    ...     layer="diamond"
    ... )
    >>> print(record.es)
    'Hola'
"""

from datetime import datetime
from typing import Any, Dict, Optional

from pydantic import BaseModel, Field, field_validator

from src.models.enums import DataLayer, DataSource, Language, MayaVariant, NahuatlDialect


class RecordMetadata(BaseModel):
    """
    Metadata for a data record.

    Tracks provenance, quality, and processing information.
    """

    source: DataSource = Field(
        default=DataSource.UNKNOWN,
        description="Data source type",
    )
    layer: DataLayer = Field(
        default=DataLayer.BRONZE,
        description="Pipeline layer",
    )
    origin_file: Optional[str] = Field(
        default=None,
        description="Original file path",
    )
    record_id: Optional[str] = Field(
        default=None,
        description="Unique record identifier",
    )
    created_at: datetime = Field(
        default_factory=datetime.utcnow,
        description="Creation timestamp",
    )
    category: Optional[str] = Field(
        default=None,
        description="Content category (e.g., 'bible', 'conversation')",
    )
    dialect: Optional[NahuatlDialect] = Field(
        default=None,
        description="Náhuatl dialect variant",
    )
    maya_variant: Optional[MayaVariant] = Field(
        default=None,
        description="Maya language variant",
    )
    quality_score: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Quality score (0.0 - 1.0)",
    )
    is_validated: bool = Field(
        default=False,
        description="Whether record has been manually validated",
    )

    class Config:
        """Pydantic configuration."""

        use_enum_values = True
        json_encoders = {
            datetime: lambda v: v.isoformat(),
        }


class Record(BaseModel):
    """
    A multilingual data record.

    Represents a translation pair or triplet (Spanish + Náhuatl + Maya).

    Example:
        >>> record = Record(
        ...     es="Buenos días",
        ...     nah="Cualli tonalli",
        ...     source="manual",
        ...     layer="diamond"
        ... )
        >>> record.has_nahuatl()
        True
    """

    # Language fields
    es: Optional[str] = Field(
        default=None,
        description="Spanish text",
    )
    nah: Optional[str] = Field(
        default=None,
        description="Náhuatl text",
    )
    myn: Optional[str] = Field(
        default=None,
        description="Maya text",
    )

    # Metadata (flat structure for compatibility with legacy format)
    source: DataSource = Field(
        default=DataSource.UNKNOWN,
        description="Data source type",
    )
    layer: DataLayer = Field(
        default=DataLayer.BRONZE,
        description="Pipeline layer",
    )
    origin_file: Optional[str] = Field(
        default=None,
        description="Original file path",
    )
    category: Optional[str] = Field(
        default=None,
        description="Content category",
    )

    # Extended metadata (optional)
    metadata: Optional[RecordMetadata] = Field(
        default=None,
        description="Extended metadata",
    )

    class Config:
        """Pydantic configuration."""

        use_enum_values = True

    @field_validator("es", "nah", "myn")
    @classmethod
    def strip_whitespace(cls, v: Optional[str]) -> Optional[str]:
        """Strip leading/trailing whitespace from text fields."""
        if v is not None:
            v = v.strip()
            return v if v else None
        return v

    @field_validator("es", "nah", "myn")
    @classmethod
    def reject_empty_strings(cls, v: Optional[str]) -> Optional[str]:
        """Convert empty strings to None."""
        if v == "":
            return None
        return v

    def has_spanish(self) -> bool:
        """Check if record has Spanish text."""
        return self.es is not None and len(self.es) > 0

    def has_nahuatl(self) -> bool:
        """Check if record has Náhuatl text."""
        return self.nah is not None and len(self.nah) > 0

    def has_maya(self) -> bool:
        """Check if record has Maya text."""
        return self.myn is not None and len(self.myn) > 0

    def has_translation_pair(self) -> bool:
        """Check if record has at least Spanish + one indigenous language."""
        return self.has_spanish() and (self.has_nahuatl() or self.has_maya())

    def get_dedup_key(self) -> str:
        """
        Generate deduplication key.

        Combines normalized Spanish + Náhuatl + Maya text for deduplication.

        Returns:
            str: Deduplication key
        """
        es_norm = (self.es or "").lower().strip()
        nah_norm = (self.nah or "").lower().strip()
        myn_norm = (self.myn or "").lower().strip()
        return f"{es_norm}|{nah_norm}|{myn_norm}"

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert to dictionary (compatible with legacy format).

        Returns:
            dict: Record as dictionary
        """
        return self.model_dump(
            exclude_none=True,
            exclude={"metadata"},
            by_alias=True,
        )

    def to_jsonl(self) -> str:
        """
        Convert to JSONL line.

        Returns:
            str: JSON string (without newline)
        """
        import json

        return json.dumps(self.to_dict(), ensure_ascii=False)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Record":
        """
        Create record from dictionary.

        Args:
            data: Dictionary with record data

        Returns:
            Record: Validated record instance
        """
        return cls(**data)

    @classmethod
    def from_legacy_format(cls, data: Dict[str, Any]) -> "Record":
        """
        Create record from legacy unify_datasets.py format.

        Handles variations like:
        - es_translation → es
        - nah_translation → nah
        - original_audio_text → nah/myn (based on detected_language)
        - prompt/chosen (DPO format) → es/nah

        Args:
            data: Legacy format dictionary

        Returns:
            Record: Normalized record
        """
        # Extract Spanish
        es = (
            data.get("es")
            or data.get("es_translation")
            or data.get("original_es")
            or data.get("prompt")
        )

        # Extract Náhuatl
        nah = data.get("nah") or data.get("nah_translation") or data.get("chosen")

        if (
            not nah
            and data.get("original_audio_text")
            and data.get("detected_language") == "nah"
        ):
            nah = data["original_audio_text"]

        # Extract Maya
        myn = data.get("myn") or data.get("myn_translation")

        if (
            not myn
            and data.get("original_audio_text")
            and data.get("detected_language") == "myn"
        ):
            myn = data["original_audio_text"]

        # Extract metadata
        source = data.get("source") or DataSource.UNKNOWN
        layer = data.get("layer") or DataLayer.BRONZE
        origin_file = data.get("origin_file") or data.get("source_file")
        category = data.get("category")

        return cls(
            es=es,
            nah=nah,
            myn=myn,
            source=source,
            layer=layer,
            origin_file=origin_file,
            category=category,
        )


# Example usage
if __name__ == "__main__":
    # Create a record
    record = Record(
        es="Buenos días",
        nah="Cualli tonalli",
        source=DataSource.MANUAL,
        layer=DataLayer.DIAMOND,
        origin_file="manual_translations.jsonl",
    )

    print("Record created:")
    print(f"  Spanish: {record.es}")
    print(f"  Náhuatl: {record.nah}")
    print(f"  Source: {record.source}")
    print(f"  Layer: {record.layer}")
    print(f"  Has translation pair: {record.has_translation_pair()}")
    print(f"  Dedup key: {record.get_dedup_key()}")

    # Convert to dict
    print("\nAs dictionary:")
    print(record.to_dict())

    # Convert to JSONL
    print("\nAs JSONL:")
    print(record.to_jsonl())

    # Test legacy format conversion
    print("\nFrom legacy format:")
    legacy_data = {
        "es_translation": "Hola",
        "nah_translation": "Piyali",
        "source_file": "legacy.json",
        "layer": "silver",
    }
    legacy_record = Record.from_legacy_format(legacy_data)
    print(f"  Spanish: {legacy_record.es}")
    print(f"  Náhuatl: {legacy_record.nah}")
