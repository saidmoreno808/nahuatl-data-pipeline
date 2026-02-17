"""
Configuration Management

Uses Pydantic for type-safe settings with environment variable support.
Follows 12-factor app principles for configuration.

Example:
    >>> from src.utils.config import get_settings
    >>> settings = get_settings()
    >>> print(settings.data_dir)
    PosixPath('data')
"""

import os
from functools import lru_cache
from pathlib import Path
from typing import Optional

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Application settings with environment variable support.

    Environment variables are prefixed with CORC_NAH_
    Example: CORC_NAH_DATA_DIR=/custom/path
    """

    # Directories
    project_root: Path = Field(
        default_factory=lambda: Path(__file__).parent.parent.parent,
        description="Project root directory",
    )
    data_dir: Path = Field(
        default=Path("data"),
        description="Root data directory (lakehouse)",
    )
    bronze_dir: Path = Field(
        default=Path("data/bronze"),
        description="Bronze layer (raw ingestion)",
    )
    silver_dir: Path = Field(
        default=Path("data/silver"),
        description="Silver layer (cleaned + normalized)",
    )
    diamond_dir: Path = Field(
        default=Path("data/diamond"),
        description="Diamond layer (human-validated + synthetic)",
    )
    gold_dir: Path = Field(
        default=Path("data/gold"),
        description="Gold layer (training-ready splits)",
    )
    logs_dir: Path = Field(
        default=Path("logs"),
        description="Logs directory",
    )
    benchmark_dir: Path = Field(
        default=Path("benchmark"),
        description="Golden dataset directory",
    )

    # Database
    metadata_db_path: Path = Field(
        default=Path("logs/metadata.db"),
        description="SQLite metadata database path",
    )

    # Pipeline settings
    seed: int = Field(
        default=42,
        description="Random seed for reproducibility",
    )
    train_ratio: float = Field(
        default=0.9,
        ge=0.0,
        le=1.0,
        description="Training set ratio",
    )
    val_ratio: float = Field(
        default=0.05,
        ge=0.0,
        le=1.0,
        description="Validation set ratio",
    )
    test_ratio: float = Field(
        default=0.05,
        ge=0.0,
        le=1.0,
        description="Test set ratio",
    )

    # Data processing
    max_text_length: int = Field(
        default=1000,
        gt=0,
        description="Maximum text length (characters)",
    )
    min_text_length: int = Field(
        default=3,
        gt=0,
        description="Minimum text length (characters)",
    )
    max_duplicate_rate: float = Field(
        default=0.05,
        ge=0.0,
        le=1.0,
        description="Maximum acceptable duplicate rate",
    )
    max_null_rate: float = Field(
        default=0.10,
        ge=0.0,
        le=1.0,
        description="Maximum acceptable null rate",
    )

    # Unicode normalization
    unicode_normalization: str = Field(
        default="NFC",
        description="Unicode normalization form (NFC preserves macrons)",
    )

    # Logging
    log_level: str = Field(
        default="INFO",
        description="Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)",
    )
    log_format: str = Field(
        default="json",
        description="Log format (json or text)",
    )

    # Performance
    batch_size: int = Field(
        default=1000,
        gt=0,
        description="Batch size for processing",
    )
    n_workers: int = Field(
        default=4,
        gt=0,
        description="Number of worker processes",
    )

    # Environment
    environment: str = Field(
        default="development",
        description="Environment (development, staging, production)",
    )
    debug: bool = Field(
        default=False,
        description="Enable debug mode",
    )

    model_config = SettingsConfigDict(
        env_prefix="CORC_NAH_",
        case_sensitive=False,
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    @field_validator("test_ratio")
    @classmethod
    def validate_ratios_sum_to_one(cls, v, info):
        """Ensure train/val/test ratios sum to 1.0."""
        values = info.data
        if "train_ratio" in values and "val_ratio" in values:
            total = values["train_ratio"] + values["val_ratio"] + v
            if not 0.99 <= total <= 1.01:
                raise ValueError(
                    f"Ratios must sum to 1.0, got {total:.3f}"
                )
        return v

    @field_validator("unicode_normalization")
    @classmethod
    def validate_unicode_normalization(cls, v):
        """Ensure valid Unicode normalization form."""
        valid_forms = ["NFC", "NFD", "NFKC", "NFKD"]
        if v not in valid_forms:
            raise ValueError(
                f"Invalid normalization form '{v}'. Must be one of {valid_forms}"
            )
        return v

    @field_validator("log_level")
    @classmethod
    def validate_log_level(cls, v):
        """Ensure valid log level."""
        valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        v_upper = v.upper()
        if v_upper not in valid_levels:
            raise ValueError(
                f"Invalid log level '{v}'. Must be one of {valid_levels}"
            )
        return v_upper

    def ensure_directories_exist(self) -> None:
        """Create all required directories if they don't exist."""
        directories = [
            self.data_dir,
            self.bronze_dir,
            self.silver_dir,
            self.diamond_dir,
            self.gold_dir,
            self.logs_dir,
            self.benchmark_dir,
        ]

        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)

    def get_absolute_path(self, path: Path) -> Path:
        """Convert relative path to absolute based on project root."""
        if path.is_absolute():
            return path
        return (self.project_root / path).resolve()


@lru_cache()
def get_settings() -> Settings:
    """
    Get cached settings instance.

    Uses lru_cache to ensure settings are only loaded once.
    Override with environment variables.

    Returns:
        Settings: Application settings instance

    Example:
        >>> settings = get_settings()
        >>> print(settings.seed)
        42
    """
    settings = Settings()
    settings.ensure_directories_exist()
    return settings


def override_settings(**kwargs) -> Settings:
    """
    Create settings instance with overrides (for testing).

    Args:
        **kwargs: Setting overrides

    Returns:
        Settings: New settings instance

    Example:
        >>> settings = override_settings(seed=123, debug=True)
        >>> assert settings.seed == 123
    """
    return Settings(**kwargs)
