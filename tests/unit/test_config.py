"""
Unit tests for configuration management.

Run with:
    pytest tests/unit/test_config.py -v
"""

import os
from pathlib import Path

import pytest

from src.utils.config import Settings, get_settings, override_settings


class TestSettings:
    """Test Settings model."""

    def test_default_settings(self):
        """Test default settings values."""
        settings = Settings()

        assert settings.seed == 42
        assert settings.train_ratio == 0.9
        assert settings.val_ratio == 0.05
        assert settings.test_ratio == 0.05
        assert settings.unicode_normalization == "NFC"
        assert settings.log_level == "INFO"

    def test_ratio_validation(self):
        """Test that train/val/test ratios must sum to 1.0."""
        # Valid ratios
        settings = Settings(train_ratio=0.8, val_ratio=0.1, test_ratio=0.1)
        assert settings.train_ratio == 0.8

        # Invalid ratios (sum > 1.0)
        with pytest.raises(ValueError, match="must sum to 1.0"):
            Settings(train_ratio=0.9, val_ratio=0.2, test_ratio=0.1)

    def test_unicode_normalization_validation(self):
        """Test Unicode normalization form validation."""
        # Valid forms
        for form in ["NFC", "NFD", "NFKC", "NFKD"]:
            settings = Settings(unicode_normalization=form)
            assert settings.unicode_normalization == form

        # Invalid form
        with pytest.raises(ValueError, match="Invalid normalization form"):
            Settings(unicode_normalization="INVALID")

    def test_log_level_validation(self):
        """Test log level validation."""
        # Valid levels (case insensitive)
        for level in ["debug", "INFO", "Warning", "ERROR", "CRITICAL"]:
            settings = Settings(log_level=level)
            assert settings.log_level in [
                "DEBUG",
                "INFO",
                "WARNING",
                "ERROR",
                "CRITICAL",
            ]

        # Invalid level
        with pytest.raises(ValueError, match="Invalid log level"):
            Settings(log_level="INVALID")

    def test_environment_variables(self, monkeypatch):
        """Test settings loading from environment variables."""
        # Set environment variables
        monkeypatch.setenv("CORC_NAH_SEED", "123")
        monkeypatch.setenv("CORC_NAH_LOG_LEVEL", "DEBUG")
        monkeypatch.setenv("CORC_NAH_DEBUG", "true")

        # Clear cache and reload settings
        get_settings.cache_clear()
        settings = get_settings()

        assert settings.seed == 123
        assert settings.log_level == "DEBUG"
        assert settings.debug is True

    def test_ensure_directories_exist(self, tmp_path):
        """Test directory creation."""
        settings = Settings(
            project_root=tmp_path,
            data_dir=tmp_path / "data",
            bronze_dir=tmp_path / "data" / "bronze",
        )

        # Directories don't exist yet
        assert not (tmp_path / "data").exists()

        # Create directories
        settings.ensure_directories_exist()

        # Directories now exist
        assert (tmp_path / "data").exists()
        assert (tmp_path / "data" / "bronze").exists()

    def test_get_absolute_path(self):
        """Test absolute path conversion."""
        settings = Settings()

        # Relative path
        rel_path = Path("data/bronze")
        abs_path = settings.get_absolute_path(rel_path)
        assert abs_path.is_absolute()
        # Use Path comparison to avoid Windows/Linux separator issues
        assert abs_path.parts[-2] == "data"
        assert abs_path.parts[-1] == "bronze"

        # Already absolute path (use tmp_path for cross-platform compatibility)
        import tempfile

        abs_input = Path(tempfile.gettempdir()) / "test_path"
        abs_output = settings.get_absolute_path(abs_input)
        assert abs_output == abs_input


class TestOverrideSettings:
    """Test settings override function."""

    def test_override_settings(self):
        """Test creating settings with overrides."""
        settings = override_settings(
            seed=999,
            debug=True,
            log_level="DEBUG",
        )

        assert settings.seed == 999
        assert settings.debug is True
        assert settings.log_level == "DEBUG"

        # Default values should still be present
        assert settings.train_ratio == 0.9


class TestGetSettings:
    """Test singleton settings instance."""

    def test_get_settings_is_cached(self):
        """Test that get_settings() returns cached instance."""
        get_settings.cache_clear()

        settings1 = get_settings()
        settings2 = get_settings()

        # Should be the same instance
        assert settings1 is settings2

    def test_get_settings_creates_directories(self, tmp_path):
        """Test that ensure_directories_exist() creates directories."""
        data_dir = tmp_path / "data"
        bronze_dir = tmp_path / "data" / "bronze"

        settings = Settings(
            project_root=tmp_path,
            data_dir=data_dir,
            bronze_dir=bronze_dir,
        )

        # Directories don't exist yet
        assert not data_dir.exists()
        assert not bronze_dir.exists()

        # Create directories
        settings.ensure_directories_exist()

        # Directories now exist
        assert data_dir.exists()
        assert bronze_dir.exists()
