"""
Tests for CLI interface.

Run with:
    pytest tests/unit/test_cli.py -v
"""

import json
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

from src.pipeline.cli import cmd_run, cmd_stats, cmd_validate


class SimpleArgs:
    """Simple args namespace for testing."""

    def __init__(self, **kwargs):
        self.verbose = kwargs.get("verbose", False)
        self.quiet = kwargs.get("quiet", True)  # Default quiet for tests
        self.silver_dir = kwargs.get("silver_dir", None)
        self.diamond_dir = kwargs.get("diamond_dir", None)
        self.gold_dir = kwargs.get("gold_dir", None)
        self.seed = kwargs.get("seed", None)


class TestCmdValidate:
    """Tests for validate command."""

    def test_validate_returns_int(self):
        """Validate command should return an integer."""
        args = SimpleArgs()
        result = cmd_validate(args)
        assert isinstance(result, int)

    def test_validate_checks_ratios(self, capsys):
        """Should check that train/val/test ratios sum to 1."""
        args = SimpleArgs()
        cmd_validate(args)
        captured = capsys.readouterr()
        assert "Ratio validation" in captured.out


class TestCmdStats:
    """Tests for stats command."""

    def test_stats_returns_int(self):
        """Stats command should return an integer."""
        args = SimpleArgs()
        result = cmd_stats(args)
        assert isinstance(result, int)

    def test_stats_shows_layers(self, capsys):
        """Should show layer information."""
        args = SimpleArgs()
        cmd_stats(args)
        captured = capsys.readouterr()
        assert "Bronze" in captured.out
        assert "Silver" in captured.out
        assert "Diamond" in captured.out
        assert "Gold" in captured.out


class TestCmdRun:
    """Tests for run command."""

    def test_run_with_empty_dirs(self, tmp_path):
        """Pipeline should fail gracefully with empty directories."""
        args = SimpleArgs(
            silver_dir=str(tmp_path / "silver"),
            diamond_dir=str(tmp_path / "diamond"),
            gold_dir=str(tmp_path / "gold"),
        )
        # Create directories
        (tmp_path / "silver").mkdir()
        (tmp_path / "diamond").mkdir()
        (tmp_path / "gold").mkdir()

        # Should return error code (no records)
        result = cmd_run(args)
        assert result == 1

    def test_run_with_sample_data(self, tmp_path):
        """Pipeline should succeed with valid data."""
        # Create sample data
        silver_dir = tmp_path / "silver"
        silver_dir.mkdir()

        records = [
            {"es": "Hola", "nah": "Piyali"},
            {"es": "Gracias", "nah": "Tlazohcamati"},
            {"es": "Buenos dias", "nah": "Cualli tonalli"},
        ]

        with open(silver_dir / "test.jsonl", 'w', encoding='utf-8') as f:
            for r in records:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

        (tmp_path / "diamond").mkdir()
        gold_dir = tmp_path / "gold"
        gold_dir.mkdir()

        args = SimpleArgs(
            silver_dir=str(silver_dir),
            diamond_dir=str(tmp_path / "diamond"),
            gold_dir=str(gold_dir),
            seed=42,
        )

        result = cmd_run(args)
        assert result == 0

        # Verify output files created
        assert (gold_dir / "train_v1.jsonl").exists()
        assert (gold_dir / "validation_v1.jsonl").exists()
        assert (gold_dir / "test_v1.jsonl").exists()


class TestExceptions:
    """Tests for custom exceptions."""

    def test_pipeline_error_str(self):
        """PipelineError should format with context."""
        from src.exceptions import PipelineError

        err = PipelineError("Failed", stage="load", count=5)
        msg = str(err)
        assert "Failed" in msg
        assert "stage=load" in msg
        assert "count=5" in msg

    def test_pipeline_error_to_dict(self):
        """to_dict should return structured data."""
        from src.exceptions import DataLoadError

        err = DataLoadError("File missing", path="data.jsonl")
        d = err.to_dict()
        assert d["error_type"] == "DataLoadError"
        assert d["message"] == "File missing"
        assert d["context"]["path"] == "data.jsonl"

    def test_exception_hierarchy(self):
        """All exceptions should inherit from CorcNahException."""
        from src.exceptions import (
            CorcNahException,
            ConfigurationError,
            DatabaseError,
            DataLoadError,
            DataTransformError,
            DataValidationError,
            MetricsError,
            PipelineError,
        )

        for ExcClass in [
            ConfigurationError,
            DatabaseError,
            DataLoadError,
            DataTransformError,
            DataValidationError,
            MetricsError,
            PipelineError,
        ]:
            err = ExcClass("test error")
            assert isinstance(err, CorcNahException)
            assert isinstance(err, Exception)


class TestPipelineV2:
    """Tests for the production-ready pipeline v2."""

    def test_pipeline_v2_import(self):
        """UnifiedPipeline v2 should be importable."""
        from src.pipeline.unify_v2 import UnifiedPipeline

        assert UnifiedPipeline is not None

    def test_pipeline_v2_initializes(self, tmp_path):
        """Pipeline v2 should initialize with custom config."""
        from src.pipeline.unify_v2 import UnifiedPipeline

        pipeline = UnifiedPipeline(
            silver_dir=tmp_path / "silver",
            diamond_dir=tmp_path / "diamond",
            gold_dir=tmp_path / "gold",
            seed=123,
            show_progress=False,
            batch_size=500,
            save_metadata=False,
        )

        assert pipeline.seed == 123
        assert pipeline.batch_size == 500
        assert pipeline.show_progress is False
        assert pipeline.save_metadata is False

    def test_pipeline_v2_with_data(self, tmp_path):
        """Pipeline v2 should process data correctly."""
        from src.pipeline.unify_v2 import UnifiedPipeline

        # Create sample data
        silver_dir = tmp_path / "silver"
        silver_dir.mkdir()

        records = [
            {"es": "Hola", "nah": "Piyali"},
            {"es": "Gracias", "nah": "Tlazohcamati"},
            {"es": "Buenos dias", "nah": "Cualli tonalli"},
            {"es": "Adios", "nah": "Tlahual"},
            {"es": "Buenas noches", "nah": "Cualli yohualli"},
        ]

        with open(silver_dir / "test.jsonl", 'w', encoding='utf-8') as f:
            for r in records:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

        (tmp_path / "diamond").mkdir()
        gold_dir = tmp_path / "gold"
        gold_dir.mkdir()

        pipeline = UnifiedPipeline(
            silver_dir=silver_dir,
            diamond_dir=tmp_path / "diamond",
            gold_dir=gold_dir,
            seed=42,
            show_progress=False,
            save_metadata=False,
        )

        stats = pipeline.run()

        assert stats["total_records"] == 5
        assert stats["train_count"] + stats["val_count"] + stats["test_count"] == 5
        assert "duration_seconds" in stats
        assert stats["errors_count"] == 0

    def test_pipeline_v2_empty_data_raises(self, tmp_path):
        """Pipeline v2 should raise PipelineError when no data found."""
        from src.exceptions import PipelineError
        from src.pipeline.unify_v2 import UnifiedPipeline

        (tmp_path / "silver").mkdir()
        (tmp_path / "diamond").mkdir()
        (tmp_path / "gold").mkdir()

        pipeline = UnifiedPipeline(
            silver_dir=tmp_path / "silver",
            diamond_dir=tmp_path / "diamond",
            gold_dir=tmp_path / "gold",
            show_progress=False,
            save_metadata=False,
        )

        with pytest.raises(PipelineError, match="Pipeline execution failed"):
            pipeline.run()

    def test_pipeline_v2_deduplication(self, tmp_path):
        """Pipeline v2 should deduplicate records."""
        from src.pipeline.unify_v2 import UnifiedPipeline

        silver_dir = tmp_path / "silver"
        silver_dir.mkdir()

        # 2 unique + 1 duplicate
        records = [
            {"es": "Hola", "nah": "Piyali"},
            {"es": "Gracias", "nah": "Tlazohcamati"},
            {"es": "Hola", "nah": "Piyali"},  # duplicate
        ]

        with open(silver_dir / "test.jsonl", 'w', encoding='utf-8') as f:
            for r in records:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

        (tmp_path / "diamond").mkdir()

        pipeline = UnifiedPipeline(
            silver_dir=silver_dir,
            diamond_dir=tmp_path / "diamond",
            gold_dir=tmp_path / "gold",
            show_progress=False,
            save_metadata=False,
        )

        stats = pipeline.run()

        assert stats["total_records"] == 2
        assert stats["duplicates_removed"] == 1

    def test_pipeline_v2_returns_run_id(self, tmp_path):
        """Pipeline v2 stats should include run_id."""
        from src.pipeline.unify_v2 import UnifiedPipeline

        silver_dir = tmp_path / "silver"
        silver_dir.mkdir()

        with open(silver_dir / "test.jsonl", 'w', encoding='utf-8') as f:
            f.write(json.dumps({"es": "Hola", "nah": "Piyali"}, ensure_ascii=False) + "\n")

        (tmp_path / "diamond").mkdir()

        pipeline = UnifiedPipeline(
            silver_dir=silver_dir,
            diamond_dir=tmp_path / "diamond",
            gold_dir=tmp_path / "gold",
            show_progress=False,
            save_metadata=False,
        )

        stats = pipeline.run()
        assert "run_id" in stats
