"""
CLI Interface for Pipeline

Command-line interface for running the unified pipeline.

Usage:
    python -m src.pipeline.cli --help
    python -m src.pipeline.cli run --seed 42
    python -m src.pipeline.cli validate
    python -m src.pipeline.cli stats
"""

import argparse
import sys
from pathlib import Path
from typing import Optional

from src.pipeline.unify import UnifiedPipeline
from src.utils.config import get_settings
from src.utils.logger import configure_root_logger, get_logger

logger = get_logger(__name__)


def setup_logging(verbose: bool = False, quiet: bool = False):
    """
    Configure logging based on verbosity flags.

    Args:
        verbose: Enable debug logging
        quiet: Suppress all output except errors
    """
    if quiet:
        level = "ERROR"
    elif verbose:
        level = "DEBUG"
    else:
        level = "INFO"

    configure_root_logger(level=level)


def cmd_run(args):
    """Run the pipeline."""
    setup_logging(args.verbose, args.quiet)

    logger.info("Starting pipeline run")

    # Override settings if provided
    kwargs = {}
    if args.silver_dir:
        kwargs["silver_dir"] = Path(args.silver_dir)
    if args.diamond_dir:
        kwargs["diamond_dir"] = Path(args.diamond_dir)
    if args.gold_dir:
        kwargs["gold_dir"] = Path(args.gold_dir)
    if args.seed is not None:
        kwargs["seed"] = args.seed

    # Create pipeline
    pipeline = UnifiedPipeline(**kwargs)

    # Run pipeline
    try:
        stats = pipeline.run()

        # Print results
        print("\n" + "=" * 70)
        print("  Pipeline Complete")
        print("=" * 70)
        print(f"\nTotal unique records:  {stats['total_records']}")
        print(f"  Train:               {stats['train_count']:>6} ({stats['train_count']/stats['total_records']*100:.1f}%)")
        print(f"  Validation:          {stats['val_count']:>6} ({stats['val_count']/stats['total_records']*100:.1f}%)")
        print(f"  Test:                {stats['test_count']:>6} ({stats['test_count']/stats['total_records']*100:.1f}%)")
        print(f"\nLanguage distribution:")
        print(f"  Nahuatl:             {stats['nahuatl_count']:>6}")
        print(f"  Maya:                {stats['maya_count']:>6}")
        print("\n" + "=" * 70)

        return 0
    except Exception as e:
        logger.error(f"Pipeline failed: {e}", exc_info=True)
        print(f"\n[ERROR] Pipeline failed: {e}", file=sys.stderr)
        return 1


def cmd_validate(args):
    """Validate pipeline configuration."""
    setup_logging(args.verbose, args.quiet)

    logger.info("Validating pipeline configuration")

    try:
        # Check settings
        settings = get_settings()
        print("\n" + "=" * 70)
        print("  Configuration Validation")
        print("=" * 70)

        # Check ratios sum to 1
        ratio_sum = settings.train_ratio + settings.val_ratio + settings.test_ratio
        ratio_ok = abs(ratio_sum - 1.0) < 0.001

        print(f"\n[{'PASS' if ratio_ok else 'FAIL'}] Ratio validation: {ratio_sum:.3f} (expected 1.000)")

        # Check directories exist
        dirs_to_check = [
            ("Silver", settings.silver_dir),
            ("Diamond", settings.diamond_dir),
            ("Gold", settings.gold_dir),
        ]

        all_dirs_ok = True
        for name, directory in dirs_to_check:
            exists = directory.exists()
            all_dirs_ok = all_dirs_ok and exists
            print(f"[{'PASS' if exists else 'FAIL'}] {name} directory: {directory}")

        # Check database
        db_ok = settings.metadata_db_path.parent.exists()
        print(f"[{'PASS' if db_ok else 'FAIL'}] Database directory: {settings.metadata_db_path.parent}")

        # Overall result
        print("\n" + "=" * 70)
        if ratio_ok and all_dirs_ok and db_ok:
            print("  [SUCCESS] All validations passed")
            print("=" * 70)
            return 0
        else:
            print("  [FAILED] Some validations failed")
            print("=" * 70)
            return 1

    except Exception as e:
        logger.error(f"Validation failed: {e}", exc_info=True)
        print(f"\n[ERROR] Validation failed: {e}", file=sys.stderr)
        return 1


def cmd_stats(args):
    """Show statistics about data layers."""
    setup_logging(args.verbose, args.quiet)

    logger.info("Collecting layer statistics")

    try:
        settings = get_settings()

        print("\n" + "=" * 70)
        print("  Data Layer Statistics")
        print("=" * 70)

        # Count files in each layer
        layers = [
            ("Bronze", settings.bronze_dir),
            ("Silver", settings.silver_dir),
            ("Diamond", settings.diamond_dir),
            ("Gold", settings.gold_dir),
        ]

        for name, directory in layers:
            if directory.exists():
                jsonl_files = list(directory.glob("*.jsonl"))
                json_files = list(directory.glob("*.json"))
                total_files = len(jsonl_files) + len(json_files)

                print(f"\n{name} Layer ({directory}):")
                print(f"  JSONL files: {len(jsonl_files)}")
                print(f"  JSON files:  {len(json_files)}")
                print(f"  Total:       {total_files}")

                # Count records in JSONL files
                total_records = 0
                for jsonl_file in jsonl_files:
                    try:
                        with open(jsonl_file, 'r', encoding='utf-8') as f:
                            total_records += sum(1 for _ in f)
                    except Exception:
                        pass

                if total_records > 0:
                    print(f"  Records:     {total_records:,}")
            else:
                print(f"\n{name} Layer ({directory}):")
                print(f"  [NOT FOUND]")

        print("\n" + "=" * 70)
        return 0

    except Exception as e:
        logger.error(f"Stats collection failed: {e}", exc_info=True)
        print(f"\n[ERROR] Stats failed: {e}", file=sys.stderr)
        return 1


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="CORC-NAH Pipeline - ETL for Náhuatl/Maya data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run pipeline with default settings
  python -m src.pipeline.cli run

  # Run with custom seed
  python -m src.pipeline.cli run --seed 123

  # Run with custom directories
  python -m src.pipeline.cli run --silver-dir data/silver --gold-dir output/gold

  # Validate configuration
  python -m src.pipeline.cli validate

  # Show layer statistics
  python -m src.pipeline.cli stats

  # Run with verbose logging
  python -m src.pipeline.cli run --verbose

  # Run quietly (errors only)
  python -m src.pipeline.cli run --quiet
        """,
    )

    # Global flags
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose (debug) logging",
    )
    parser.add_argument(
        "-q", "--quiet",
        action="store_true",
        help="Suppress output (errors only)",
    )

    # Subcommands
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Run command
    run_parser = subparsers.add_parser(
        "run",
        help="Run the pipeline",
        description="Execute the full ETL pipeline",
    )
    run_parser.add_argument(
        "--silver-dir",
        type=str,
        help="Path to Silver layer directory",
    )
    run_parser.add_argument(
        "--diamond-dir",
        type=str,
        help="Path to Diamond layer directory",
    )
    run_parser.add_argument(
        "--gold-dir",
        type=str,
        help="Path to Gold layer directory",
    )
    run_parser.add_argument(
        "--seed",
        type=int,
        help="Random seed for reproducible splits",
    )
    run_parser.set_defaults(func=cmd_run)

    # Validate command
    validate_parser = subparsers.add_parser(
        "validate",
        help="Validate configuration",
        description="Check that configuration and directories are valid",
    )
    validate_parser.set_defaults(func=cmd_validate)

    # Stats command
    stats_parser = subparsers.add_parser(
        "stats",
        help="Show layer statistics",
        description="Display file counts and record counts for each layer",
    )
    stats_parser.set_defaults(func=cmd_stats)

    # Parse args
    args = parser.parse_args()

    # Show help if no command provided
    if not args.command:
        parser.print_help()
        return 1

    # Run command
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
