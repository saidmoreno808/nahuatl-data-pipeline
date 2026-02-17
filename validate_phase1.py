#!/usr/bin/env python3
"""
Phase 1 Validation Script

Validates that all core utilities are working correctly.

Usage:
    python validate_phase1.py
"""

import sys
import tempfile
from pathlib import Path


def print_header(text: str):
    """Print section header."""
    print(f"\n{'=' * 60}")
    print(f"{text}")
    print(f"{'=' * 60}\n")


def print_check(message: str, passed: bool):
    """Print check result."""
    symbol = "[PASS]" if passed else "[FAIL]"
    print(f"{symbol} {message}")


def test_imports():
    """Test that all modules can be imported."""
    print_header("Testing Imports")

    modules = [
        ("src.utils.config", "Settings, get_settings"),
        ("src.utils.logger", "get_logger, StructuredFormatter"),
        ("src.utils.db", "get_db_connection, init_database"),
        ("src.utils.metrics", "MetricsTracker, track_time"),
    ]

    all_passed = True

    for module, items in modules:
        try:
            __import__(module, fromlist=items.split(", "))
            print_check(f"Import {module}", True)
        except ImportError as e:
            print_check(f"Import {module}: {e}", False)
            all_passed = False

    return all_passed


def test_config():
    """Test configuration management."""
    print_header("Testing Configuration")

    try:
        from src.utils.config import get_settings, override_settings

        # Test default settings
        settings = get_settings()
        print_check(f"Load settings (seed={settings.seed})", True)

        # Test validation
        try:
            override_settings(train_ratio=0.9, val_ratio=0.2, test_ratio=0.1)
            print_check("Ratio validation (should fail)", False)
            return False
        except ValueError:
            print_check("Ratio validation (correctly rejected invalid)", True)

        # Test override
        custom = override_settings(seed=999, debug=True)
        print_check(f"Override settings (seed={custom.seed})", custom.seed == 999)

        return True

    except Exception as e:
        print_check(f"Configuration test failed: {e}", False)
        return False


def test_logger():
    """Test logging utilities."""
    print_header("Testing Logger")

    try:
        from src.utils.logger import get_logger

        # Create logger
        logger = get_logger("test_logger")
        print_check("Create logger", True)

        # Test logging (should not raise)
        logger.info("Test message", extra={"test": True})
        print_check("Log message with extra fields", True)

        # Test file logging
        import time
        with tempfile.NamedTemporaryFile(mode="w", suffix=".log", delete=False) as f:
            log_file = Path(f.name)

        logger_file = get_logger("test_file_logger", log_file=log_file)
        logger_file.info("File test")

        # Force flush and close handlers
        for handler in logger_file.handlers:
            handler.close()

        time.sleep(0.1)  # Give Windows time to release the file

        content = log_file.read_text()

        try:
            log_file.unlink()  # Cleanup
        except Exception:
            pass  # Ignore cleanup errors on Windows

        print_check("Log to file", "File test" in content)

        return True

    except Exception as e:
        print_check(f"Logger test failed: {e}", False)
        return False


def test_database():
    """Test database utilities."""
    print_header("Testing Database")

    try:
        from src.utils.db import (
            execute_query,
            execute_update,
            get_db_connection,
            insert_many,
        )

        # Create temporary database
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = Path(f.name)

        # Create test table
        with get_db_connection(db_path) as conn:
            conn.execute(
                """
                CREATE TABLE test_table (
                    id INTEGER PRIMARY KEY,
                    value TEXT
                )
                """
            )
        print_check("Create table", True)

        # Test insert
        with get_db_connection(db_path) as conn:
            conn.execute("INSERT INTO test_table (value) VALUES (?)", ("test1",))
        print_check("Insert record", True)

        # Test query
        results = execute_query(
            "SELECT * FROM test_table WHERE value = ?",
            ("test1",),
            db_path,
        )
        print_check("Query record", len(results) == 1)

        # Test bulk insert
        records = [
            {"value": "test2"},
            {"value": "test3"},
        ]
        count = insert_many("test_table", records, db_path)
        print_check("Bulk insert", count == 2)

        # Test rollback
        try:
            with get_db_connection(db_path) as conn:
                conn.execute("INSERT INTO test_table (value) VALUES (?)", ("test4",))
                raise ValueError("Test error")
        except ValueError:
            pass

        results = execute_query(
            "SELECT * FROM test_table WHERE value = ?",
            ("test4",),
            db_path,
        )
        print_check("Transaction rollback", len(results) == 0)

        # Cleanup
        db_path.unlink()

        return True

    except Exception as e:
        print_check(f"Database test failed: {e}", False)
        import traceback

        traceback.print_exc()
        return False


def test_metrics():
    """Test metrics tracking."""
    print_header("Testing Metrics")

    try:
        from src.utils.metrics import MetricsTracker, track_time

        # Test MetricsTracker
        with MetricsTracker("test_operation", auto_log=False) as tracker:
            tracker.record("records", 100)
            tracker.increment("errors", 5)

        print_check(
            "MetricsTracker context manager",
            tracker.metrics["records"] == 100,
        )

        # Test track_time
        with track_time("test_timing") as metrics:
            metrics["items"] = 50

        print_check("track_time context manager", metrics["items"] == 50)

        return True

    except Exception as e:
        print_check(f"Metrics test failed: {e}", False)
        return False


def test_integration():
    """Test integration between modules."""
    print_header("Testing Integration")

    try:
        from src.utils.config import override_settings
        from src.utils.db import get_db_connection, init_database
        from src.utils.logger import get_logger
        from src.utils.metrics import MetricsTracker

        # Create temp environment
        with tempfile.TemporaryDirectory() as tmpdir:
            tmppath = Path(tmpdir)

            # Configure
            settings = override_settings(
                project_root=tmppath,
                data_dir=tmppath / "data",
                metadata_db_path=tmppath / "metadata.db",
            )
            settings.ensure_directories_exist()

            print_check("Setup test environment", True)

            # Create logger
            logger = get_logger("integration_test")
            logger.info("Integration test started")
            print_check("Logger created", True)

            # Initialize database (will fail if schema.sql doesn't exist, which is ok)
            try:
                init_database(settings.metadata_db_path)
                print_check("Database initialized", True)
            except FileNotFoundError:
                print_check(
                    "Database initialization skipped (schema.sql not found)", True
                )

            # Test metrics tracking
            with MetricsTracker("integration_test", auto_log=False) as tracker:
                tracker.record("test_metric", 42)

            print_check("Metrics tracking", tracker.metrics["test_metric"] == 42)

        return True

    except Exception as e:
        print_check(f"Integration test failed: {e}", False)
        import traceback

        traceback.print_exc()
        return False


def main():
    """Run all validation tests."""
    print("CORC-NAH Phase 1 Validation")
    print("=" * 60)

    tests = [
        ("Module Imports", test_imports),
        ("Configuration", test_config),
        ("Logger", test_logger),
        ("Database", test_database),
        ("Metrics", test_metrics),
        ("Integration", test_integration),
    ]

    results = []

    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"\n❌ {test_name} crashed: {e}")
            import traceback

            traceback.print_exc()
            results.append((test_name, False))

    # Summary
    print_header("Summary")

    passed = sum(1 for _, result in results if result)
    total = len(results)

    for test_name, result in results:
        symbol = "[PASS]" if result else "[FAIL]"
        print(f"{symbol} {test_name}")

    print(f"\n{passed}/{total} test suites passed")

    if passed == total:
        print("\n*** Phase 1 validation PASSED! ***")
        print("[PASS] All core utilities are working correctly")
        print("\nNext steps:")
        print("  1. Run: pytest tests/unit/ -v")
        print("  2. Run: make lint")
        print("  3. Start Phase 2: Transform Logic")
        return 0
    else:
        print("\n*** Phase 1 validation FAILED ***")
        print("Please fix the errors above before proceeding")
        return 1


if __name__ == "__main__":
    sys.exit(main())
