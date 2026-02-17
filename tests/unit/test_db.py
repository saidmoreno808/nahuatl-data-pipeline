"""
Unit tests for database utilities.

Run with:
    pytest tests/unit/test_db.py -v
"""

import sqlite3
from pathlib import Path

import pytest

from src.utils.config import override_settings
from src.utils.db import (
    execute_query,
    execute_update,
    get_db_connection,
    get_latest_run,
    init_database,
    insert_many,
)


@pytest.fixture
def test_db(tmp_path):
    """Create a temporary test database."""
    db_path = tmp_path / "test.db"

    # Create minimal schema
    with sqlite3.connect(db_path) as conn:
        conn.executescript(
            """
            CREATE TABLE pipeline_runs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id TEXT UNIQUE NOT NULL,
                pipeline_name TEXT NOT NULL,
                started_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                ended_at TIMESTAMP,
                status TEXT NOT NULL,
                records_input INTEGER,
                records_output INTEGER,
                records_filtered INTEGER,
                duration_seconds REAL,
                error_message TEXT,
                git_commit_hash TEXT,
                config_snapshot TEXT
            );

            CREATE TABLE quality_metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id TEXT NOT NULL,
                metric_name TEXT NOT NULL,
                metric_value REAL NOT NULL,
                metric_unit TEXT,
                dataset_split TEXT,
                measured_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (run_id) REFERENCES pipeline_runs(run_id)
            );
            """
        )

    return db_path


class TestGetDbConnection:
    """Test database connection context manager."""

    def test_connection_opens_and_closes(self, test_db):
        """Test that connection is properly managed."""
        with get_db_connection(test_db) as conn:
            assert conn is not None
            assert isinstance(conn, sqlite3.Connection)

            # Test query
            cursor = conn.execute("SELECT 1")
            result = cursor.fetchone()
            assert result[0] == 1

        # Connection should be closed after context
        with pytest.raises(sqlite3.ProgrammingError):
            conn.execute("SELECT 1")

    def test_connection_commits_on_success(self, test_db):
        """Test that changes are committed on successful completion."""
        with get_db_connection(test_db) as conn:
            conn.execute(
                """
                INSERT INTO pipeline_runs (run_id, pipeline_name, status)
                VALUES (?, ?, ?)
                """,
                ("test-1", "test", "success"),
            )

        # Verify data was committed
        with get_db_connection(test_db, read_only=True) as conn:
            cursor = conn.execute(
                "SELECT run_id FROM pipeline_runs WHERE run_id = ?",
                ("test-1",),
            )
            result = cursor.fetchone()
            assert result is not None
            assert result["run_id"] == "test-1"

    def test_connection_rolls_back_on_error(self, test_db):
        """Test that changes are rolled back on error."""
        try:
            with get_db_connection(test_db) as conn:
                conn.execute(
                    """
                    INSERT INTO pipeline_runs (run_id, pipeline_name, status)
                    VALUES (?, ?, ?)
                    """,
                    ("test-2", "test", "success"),
                )
                # Raise an error
                raise ValueError("Test error")
        except ValueError:
            pass

        # Verify data was not committed
        with get_db_connection(test_db, read_only=True) as conn:
            cursor = conn.execute(
                "SELECT run_id FROM pipeline_runs WHERE run_id = ?",
                ("test-2",),
            )
            result = cursor.fetchone()
            assert result is None

    def test_row_factory_enabled(self, test_db):
        """Test that row factory is enabled for dict-like access."""
        with get_db_connection(test_db) as conn:
            conn.execute(
                """
                INSERT INTO pipeline_runs (run_id, pipeline_name, status)
                VALUES (?, ?, ?)
                """,
                ("test-3", "test", "success"),
            )

        with get_db_connection(test_db, read_only=True) as conn:
            cursor = conn.execute(
                "SELECT run_id, pipeline_name FROM pipeline_runs WHERE run_id = ?",
                ("test-3",),
            )
            row = cursor.fetchone()

            # Should be able to access by column name
            assert row["run_id"] == "test-3"
            assert row["pipeline_name"] == "test"


class TestExecuteQuery:
    """Test query execution helpers."""

    def test_execute_query(self, test_db):
        """Test executing SELECT queries."""
        # Insert test data
        with get_db_connection(test_db) as conn:
            conn.execute(
                """
                INSERT INTO pipeline_runs (run_id, pipeline_name, status)
                VALUES (?, ?, ?)
                """,
                ("test-4", "test", "success"),
            )

        # Query
        results = execute_query(
            "SELECT * FROM pipeline_runs WHERE run_id = ?",
            ("test-4",),
            test_db,
        )

        assert len(results) == 1
        assert results[0]["run_id"] == "test-4"

    def test_execute_query_returns_empty_list(self, test_db):
        """Test query with no results."""
        results = execute_query(
            "SELECT * FROM pipeline_runs WHERE run_id = ?",
            ("nonexistent",),
            test_db,
        )

        assert results == []


class TestExecuteUpdate:
    """Test update execution helpers."""

    def test_execute_update(self, test_db):
        """Test executing UPDATE queries."""
        # Insert test data
        with get_db_connection(test_db) as conn:
            conn.execute(
                """
                INSERT INTO pipeline_runs (run_id, pipeline_name, status)
                VALUES (?, ?, ?)
                """,
                ("test-5", "test", "running"),
            )

        # Update
        rows_affected = execute_update(
            "UPDATE pipeline_runs SET status = ? WHERE run_id = ?",
            ("success", "test-5"),
            test_db,
        )

        assert rows_affected == 1

        # Verify update
        results = execute_query(
            "SELECT status FROM pipeline_runs WHERE run_id = ?",
            ("test-5",),
            test_db,
        )
        assert results[0]["status"] == "success"


class TestInsertMany:
    """Test bulk insert helper."""

    def test_insert_many(self, test_db):
        """Test bulk inserting records."""
        records = [
            {"run_id": "bulk-1", "pipeline_name": "test", "status": "success"},
            {"run_id": "bulk-2", "pipeline_name": "test", "status": "failed"},
            {"run_id": "bulk-3", "pipeline_name": "test", "status": "success"},
        ]

        count = insert_many("pipeline_runs", records, test_db)

        assert count == 3

        # Verify all inserted
        results = execute_query(
            "SELECT run_id FROM pipeline_runs WHERE run_id LIKE 'bulk-%' ORDER BY run_id",
            db_path=test_db,
        )

        assert len(results) == 3
        assert results[0]["run_id"] == "bulk-1"
        assert results[1]["run_id"] == "bulk-2"
        assert results[2]["run_id"] == "bulk-3"

    def test_insert_many_empty_list(self, test_db):
        """Test inserting empty list."""
        count = insert_many("pipeline_runs", [], test_db)
        assert count == 0


class TestGetLatestRun:
    """Test getting latest pipeline run."""

    def test_get_latest_run(self, test_db):
        """Test retrieving the most recent run."""
        # Insert test data with different timestamps
        with get_db_connection(test_db) as conn:
            conn.execute(
                """
                INSERT INTO pipeline_runs (
                    run_id, pipeline_name, status, started_at
                ) VALUES
                    ('old-run', 'test', 'success', '2024-01-01 10:00:00'),
                    ('new-run', 'test', 'success', '2024-01-02 10:00:00'),
                    ('newest-run', 'test', 'success', '2024-01-03 10:00:00')
                """
            )

        latest = get_latest_run(test_db)

        assert latest is not None
        assert latest["run_id"] == "newest-run"

    def test_get_latest_run_empty_table(self, test_db):
        """Test when no runs exist."""
        latest = get_latest_run(test_db)
        assert latest is None


@pytest.mark.slow
class TestInitDatabase:
    """Test database initialization."""

    def test_init_database_creates_schema(self, tmp_path, monkeypatch):
        """Test that init_database creates all tables."""
        # This test requires the actual schema.sql file
        # Skip if not in proper project structure
        schema_path = Path(__file__).parent.parent.parent / "sql" / "schema.sql"

        if not schema_path.exists():
            pytest.skip("schema.sql not found")

        db_path = tmp_path / "initialized.db"

        # Override settings
        settings = override_settings(
            project_root=Path(__file__).parent.parent.parent,
            metadata_db_path=db_path,
        )

        # Initialize
        init_database(db_path)

        # Verify tables exist
        with get_db_connection(db_path, read_only=True) as conn:
            cursor = conn.execute(
                """
                SELECT name FROM sqlite_master
                WHERE type='table'
                ORDER BY name
                """
            )
            tables = [row["name"] for row in cursor.fetchall()]

            # Check key tables exist
            assert "pipeline_runs" in tables
            assert "quality_metrics" in tables
            assert "data_lineage" in tables
