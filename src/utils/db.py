"""
Database Utilities

Provides SQLite context manager for metadata storage.
Handles connection lifecycle, transactions, and error recovery.

Example:
    >>> from src.utils.db import get_db_connection
    >>> with get_db_connection() as conn:
    ...     conn.execute("INSERT INTO pipeline_runs ...")
"""

import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Generator, Optional

from src.utils.config import get_settings
from src.utils.logger import get_logger

logger = get_logger(__name__)


@contextmanager
def get_db_connection(
    db_path: Optional[Path] = None,
    read_only: bool = False,
) -> Generator[sqlite3.Connection, None, None]:
    """
    Context manager for SQLite database connections.

    Handles automatic commit/rollback and connection cleanup.
    Thread-safe with connection per context.

    Args:
        db_path: Database file path (defaults to settings.metadata_db_path)
        read_only: Open in read-only mode

    Yields:
        sqlite3.Connection: Database connection with row factory

    Example:
        >>> with get_db_connection() as conn:
        ...     cursor = conn.execute("SELECT * FROM pipeline_runs")
        ...     rows = cursor.fetchall()

    Raises:
        sqlite3.Error: On database errors
    """
    settings = get_settings()
    db_path = db_path or settings.metadata_db_path

    # Ensure parent directory exists
    db_path.parent.mkdir(parents=True, exist_ok=True)

    # Connection URI for read-only mode
    if read_only:
        uri = f"file:{db_path}?mode=ro"
        conn = sqlite3.connect(uri, uri=True)
    else:
        conn = sqlite3.connect(db_path)

    # Enable row factory for dict-like access
    conn.row_factory = sqlite3.Row

    # Enable foreign keys
    conn.execute("PRAGMA foreign_keys = ON")

    # Enable WAL mode for better concurrency
    if not read_only:
        conn.execute("PRAGMA journal_mode = WAL")

    try:
        logger.debug(f"Database connection opened", extra={"db_path": str(db_path)})
        yield conn

        # Commit on successful completion
        if not read_only:
            conn.commit()
            logger.debug("Transaction committed")

    except Exception as e:
        # Rollback on error
        if not read_only:
            conn.rollback()
            logger.error(
                f"Transaction rolled back due to error",
                extra={"error": str(e)},
                exc_info=True,
            )
        raise

    finally:
        conn.close()
        logger.debug("Database connection closed")


def init_database(db_path: Optional[Path] = None) -> None:
    """
    Initialize database schema from SQL file.

    Args:
        db_path: Database file path (defaults to settings.metadata_db_path)

    Raises:
        FileNotFoundError: If schema.sql is not found
        sqlite3.Error: On database errors

    Example:
        >>> init_database()  # Creates tables from sql/schema.sql
    """
    settings = get_settings()
    db_path = db_path or settings.metadata_db_path
    schema_path = settings.project_root / "sql" / "schema.sql"

    if not schema_path.exists():
        raise FileNotFoundError(f"Schema file not found: {schema_path}")

    logger.info(f"Initializing database", extra={"db_path": str(db_path)})

    # Read schema
    with open(schema_path, "r", encoding="utf-8") as f:
        schema_sql = f.read()

    # Execute schema
    with get_db_connection(db_path) as conn:
        conn.executescript(schema_sql)

    logger.info("Database initialized successfully")


def execute_query(
    query: str,
    params: Optional[tuple] = None,
    db_path: Optional[Path] = None,
) -> list:
    """
    Execute a SELECT query and return results.

    Args:
        query: SQL query string
        params: Query parameters (for parameterized queries)
        db_path: Database file path

    Returns:
        list: Query results as list of Row objects

    Example:
        >>> results = execute_query(
        ...     "SELECT * FROM pipeline_runs WHERE status = ?",
        ...     ("success",)
        ... )
        >>> for row in results:
        ...     print(row["run_id"])
    """
    with get_db_connection(db_path, read_only=True) as conn:
        cursor = conn.execute(query, params or ())
        return cursor.fetchall()


def execute_update(
    query: str,
    params: Optional[tuple] = None,
    db_path: Optional[Path] = None,
) -> int:
    """
    Execute an INSERT/UPDATE/DELETE query.

    Args:
        query: SQL query string
        params: Query parameters
        db_path: Database file path

    Returns:
        int: Number of affected rows

    Example:
        >>> rows_affected = execute_update(
        ...     "UPDATE pipeline_runs SET status = ? WHERE run_id = ?",
        ...     ("success", "run-123")
        ... )
        >>> print(f"Updated {rows_affected} rows")
    """
    with get_db_connection(db_path, read_only=False) as conn:
        cursor = conn.execute(query, params or ())
        return cursor.rowcount


def insert_many(
    table: str,
    records: list[dict],
    db_path: Optional[Path] = None,
) -> int:
    """
    Bulk insert records into a table.

    Args:
        table: Table name
        records: List of record dictionaries
        db_path: Database file path

    Returns:
        int: Number of inserted rows

    Example:
        >>> records = [
        ...     {"run_id": "run-1", "status": "success"},
        ...     {"run_id": "run-2", "status": "failed"},
        ... ]
        >>> count = insert_many("pipeline_runs", records)
        >>> print(f"Inserted {count} records")
    """
    if not records:
        return 0

    # Get column names from first record
    columns = list(records[0].keys())
    placeholders = ", ".join("?" * len(columns))
    columns_str = ", ".join(columns)

    query = f"INSERT INTO {table} ({columns_str}) VALUES ({placeholders})"

    with get_db_connection(db_path, read_only=False) as conn:
        # Convert dicts to tuples in correct order
        values = [tuple(record[col] for col in columns) for record in records]
        conn.executemany(query, values)
        return len(records)


def get_latest_run(db_path: Optional[Path] = None) -> Optional[dict]:
    """
    Get the most recent pipeline run.

    Args:
        db_path: Database file path

    Returns:
        dict: Latest run record, or None if no runs exist

    Example:
        >>> run = get_latest_run()
        >>> if run:
        ...     print(f"Latest run: {run['run_id']} ({run['status']})")
    """
    query = """
        SELECT *
        FROM pipeline_runs
        ORDER BY started_at DESC
        LIMIT 1
    """

    results = execute_query(query, db_path=db_path)
    return dict(results[0]) if results else None


def get_quality_trends(
    metric_name: str,
    days: int = 30,
    db_path: Optional[Path] = None,
) -> list[dict]:
    """
    Get quality metric trends over time.

    Args:
        metric_name: Metric name (e.g., "duplicate_rate")
        days: Number of days to look back
        db_path: Database file path

    Returns:
        list: Metric values over time

    Example:
        >>> trends = get_quality_trends("duplicate_rate", days=7)
        >>> for point in trends:
        ...     print(f"{point['date']}: {point['value']:.2%}")
    """
    query = """
        SELECT
            DATE(measured_at) as date,
            AVG(metric_value) as value
        FROM quality_metrics
        WHERE metric_name = ?
          AND measured_at >= DATE('now', ? || ' days')
        GROUP BY DATE(measured_at)
        ORDER BY date DESC
    """

    results = execute_query(query, (metric_name, -days), db_path)
    return [dict(row) for row in results]


def vacuum_database(db_path: Optional[Path] = None) -> None:
    """
    Optimize database by reclaiming space and rebuilding indexes.

    Should be run periodically (e.g., weekly) to maintain performance.

    Args:
        db_path: Database file path

    Example:
        >>> vacuum_database()  # Reclaim space from deleted records
    """
    logger.info("Running VACUUM to optimize database")

    with get_db_connection(db_path, read_only=False) as conn:
        conn.execute("VACUUM")
        conn.execute("ANALYZE")

    logger.info("Database optimization complete")


# Example usage
if __name__ == "__main__":
    from src.utils.config import override_settings

    # Use test database
    settings = override_settings(
        metadata_db_path=Path("logs/test_metadata.db")
    )

    # Initialize
    init_database(settings.metadata_db_path)

    # Insert test data
    with get_db_connection(settings.metadata_db_path) as conn:
        conn.execute(
            """
            INSERT INTO pipeline_runs (
                run_id, pipeline_name, status, records_input
            ) VALUES (?, ?, ?, ?)
            """,
            ("test-run-1", "test_pipeline", "success", 1000),
        )

    # Query
    results = execute_query(
        "SELECT * FROM pipeline_runs WHERE run_id = ?",
        ("test-run-1",),
        settings.metadata_db_path,
    )

    print(f"Found {len(results)} runs")
    for row in results:
        print(dict(row))

    print("\nDatabase utilities test completed!")
