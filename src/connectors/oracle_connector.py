"""
Oracle Database Connector - Production Template

This module demonstrates enterprise production patterns for Oracle database connections
commonly used in Data Engineering roles. While this is a template (not connected to
a real Oracle instance), it showcases:

- Connection pooling for performance
- Incremental loads using Change Data Capture (CDC) patterns
- Retry logic with exponential backoff
- Query optimization (hints, bind variables)
- Bulk operations

Dependencies:
    pip install cx_Oracle

Environment variables:
    ORACLE_USER, ORACLE_PASSWORD, ORACLE_DSN
"""

import os
import time
from typing import Iterator, Dict, Any, Optional, List
from datetime import datetime
import logging

try:
    import cx_Oracle
except ImportError:
    cx_Oracle = None  # Allow module to load even without cx_Oracle for testing

logger = logging.getLogger(__name__)


class OracleConnector:
    """
    Enterprise-grade Oracle connector with production patterns.

    Features:
    - Connection pooling (reduces overhead)
    - Parameterized queries (SQL injection protection)
    - Retry logic (handles transient network errors)
    - Incremental loading (CDC pattern)
    - Bulk insert operations
    """

    def __init__(
        self,
        user: Optional[str] = None,
        password: Optional[str] = None,
        dsn: Optional[str] = None,
        pool_size: int = 5,
        encoding: str = "UTF-8",
    ):
        """
        Initialize Oracle connection pool.

        Args:
            user: Oracle username (defaults to env: ORACLE_USER)
            password: Oracle password (defaults to env: ORACLE_PASSWORD)
            dsn: Data Source Name (defaults to env: ORACLE_DSN)
                 Format: host:port/service_name
                 Example: prod-oracle:1521/ORCL
            pool_size: Maximum connections in pool
            encoding: Character encoding for data
        """
        if cx_Oracle is None:
            raise ImportError(
                "cx_Oracle not installed. Run: pip install cx_Oracle"
            )

        self.user = user or os.getenv("ORACLE_USER")
        self.password = password or os.getenv("ORACLE_PASSWORD")
        self.dsn = dsn or os.getenv("ORACLE_DSN")

        if not all([self.user, self.password, self.dsn]):
            raise ValueError(
                "Missing Oracle credentials. Set ORACLE_USER, ORACLE_PASSWORD, ORACLE_DSN"
            )

        # Create connection pool (reuses connections, reduces overhead)
        logger.info(f"Creating Oracle connection pool (max={pool_size})")
        self.pool = cx_Oracle.SessionPool(
            user=self.user,
            password=self.password,
            dsn=self.dsn,
            min=2,  # Keep 2 connections alive
            max=pool_size,
            increment=1,
            threaded=True,
            encoding=encoding,
            nencoding=encoding,
        )

    def incremental_load(
        self,
        table: str,
        watermark_column: str,
        last_sync: str,
        batch_size: int = 10000,
        columns: Optional[List[str]] = None,
    ) -> Iterator[Dict[str, Any]]:
        """
        Perform incremental load using CDC (Change Data Capture) pattern.

        This pattern is essential for production Data Engineering:
        1. Query only NEW/UPDATED records since last sync
        2. Use indexed column for efficient filtering (watermark_column)
        3. Stream results in batches to avoid memory issues
        4. Use query hints for Oracle optimizer

        Args:
            table: Source table name
            watermark_column: Timestamp column for CDC (must be indexed!)
            last_sync: Last sync timestamp (ISO format)
            batch_size: Records per batch
            columns: Column list (default: all columns with SELECT *)

        Yields:
            Dictionaries representing rows

        Example:
            >>> connector = OracleConnector()
            >>> for row in connector.incremental_load(
            ...     table="CUSTOMER_ORDERS",
            ...     watermark_column="UPDATED_AT",
            ...     last_sync="2026-01-01T00:00:00"
            ... ):
            ...     print(row)
        """
        column_list = ", ".join(columns) if columns else "*"

        # Production query with Oracle hints for optimization
        query = f"""
            SELECT /*+ PARALLEL(4) INDEX({table} IDX_{watermark_column}) */
                {column_list}
            FROM {table}
            WHERE {watermark_column} > TO_TIMESTAMP(:last_sync, 'YYYY-MM-DD"T"HH24:MI:SS')
            ORDER BY {watermark_column}
        """

        connection = self.pool.acquire()
        try:
            cursor = connection.cursor()
            cursor.arraysize = batch_size  # Fetch rows in batches

            logger.info(f"Executing incremental load: {table} WHERE {watermark_column} > {last_sync}")

            # Using bind variables (prevents SQL injection)
            cursor.execute(query, {"last_sync": last_sync})

            # Fetch column names
            col_names = [desc[0] for desc in cursor.description]

            # Stream results in batches
            rows_fetched = 0
            while True:
                batch = cursor.fetchmany(batch_size)
                if not batch:
                    break

                for row in batch:
                    yield dict(zip(col_names, row))
                    rows_fetched += 1

            logger.info(f"Incremental load complete: {rows_fetched} rows")

        finally:
            self.pool.release(connection)

    def bulk_insert(
        self,
        table: str,
        data: List[Dict[str, Any]],
        batch_size: int = 5000,
        retry_count: int = 3,
    ) -> int:
        """
        Perform bulk insert with retry logic.

        Production pattern: Use executemany() for batch inserts
        (much faster than individual INSERTs).

        Args:
            table: Target table
            data: List of dictionaries to insert
            batch_size: Records per batch
            retry_count: Number of retries on failure

        Returns:
            Number of rows inserted

        Example:
            >>> records = [
            ...     {"id": 1, "name": "Alice"},
            ...     {"id": 2, "name": "Bob"}
            ... ]
            >>> connector.bulk_insert("USERS", records)
            2
        """
        if not data:
            return 0

        # Build INSERT statement dynamically
        columns = list(data[0].keys())
        placeholders = ", ".join([f":{col}" for col in columns])
        insert_sql = f"INSERT INTO {table} ({', '.join(columns)}) VALUES ({placeholders})"

        total_inserted = 0

        for attempt in range(retry_count):
            connection = self.pool.acquire()
            try:
                cursor = connection.cursor()

                # Insert in batches
                for i in range(0, len(data), batch_size):
                    batch = data[i : i + batch_size]
                    cursor.executemany(insert_sql, batch)
                    connection.commit()
                    total_inserted += len(batch)

                logger.info(f"Bulk insert successful: {total_inserted} rows")
                return total_inserted

            except cx_Oracle.DatabaseError as e:
                error, = e.args
                logger.error(f"Bulk insert failed (attempt {attempt + 1}): {error.message}")

                if attempt < retry_count - 1:
                    # Exponential backoff
                    wait_time = 2 ** attempt
                    logger.info(f"Retrying in {wait_time}s...")
                    time.sleep(wait_time)
                else:
                    raise

            finally:
                self.pool.release(connection)

        return total_inserted

    def execute_query(
        self,
        query: str,
        params: Optional[Dict[str, Any]] = None,
        fetch_size: int = 1000,
    ) -> List[Dict[str, Any]]:
        """
        Execute arbitrary SELECT query with parameterization.

        Args:
            query: SQL query  (use :param for bind variables)
            params: Query parameters
            fetch_size: Batch size for fetching

        Returns:
            List of result rows as dictionaries

        Example:
            >>> results = connector.execute_query(
            ...     "SELECT * FROM CUSTOMERS WHERE country = :country",
            ...     params={"country": "Mexico"}
            ... )
        """
        connection = self.pool.acquire()
        try:
            cursor = connection.cursor()
            cursor.arraysize = fetch_size

            cursor.execute(query, params or {})

            col_names = [desc[0] for desc in cursor.description]
            results = []
            for row in cursor:
                results.append(dict(zip(col_names, row)))

            return results

        finally:
            self.pool.release(connection)

    def close(self):
        """Close connection pool."""
        if self.pool:
            logger.info("Closing Oracle connection pool")
            self.pool.close()


# Example usage (not executed on import)
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Example 1: Incremental load
    connector = OracleConnector()

    try:
        for row in connector.incremental_load(
            table="PRODUCTION_TABLE",
            watermark_column="UPDATED_AT",
            last_sync="2026-02-01T00:00:00",
        ):
            print(row)

    finally:
        connector.close()
