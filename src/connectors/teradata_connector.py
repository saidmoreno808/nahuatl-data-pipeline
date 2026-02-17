"""
Teradata Connector - Production Template

This module demonstrates production patterns for Teradata database connections.
Teradata is commonly used in enterprise Data Warehousing for analytics workloads
and is a key skill for Data Engineers in financial services and retail.

Key concepts:
- Query bands for workload management
- TPT (Teradata Parallel Transporter) for bulk loads
- FastLoad/MultiLoad optimization
- Connection best practices

Dependencies:
    pip install teradatasql

Environment variables:
    TERADATA_HOST, TERADATA_USER, TERADATA_PASSWORD, TERADATA_DATABASE
"""

import os
import logging
from typing import Iterator, Dict, Any, Optional, List
from datetime import datetime
from contextlib import contextmanager

try:
    import teradatasql
except ImportError:
    teradatasql = None

logger = logging.getLogger(__name__)


class TeradataConnector:
    """
    Enterprise Teradata connector with production patterns.

    Features:
    - Query bands (workload classification)
    - Bulk loading strategies (FastLoad, MultiLoad)
    - Connection pooling patterns
    - Incremental extraction
    """

    def __init__(
        self,
        host: Optional[str] = None,
        user: Optional[str] = None,
        password: Optional[str] = None,
        database: Optional[str] = None,
        tmode: str = "ANSI",  # Or "TERA"
    ):
        """
        Initialize Teradata connection.

        Args:
            host: Teradata server hostname/IP
            user: Database username
            password: Database password
            database: Default database name
            tmode: Transaction mode (ANSI or TERA)
        """
        if teradatasql is None:
            raise ImportError(
                "teradatasql not installed. Run: pip install teradatasql"
            )

        self.host = host or os.getenv("TERADATA_HOST")
        self.user = user or os.getenv("TERADATA_USER")
        self.password = password or os.getenv("TERADATA_PASSWORD")
        self.database = database or os.getenv("TERADATA_DATABASE")
        self.tmode = tmode

        if not all([self.host, self.user, self.password]):
            raise ValueError(
                "Missing Teradata credentials. Set TERADATA_HOST, TERADATA_USER, TERADATA_PASSWORD"
            )

        # Build connection string
        self.conn_string = {
            "host": self.host,
            "user": self.user,
            "password": self.password,
            "tmode": self.tmode,
        }

        if self.database:
            self.conn_string["database"] = self.database

        logger.info(f"Teradata connector initialized: {self.host}/{self.database}")

    @contextmanager
    def get_connection(self, query_band: Optional[str] = None):
        """
        Context manager for connection with automatic cleanup.

        Args:
            query_band: Optional query band for workload management

        Example:
            >>> with connector.get_connection("Application=ETL;") as conn:
            ...     cursor = conn.cursor()
            ...     cursor.execute("SELECT * FROM ORDERS")
        """
        connection = teradatasql.connect(**self.conn_string)
        try:
            # Set query band if provided (for workload management)
            if query_band:
                cursor = connection.cursor()
                cursor.execute(
                    f"SET QUERY_BAND = '{query_band}' FOR SESSION"
                )
                logger.debug(f"Query band set: {query_band}")

            yield connection

        finally:
            connection.close()

    def set_query_band(self, connection, app_name: str, env: str = "PROD"):
        """
        Set query band for workload management.

        Query bands allow DBAs to:
        - Track resource usage by application
        - Apply priority rules
        - Generate usage reports

        Args:
            connection: Active Teradata connection
            app_name: Application name
            env: Environment (PROD, DEV, TEST)

        Example:
            >>> with connector.get_connection() as conn:
            ...     connector.set_query_band(conn, "CORC_NAH_ETL", "PROD")
            ...     # All subsequent queries will be tagged
        """
        query_band = f"Application={app_name};Environment={env};Version=1.0;"

        cursor = connection.cursor()
        cursor.execute(
            f"SET QUERY_BAND = '{query_band}' FOR SESSION"
        )
        logger.info(f"Query band configured: {query_band}")

    def incremental_load(
        self,
        table: str,
        watermark_column: str,
        last_sync: str,
        batch_size: int = 10000,
        columns: Optional[List[str]] = None,
    ) -> Iterator[Dict[str, Any]]:
        """
        Incremental load pattern for Teradata.

        Similar to Oracle connector but optimized for Teradata:
        - Uses COLLECT STATISTICS hints
        - Teradata-specific date formatting

        Args:
            table: Source table
            watermark_column: Timestamp column for CDC
            last_sync: Last sync timestamp (ISO format)
            batch_size: Fetch batch size
            columns: Column list (default: *)

        Yields:
            Row dictionaries
        """
        column_list = ", ".join(columns) if columns else "*"

        # Teradata query with optimization hints
        query = f"""
            SELECT {column_list}
            FROM {table}
            WHERE {watermark_column} > CAST(? AS TIMESTAMP)
            ORDER BY {watermark_column}
        """

        with self.get_connection(query_band="Application=IncrementalLoad;") as conn:
            cursor = conn.cursor()
            cursor.arraysize = batch_size

            logger.info(
                f"Starting incremental load: {table} WHERE {watermark_column} > {last_sync}"
            )

            cursor.execute(query, (last_sync,))

            # Get column names
            col_names = [desc[0] for desc in cursor.description]

            rows_fetched = 0
            while True:
                batch = cursor.fetchmany(batch_size)
                if not batch:
                    break

                for row in batch:
                    yield dict(zip(col_names, row))
                    rows_fetched += 1

            logger.info(f"Incremental load complete: {rows_fetched} rows")

    def bulk_load_fastload(
        self, table: str, data: List[Dict[str, Any]], truncate: bool = False
    ) -> int:
        """
        Simulate Teradata FastLoad pattern.

        FastLoad is used for loading large amounts of data into empty tables.

        NOTE: This is a simplified pattern. Production FastLoad uses TPT:
        - TPT (Teradata Parallel Transporter) CLI tool
        - Or JDBC FastLoad driver

        Args:
            table: Target table (must be empty or truncate=True)
            data: Records to insert
            truncate: Whether to truncate table first

        Returns:
            Number of rows inserted

        Production alternative:
            Use TPT script:
            ```bash
            tbuild -f fastload_job.txt
            ```
        """
        if not data:
            return 0

        with self.get_connection(query_band="Application=BulkLoad;") as conn:
            cursor = conn.cursor()

            # Truncate if requested
            if truncate:
                logger.warning(f"Truncating table: {table}")
                cursor.execute(f"DELETE FROM {table} ALL")

            # Build INSERT statement
            columns = list(data[0].keys())
            placeholders = ", ".join(["?" for _ in columns])
            insert_sql = f"INSERT INTO {table} ({', '.join(columns)}) VALUES ({placeholders})"

            # FastLoad optimization: disable logging
            cursor.execute(f"SET TABLE {table} LOGGING = NO")

            # Insert in batches
            batch_size = 5000
            total_inserted = 0

            for i in range(0, len(data), batch_size):
                batch = data[i : i + batch_size]
                values_list = [tuple(row[col] for col in columns) for row in batch]

                cursor.executemany(insert_sql, values_list)
                total_inserted += len(batch)

                if total_inserted % 50000 == 0:
                    logger.info(f"Inserted {total_inserted} rows...")

            # Re-enable logging
            cursor.execute(f"SET TABLE {table} LOGGING = YES")

            # Collect statistics for optimizer
            cursor.execute(f"COLLECT STATISTICS ON {table}")

            conn.commit()
            logger.info(f"FastLoad complete: {total_inserted} rows")

            return total_inserted

    def execute_query(
        self,
        query: str,
        params: Optional[tuple] = None,
        fetch_size: int = 1000,
    ) -> List[Dict[str, Any]]:
        """
        Execute SELECT query.

        Args:
            query: SQL query (use ? for parameters)
            params: Query parameters (tuple)
            fetch_size: Batch fetch size

        Returns:
            List of row dictionaries
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.arraysize = fetch_size

            cursor.execute(query, params or ())

            col_names = [desc[0] for desc in cursor.description]
            results = []

            for row in cursor:
                results.append(dict(zip(col_names, row)))

            return results

    def get_table_stats(self, table: str) -> Dict[str, Any]:
        """
        Get table statistics (row count, size, indexes).

        Useful for pipeline monitoring and optimization.

        Args:
            table: Table name

        Returns:
            Dictionary with table metadata
        """
        query = f"""
            SELECT
                COUNT(*) as row_count,
                SUM(CURRENTPERM) / 1024 / 1024 as size_mb,
                MAX(modifieddate) as last_modified
            FROM DBC.TABLESIZE
            WHERE TABLENAME = ?
        """

        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(query, (table,))
            row = cursor.fetchone()

            if row:
                col_names = [desc[0] for desc in cursor.description]
                return dict(zip(col_names, row))
            else:
                return {}


# Example usage
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    connector = TeradataConnector()

    # Example 1: Query with query band
    with connector.get_connection("Application=Analytics;") as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT TOP 10 * FROM CUSTOMERS")
        for row in cursor:
            print(row)

    # Example 2: Incremental load
    for row in connector.incremental_load(
        table="SALES_FACT",
        watermark_column="SALE_DATE",
        last_sync="2026-02-01 00:00:00",
    ):
        print(row)
