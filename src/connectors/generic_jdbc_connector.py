"""
Generic JDBC Connector using JayDeBeApi

This connector provides a universal interface to ANY database with a JDBC driver.
Useful for:
- PostgreSQL, MySQL, SQL Server
- DB2, SAP HANA
- Any enterprise/legacy database

Dependencies:
    pip install JayDeBeApi

Requirements:
    1. Java Runtime (JRE 8+)
    2. JDBC driver JAR file for target database

Environment variables:
    JDBC_DRIVER_PATH=/path/to/driver.jar
"""

import os
import logging
from typing import Iterator, Dict, Any, Optional, List
from pathlib import Path

try:
    import jaydebeapi
except ImportError:
    jaydebeapi = None

logger = logging.getLogger(__name__)


class GenericJDBCConnector:
    """
    Universal JDBC connector for enterprise databases.

    Supported databases:
    - PostgreSQL: org.postgresql.Driver
    - MySQL: com.mysql.cj.jdbc.Driver
    - SQL Server: com.microsoft.sqlserver.jdbc.SQLServerDriver
    - DB2: com.ibm.db2.jcc.DB2Driver
    - Oracle: oracle.jdbc.OracleDriver (alternative to cx_Oracle)
    """

    # Common JDBC drivers
    DRIVERS = {
        "postgresql": {
            "class": "org.postgresql.Driver",
            "url_template": "jdbc:postgresql://{host}:{port}/{database}",
            "default_port": 5432,
        },
        "mysql": {
            "class": "com.mysql.cj.jdbc.Driver",
            "url_template": "jdbc:mysql://{host}:{port}/{database}",
            "default_port": 3306,
        },
        "sqlserver": {
            "class": "com.microsoft.sqlserver.jdbc.SQLServerDriver",
            "url_template": "jdbc:sqlserver://{host}:{port};databaseName={database}",
            "default_port": 1433,
        },
        "db2": {
            "class": "com.ibm.db2.jcc.DB2Driver",
            "url_template": "jdbc:db2://{host}:{port}/{database}",
            "default_port": 50000,
        },
    }

    def __init__(
        self,
        db_type: str,
        host: str,
        database: str,
        user: str,
        password: str,
        port: Optional[int] = None,
        driver_path: Optional[str] = None,
    ):
        """
        Initialize generic JDBC connector.

        Args:
            db_type: Database type ("postgresql", "mysql", "sqlserver", "db2")
            host: Database hostname
            database: Database name
            user: Username
            password: Password
            port: Port (defaults to standard port for db_type)
            driver_path: Path to JDBC driver JAR (defaults to JDBC_DRIVER_PATH env var)

        Example:
            >>> connector = GenericJDBCConnector(
            ...     db_type="postgresql",
            ...     host="localhost",
            ...     database="production",
            ...     user="etl_user",
            ...     password="secret123"
            ... )
        """
        if jaydebeapi is None:
            raise ImportError("JayDeBeApi not installed. Run: pip install JayDeBeApi")

        if db_type not in self.DRIVERS:
            raise ValueError(
                f"Unsupported db_type: {db_type}. Choose from: {list(self.DRIVERS.keys())}"
            )

        self.db_type = db_type
        self.driver_info = self.DRIVERS[db_type]

        # Build JDBC URL
        port = port or self.driver_info["default_port"]
        self.jdbc_url = self.driver_info["url_template"].format(
            host=host, port=port, database=database
        )

        self.user = user
        self.password = password

        # JDBC driver
        self.driver_class = self.driver_info["class"]
        self.driver_path = driver_path or os.getenv("JDBC_DRIVER_PATH")

        if not self.driver_path:
            logger.warning(
                "No JDBC driver path specified. Set JDBC_DRIVER_PATH or pass driver_path argument"
            )

        logger.info(f"JDBC connector initialized: {self.jdbc_url}")

    def connect(self):
        """
        Create JDBC connection.

        Returns:
            JayDeBeApi connection object
        """
        if self.driver_path and not Path(self.driver_path).exists():
            raise FileNotFoundError(
                f"JDBC driver not found: {self.driver_path}\n"
                f"Download from database vendor website"
            )

        logger.debug(f"Connecting via JDBC: {self.jdbc_url}")

        conn = jaydebeapi.connect(
            self.driver_class,
            self.jdbc_url,
            [self.user, self.password],
            self.driver_path,
        )

        return conn

    def execute_query(
        self, query: str, params: Optional[tuple] = None
    ) -> List[Dict[str, Any]]:
        """
        Execute SELECT query.

        Args:
            query: SQL query (use ? for parameters)
            params: Query parameters

        Returns:
            List of row dictionaries
        """
        conn = self.connect()
        try:
            cursor = conn.cursor()
            cursor.execute(query, params or ())

            # Get column names
            col_names = [desc[0] for desc in cursor.description]

            results = []
            for row in cursor:
                results.append(dict(zip(col_names, row)))

            return results

        finally:
            conn.close()

    def incremental_load(
        self,
        table: str,
        watermark_column: str,
        last_sync: str,
        batch_size: int = 10000,
    ) -> Iterator[Dict[str, Any]]:
        """
        Generic incremental load pattern.

        Args:
            table: Table name
            watermark_column: Timestamp column
            last_sync: Last sync timestamp
            batch_size: Fetch batch size

        Yields:
            Row dictionaries
        """
        # Generic SQL (works across most databases)
        query = f"""
            SELECT *
            FROM {table}
            WHERE {watermark_column} > ?
            ORDER BY {watermark_column}
        """

        conn = self.connect()
        try:
            cursor = conn.cursor()
            cursor.execute(query, (last_sync,))

            col_names = [desc[0] for desc in cursor.description]

            while True:
                batch = cursor.fetchmany(batch_size)
                if not batch:
                    break

                for row in batch:
                    yield dict(zip(col_names, row))

        finally:
            conn.close()

    @staticmethod
    def get_driver_download_links() -> Dict[str, str]:
        """
        Get download links for JDBC drivers.

        Returns:
            Dictionary mapping database to driver download URL
        """
        return {
            "PostgreSQL": "https://jdbc.postgresql.org/download/",
            "MySQL": "https://dev.mysql.com/downloads/connector/j/",
            "SQL Server": "https://learn.microsoft.com/en-us/sql/connect/jdbc/download-microsoft-jdbc-driver-for-sql-server",
            "DB2": "https://www.ibm.com/support/pages/db2-jdbc-driver-versions-and-downloads",
        }


# Example usage
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Example 1: PostgreSQL
    pg_connector = GenericJDBCConnector(
        db_type="postgresql",
        host="localhost",
        database="analytics",
        user="readonly_user",
        password="password123",
        driver_path="/path/to/postgresql-42.6.0.jar",
    )

    results = pg_connector.execute_query(
        "SELECT * FROM customers WHERE country = ?", ("Mexico",)
    )
    print(f"Found {len(results)} customers")

    # Example 2: SQL Server incremental load
    sql_connector = GenericJDBCConnector(
        db_type="sqlserver",
        host="prod-sqlserver",
        database="ERP",
        user="etl_service",
        password="super_secret",
        driver_path="/path/to/mssql-jdbc-12.2.0.jar",
    )

    for row in sql_connector.incremental_load(
        table="SALES",
        watermark_column="UpdatedDate",
        last_sync="2026-02-01 00:00:00",
    ):
        print(row)
