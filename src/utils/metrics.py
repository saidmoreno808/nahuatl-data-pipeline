"""
Performance Metrics Tracking

Provides utilities for tracking pipeline performance metrics.
Integrates with the metadata database for historical analysis.

Example:
    >>> from src.utils.metrics import track_time
    >>> @track_time
    ... def process_data():
    ...     # ... processing logic
    ...     pass
"""

import time
from contextlib import contextmanager
from datetime import datetime
from functools import wraps
from typing import Any, Callable, Dict, Generator, Optional

from src.utils.config import get_settings
from src.utils.db import get_db_connection
from src.utils.logger import get_logger

logger = get_logger(__name__)


class MetricsTracker:
    """
    Context manager for tracking operation metrics.

    Automatically logs start/end times and metrics to database.

    Example:
        >>> with MetricsTracker("data_ingestion") as tracker:
        ...     # ... do work
        ...     tracker.record("records_processed", 1000)
        ...     tracker.record("errors_count", 5)
    """

    def __init__(
        self,
        operation_name: str,
        run_id: Optional[str] = None,
        auto_log: bool = True,
    ):
        """
        Initialize metrics tracker.

        Args:
            operation_name: Name of the operation being tracked
            run_id: Pipeline run ID (for correlation)
            auto_log: Automatically log to database on exit
        """
        self.operation_name = operation_name
        self.run_id = run_id or f"{operation_name}-{int(time.time())}"
        self.auto_log = auto_log

        self.start_time: Optional[float] = None
        self.end_time: Optional[float] = None
        self.metrics: Dict[str, Any] = {}
        self.success: bool = True
        self.error_message: Optional[str] = None

    def __enter__(self) -> "MetricsTracker":
        """Start tracking."""
        self.start_time = time.time()
        logger.info(
            f"Started {self.operation_name}",
            extra={
                "operation": self.operation_name,
                "run_id": self.run_id,
            },
        )
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Stop tracking and optionally log to database."""
        self.end_time = time.time()
        duration = self.end_time - self.start_time

        # Mark as failed if exception occurred
        if exc_type is not None:
            self.success = False
            self.error_message = str(exc_val)

        # Log completion
        logger.info(
            f"Completed {self.operation_name}",
            extra={
                "operation": self.operation_name,
                "run_id": self.run_id,
                "duration_seconds": round(duration, 2),
                "success": self.success,
                **self.metrics,
            },
        )

        # Save to database if enabled
        if self.auto_log:
            try:
                self._save_to_database(duration)
            except Exception as e:
                logger.error(
                    f"Failed to save metrics to database: {e}",
                    exc_info=True,
                )

        # Don't suppress exceptions
        return False

    def record(self, metric_name: str, value: Any) -> None:
        """
        Record a metric value.

        Args:
            metric_name: Metric name
            value: Metric value

        Example:
            >>> tracker.record("records_processed", 1000)
        """
        self.metrics[metric_name] = value
        logger.debug(
            f"Recorded metric: {metric_name}={value}",
            extra={"metric": metric_name, "value": value},
        )

    def increment(self, metric_name: str, amount: int = 1) -> None:
        """
        Increment a counter metric.

        Args:
            metric_name: Metric name
            amount: Amount to increment by

        Example:
            >>> tracker.increment("errors_count")
        """
        self.metrics[metric_name] = self.metrics.get(metric_name, 0) + amount

    def _save_to_database(self, duration: float) -> None:
        """Save metrics to database."""
        settings = get_settings()

        with get_db_connection() as conn:
            # Insert pipeline run
            conn.execute(
                """
                INSERT INTO pipeline_runs (
                    run_id, pipeline_name, started_at, ended_at,
                    status, duration_seconds, error_message,
                    records_input, records_output, records_filtered
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    self.run_id,
                    self.operation_name,
                    datetime.fromtimestamp(self.start_time).isoformat(),
                    datetime.fromtimestamp(self.end_time).isoformat(),
                    "success" if self.success else "failed",
                    round(duration, 2),
                    self.error_message,
                    self.metrics.get("records_input"),
                    self.metrics.get("records_output"),
                    self.metrics.get("records_filtered"),
                ),
            )

            # Insert individual metrics
            for metric_name, metric_value in self.metrics.items():
                # Skip if already saved as part of pipeline_runs
                if metric_name in {
                    "records_input",
                    "records_output",
                    "records_filtered",
                }:
                    continue

                # Only save numeric metrics
                if isinstance(metric_value, (int, float)):
                    conn.execute(
                        """
                        INSERT INTO quality_metrics (
                            run_id, metric_name, metric_value, metric_unit
                        ) VALUES (?, ?, ?, ?)
                        """,
                        (self.run_id, metric_name, float(metric_value), None),
                    )


@contextmanager
def track_time(
    operation_name: str,
) -> Generator[Dict[str, Any], None, None]:
    """
    Context manager for simple time tracking without database logging.

    Args:
        operation_name: Name of the operation

    Yields:
        dict: Metrics dictionary (can be modified during execution)

    Example:
        >>> with track_time("data_loading") as metrics:
        ...     data = load_data()
        ...     metrics["records_loaded"] = len(data)
    """
    start_time = time.time()
    metrics: Dict[str, Any] = {}

    logger.debug(f"Started {operation_name}")

    try:
        yield metrics
    finally:
        duration = time.time() - start_time
        logger.info(
            f"Completed {operation_name}",
            extra={
                "operation": operation_name,
                "duration_seconds": round(duration, 2),
                **metrics,
            },
        )


def time_function(func: Callable) -> Callable:
    """
    Decorator to time function execution.

    Args:
        func: Function to time

    Returns:
        Callable: Wrapped function

    Example:
        >>> @time_function
        ... def slow_operation():
        ...     time.sleep(1)
        ...     return "done"
    """

    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        duration = time.time() - start_time

        logger.info(
            f"Function {func.__name__} completed",
            extra={
                "function": func.__name__,
                "duration_seconds": round(duration, 2),
            },
        )

        return result

    return wrapper


# Example usage
if __name__ == "__main__":
    from src.utils.logger import configure_root_logger

    configure_root_logger()

    # Example 1: Using MetricsTracker
    print("Example 1: MetricsTracker")
    with MetricsTracker("example_operation", auto_log=False) as tracker:
        time.sleep(0.1)
        tracker.record("records_processed", 1000)
        tracker.record("errors_count", 5)
        tracker.increment("warnings_count", 3)

    # Example 2: Using track_time context manager
    print("\nExample 2: track_time context manager")
    with track_time("data_loading") as metrics:
        time.sleep(0.05)
        metrics["records"] = 500

    # Example 3: Using decorator
    print("\nExample 3: time_function decorator")

    @time_function
    def example_function():
        time.sleep(0.05)
        return "completed"

    result = example_function()
    print(f"Result: {result}")

    print("\nMetrics utilities test completed!")
