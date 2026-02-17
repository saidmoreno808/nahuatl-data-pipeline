"""
Custom Exceptions

Domain-specific exceptions for better error handling and debugging.

Example:
    >>> raise DataValidationError("Invalid record format", record_id=123)
"""


class CorcNahException(Exception):
    """Base exception for all CORC-NAH errors."""

    def __init__(self, message: str, **context):
        """
        Initialize exception with message and optional context.

        Args:
            message: Error message
            **context: Additional context for debugging
        """
        self.message = message
        self.context = context
        super().__init__(message)

    def __str__(self):
        """Format exception with context."""
        if self.context:
            context_str = ", ".join(f"{k}={v}" for k, v in self.context.items())
            return f"{self.message} ({context_str})"
        return self.message

    def to_dict(self):
        """Convert to dictionary for structured logging."""
        return {
            "error_type": self.__class__.__name__,
            "message": self.message,
            "context": self.context,
        }


class ConfigurationError(CorcNahException):
    """
    Configuration error.

    Raised when settings are invalid or missing.

    Example:
        >>> raise ConfigurationError("Invalid ratio", ratio=1.5)
    """
    pass


class DataValidationError(CorcNahException):
    """
    Data validation error.

    Raised when data fails validation checks.

    Example:
        >>> raise DataValidationError("Missing required field", field="es")
    """
    pass


class DataLoadError(CorcNahException):
    """
    Data loading error.

    Raised when data cannot be loaded from source.

    Example:
        >>> raise DataLoadError("File not found", path="data.jsonl")
    """
    pass


class DataTransformError(CorcNahException):
    """
    Data transformation error.

    Raised when transformation fails.

    Example:
        >>> raise DataTransformError("Normalization failed", text="bad")
    """
    pass


class PipelineError(CorcNahException):
    """
    Pipeline execution error.

    Raised when pipeline fails during execution.

    Example:
        >>> raise PipelineError("Pipeline failed", stage="normalize")
    """
    pass


class DatabaseError(CorcNahException):
    """
    Database operation error.

    Raised when database operations fail.

    Example:
        >>> raise DatabaseError("Connection failed", db_path="metadata.db")
    """
    pass


class MetricsError(CorcNahException):
    """
    Metrics tracking error.

    Raised when metrics cannot be tracked or saved.

    Example:
        >>> raise MetricsError("Failed to save metrics", metric="duration")
    """
    pass
