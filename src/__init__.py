"""
CORC-NAH: Corpus Náhuatl/Maya ETL Pipeline

Professional ETL pipeline for indigenous language data processing.
"""

__version__ = "2.0.0"
__author__ = "Said Moreno"
__license__ = "MIT"

# Public API
from src.utils.config import Settings, get_settings
from src.utils.logger import get_logger

__all__ = ["Settings", "get_settings", "get_logger", "__version__"]
