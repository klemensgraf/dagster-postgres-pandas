"""
Dagster PostgreSQL Pandas I/O Manager

A robust I/O Manager for PostgreSQL with Pandas DataFrame support.
"""

from .exceptions import (
    ConnectionError,
    InvalidConfigurationError,
    PostgresIOManagerError,
    SchemaNotFoundError,
)
from .io_manager import (
    PostgresPandasIOManager,
)
from .types import PostgresConfig

__version__ = "0.1.0"
__all__ = [
    "PostgresPandasIOManager",
    "PostgresIOManagerError",
    "SchemaNotFoundError",
    "ConnectionError",
    "InvalidConfigurationError",
    "PostgresConfig",
]
