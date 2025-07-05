"""Custom exceptions for PostgreSQL I/O Manager."""


class PostgresIOManagerError(Exception):
    """Base exception for PostgreSQL I/O Manager."""

    pass


class SchemaNotFoundError(PostgresIOManagerError):
    """Raised when a required schema is not found."""

    pass


class ConnectionError(PostgresIOManagerError):
    """Raised when database connection fails."""

    pass


class InvalidConfigurationError(PostgresIOManagerError):
    """Raised when configuration is invalid."""

    pass
