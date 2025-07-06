from dagster_postgres_pandas.exceptions import (
    ConnectionError,
    InvalidConfigurationError,
    PostgresIOManagerError,
    SchemaNotFoundError,
)


def test_exception_hierarchy():
    """Test exception inheritance hierarchy."""
    # All custom exceptions should inherit from PostgresIOManagerError
    assert issubclass(SchemaNotFoundError, PostgresIOManagerError)
    assert issubclass(ConnectionError, PostgresIOManagerError)
    assert issubclass(InvalidConfigurationError, PostgresIOManagerError)


def test_exception_messages():
    """Test exception messages."""
    schema_error = SchemaNotFoundError("Schema 'test' not found")
    assert str(schema_error) == "Schema 'test' not found"

    conn_error = ConnectionError("Failed to connect to database")
    assert str(conn_error) == "Failed to connect to database"

    config_error = InvalidConfigurationError("Invalid if_exists value")
    assert str(config_error) == "Invalid if_exists value"
