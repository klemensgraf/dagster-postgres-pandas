from dagster_postgres_pandas.types import PostgresConfig


def test_postgres_config_defaults():
    """Test PostgresConfig with default values."""
    config = PostgresConfig(
        connection_string="postgresql://user:pass@localhost:5432/test"
    )

    assert config.connection_string == "postgresql://user:pass@localhost:5432/test"
    assert config.default_schema == "public"
    assert config.if_exists == "replace"
    assert config.index is False
    assert config.chunk_size is None
    assert config.timeout == 30


def test_postgres_config_custom_values():
    """Test PostgresConfig with custom values."""
    config = PostgresConfig(
        connection_string="postgresql://user:pass@localhost:5432/test",
        default_schema="custom_schema",
        if_exists="append",
        index=True,
        chunk_size=1000,
        timeout=60,
    )

    assert config.connection_string == "postgresql://user:pass@localhost:5432/test"
    assert config.default_schema == "custom_schema"
    assert config.if_exists == "append"
    assert config.index is True
    assert config.chunk_size == 1000
    assert config.timeout == 60
