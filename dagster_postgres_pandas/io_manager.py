"""
PostgreSQL I/O Manager for Dagster with Pandas DataFrame support.

This module provides a custom I/O Manager for Dagster that uses PostgreSQL as the data store.
It supports reading and writing Pandas DataFrames to and from PostgreSQL tables with dynamic schema selection capabilities.
"""

import logging
from typing import Optional

import pandas as pd
import sqlalchemy as sa
from dagster import ConfigurableIOManager, InputContext, OutputContext

from dagster_postgres_pandas.exceptions import (
    InvalidConfigurationError,
    PostgresIOManagerError,
    SchemaNotFoundError,
)
from dagster_postgres_pandas.types import PostgresIfExists

logger = logging.getLogger(__name__)


class PostgresPandasIOManager(ConfigurableIOManager):
    """
    PostgreSQL I/O Manager for Pandas DataFrames.

    This I/O manager provides seamless integration between Dagster assets and PostgreSQL
    databases, with support for dynamic schema selection, automatic schema creation,
    and robust error handling.

    Attributes:
        connection_string: PostgreSQL connection string (supports dg.EnvVar)
        default_schema: Default schema for assets without explicit schema
        if_exists: Behavior when table exists ('fail', 'replace', 'append')
        index: Whether to store DataFrame index
        chunk_size: Number of rows to insert at once (None for all)
        timeout: Connection timeout in seconds

    Example:
        >>> io_manager = PostgresPandasIOManager(
        ...     connection_string=dg.EnvVar("POSTGRES_CONNECTION_STRING"),
        ...     default_schema="analytics",
        ...     if_exists="replace"
        ... )
    """

    connection_string: str
    default_schema: str = "public"
    if_exists: PostgresIfExists = "replace"
    index: bool = False
    chunk_size: Optional[int] = None
    timeout: int = 30
    require_ssl: bool = False
    max_identifier_length: int = 63  # PostgreSQL limit

    def __init__(self, **kwargs):
        """Initialize the IO manager and validate configuration."""
        super().__init__(**kwargs)
        self._validate_config()

    def _validate_config(self):
        """Validate configuration after initialization."""
        # Handle EnvVar objects - get the actual string value
        connection_string_value = self.connection_string
        if hasattr(self.connection_string, "get_value"):
            # This is a Dagster EnvVar object
            try:
                connection_string_value = self.connection_string.get_value()
            except Exception:
                # Can't validate EnvVar at config time, will validate at runtime
                logger.info(
                    "Skipping connection string validation for EnvVar - will validate at runtime"
                )
                return

        if not connection_string_value:
            raise InvalidConfigurationError("Connection string cannot be empty")

        # Validate connection string format
        if not connection_string_value.startswith(
            ("postgresql://", "postgresql+psycopg2://")
        ):
            raise InvalidConfigurationError(
                "Invalid PostgreSQL connection string format"
            )

        # Validate default schema
        if not self._is_valid_identifier(self.default_schema):
            raise InvalidConfigurationError(
                f"Invalid default schema: {self.default_schema}"
            )

    def _get_engine(self) -> sa.Engine:
        """
        Create SQLAlchemy engine with configured connection string.

        Returns:
            SQLAlchemy Engine instance

        Raises:
            ConnectionError: If connection cannot be established
        """
        # Resolve connection string (handle EnvVar objects)
        connection_string_value = self.connection_string
        if hasattr(self.connection_string, "get_value"):
            connection_string_value = self.connection_string.get_value()

        # Validate connection string format at runtime
        if not connection_string_value:
            raise InvalidConfigurationError("Connection string cannot be empty")

        if not connection_string_value.startswith(
            ("postgresql://", "postgresql+psycopg2://")
        ):
            raise InvalidConfigurationError(
                "Invalid PostgreSQL connection string format"
            )

        try:
            engine = sa.create_engine(
                connection_string_value,
                connect_args={
                    "connect_timeout": self.timeout,
                    "sslmode": "require" if self.require_ssl else "prefer",
                },
                pool_pre_ping=True,  # Verify connections before use
                pool_recycle=300,  # Recycle connections after 5 minutes
                pool_size=5,
                max_overflow=10,
                echo=False,  # Never log SQL in production
            )

            # Test connection
            with engine.connect():
                pass
            return engine
        except Exception as e:
            raise ConnectionError(f"Failed to connect to PostgreSQL: {str(e)}") from e

    def _get_schema_and_table_for_output(
        self, context: OutputContext
    ) -> tuple[str, str]:
        """Determine schema and table name for output with validation."""
        schema = None

        # Get schema from metadata/config
        if hasattr(context, "definition_metadata") and context.definition_metadata:
            schema = context.definition_metadata.get("schema")

        if (
            not schema
            and hasattr(context, "resource_config")
            and context.resource_config
        ):
            schema = context.resource_config.get("schema")

        if not schema:
            schema = self.default_schema

        # Validate schema
        if not self._is_valid_identifier(schema):
            raise PostgresIOManagerError(f"Invalid schema name: {schema}")

        # Generate and sanitize table name
        table_name = self._sanitize_table_name("_".join(context.asset_key.path))
        if not table_name:
            raise PostgresIOManagerError(
                "Could not generate valid table name from asset key"
            )

        logger.info(f"Output - Using schema: {schema}, table: {table_name}")
        return schema, table_name

    def _get_schema_and_table_for_input(self, context: InputContext) -> tuple[str, str]:
        """Determine schema and table name for input with validation."""
        schema = None

        # Get schema from upstream output metadata
        if hasattr(context, "upstream_output") and context.upstream_output:
            upstream_context = context.upstream_output
            if (
                hasattr(upstream_context, "definition_metadata")
                and upstream_context.definition_metadata
            ):
                schema = upstream_context.definition_metadata.get("schema")

        # Fallback to default schema
        if not schema:
            schema = self.default_schema

        # Validate schema
        if not self._is_valid_identifier(schema):
            raise PostgresIOManagerError(f"Invalid schema name: {schema}")

        # Generate and sanitize table name
        table_name = self._sanitize_table_name("_".join(context.asset_key.path))
        if not table_name:
            raise PostgresIOManagerError(
                "Could not generate valid table name from asset key"
            )

        logger.info(f"Input - Using schema: {schema}, table: {table_name}")
        return schema, table_name

    def _ensure_schema_exists(self, engine: sa.Engine, schema: str) -> None:
        """Ensure that the schema exists, create if necessary."""
        if schema == "public":
            return

        # Validate schema name to prevent injection
        if not self._is_valid_identifier(schema):
            raise PostgresIOManagerError(f"Invalid schema name: {schema}")

        try:
            with engine.connect() as conn:
                # Use quoted identifier for safe schema creation
                conn.execute(sa.text(f'CREATE SCHEMA IF NOT EXISTS "{schema}"'))
                conn.commit()
                logger.info(f"Ensured schema exists: {schema}")
        except Exception as e:
            raise PostgresIOManagerError(
                f"Failed to create schema {schema}: {str(e)}"
            ) from e

    def _is_valid_identifier(self, name: str) -> bool:
        """Validate SQL identifier to prevent injection."""
        import re

        if not name or len(name) > self.max_identifier_length:
            return False
        # Allow alphanumeric, underscore, must start with letter or underscore
        return bool(re.match(r"^[a-zA-Z_][a-zA-Z0-9_]*$", name))

    def _sanitize_table_name(self, name: str) -> str:
        """
        Sanitize table name to be PostgreSQL compliant.

        Rules:
        1. Remove/ignore special characters rather than replacing with underscores
        2. Ensure starts with letter or underscore
        3. Only allow alphanumeric and underscores
        4. Truncate to max length
        """
        import re

        if not name:
            return ""

        # Remove special characters, keep only alphanumeric and underscores
        sanitized = re.sub(r"[^a-zA-Z0-9_]", "", name)

        # If starts with number, prefix with underscore
        if sanitized and sanitized[0].isdigit():
            sanitized = f"_{sanitized}"

        # If completely empty after sanitization, return empty
        if not sanitized:
            return ""

        # Truncate to PostgreSQL limit
        return sanitized[: self.max_identifier_length]  # PostgreSQL identifier limit

    def _table_exists(self, engine: sa.Engine, schema: str, table_name: str) -> bool:
        """
        Check if table exists in the specified schema.

        Args:
            engine: SQLAlchemy engine
            schema: Schema name
            table_name: Table name

        Returns:
            True if table exists, False otherwise
        """
        try:
            with engine.connect() as conn:
                result = conn.execute(
                    sa.text(
                        """
                        SELECT EXISTS (
                            SELECT FROM information_schema.tables
                            WHERE table_schema = :schema
                            AND table_name = :table_name
                        )
                        """
                    ),
                    {"schema": schema, "table_name": table_name},
                )
                return result.scalar()
        except Exception as e:
            logger.warning(f"Error checking table existence: {str(e)}")
            return False

    def handle_output(self, context: OutputContext, obj: pd.DataFrame) -> None:
        """
        Store a Pandas DataFrame in PostgreSQL.

        Args:
            context: Output context with asset information
            obj: Pandas DataFrame to store

        Raises:
            ValueError: If obj is not a pandas DataFrame
            PostgresIOManagerError: If storage operation fails
        """
        if not isinstance(obj, pd.DataFrame):
            raise ValueError(f"Expected Pandas DataFrame, got {type(obj)}")

        if obj.empty:
            logger.warning("Attempting to store empty DataFrame")
            return

        engine = self._get_engine()
        schema, table_name = self._get_schema_and_table_for_output(context)

        try:
            # Ensure schema exists
            self._ensure_schema_exists(engine, schema)

            # Store DataFrame
            with engine.connect() as conn:
                obj.to_sql(
                    name=table_name,
                    con=conn,
                    schema=schema,
                    if_exists=self.if_exists,
                    index=self.index,
                    method="multi",
                    chunksize=self.chunk_size,
                )
                conn.commit()

            logger.info(
                f"Successfully saved DataFrame with {len(obj)} rows and "
                f"{len(obj.columns)} columns to {schema}.{table_name}"
            )

        except Exception as e:
            error_msg = f"Error saving DataFrame to {schema}.{table_name}: {str(e)}"
            logger.error(error_msg)
            raise PostgresIOManagerError(error_msg) from e
        finally:
            engine.dispose()

    def load_input(self, context: InputContext) -> pd.DataFrame:
        """
        Load a Pandas DataFrame from PostgreSQL.

        Args:
            context: Input context with asset information

        Returns:
            Loaded Pandas DataFrame

        Raises:
            SchemaNotFoundError: If required table does not exist
            PostgresIOManagerError: If loading operation fails
        """
        engine = self._get_engine()
        schema, table_name = self._get_schema_and_table_for_input(context)

        try:
            # Check if table exists
            if not self._table_exists(engine, schema, table_name):
                raise SchemaNotFoundError(
                    f"Table {schema}.{table_name} does not exist. "
                    f"Make sure the upstream asset has been materialized."
                )

            # Load DataFrame
            with engine.connect() as conn:
                df = pd.read_sql_table(table_name=table_name, con=conn, schema=schema)

            logger.info(
                f"Successfully loaded DataFrame with {len(df)} rows and "
                f"{len(df.columns)} columns from {schema}.{table_name}"
            )

            return df

        except SchemaNotFoundError:
            raise
        except Exception as e:
            error_msg = f"Error loading DataFrame from {schema}.{table_name}: {str(e)}"
            logger.error(error_msg)
            raise PostgresIOManagerError(error_msg) from e
        finally:
            engine.dispose()
