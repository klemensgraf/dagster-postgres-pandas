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

    def _get_engine(self) -> sa.Engine:
        """
        Create SQLAlchemy engine with configured connection string.

        Returns:
            SQLAlchemy Engine instance

        Raises:
            ConnectionError: If connection cannot be established
        """
        try:
            engine = sa.create_engine(
                self.connection_string,
                connect_args={"connect_timeout": self.timeout},
                pool_pre_ping=True,  # Verify connections before use
                pool_recycle=500,  # Recycle connections after 5 minutes
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
        """
        Determine schema and table name for output (storing).

        Schema determination priority:
        1. Asset metadata: metadata={"schema": "schema_name"}
        2. Resource config: schema from resource configuration
        3. Default schema: fallback value from default_schema

        Args:
            context: Output context with asset information

        Returns:
            Tuple of (schema_name, table_name)
        """

        schema = None

        # 1. Check asset metadata
        if hasattr(context, "definition_metadata") and context.definition_metadata:
            schema = context.definition_metadata.get("schema")

        # 2. Check resource config
        if (
            not schema
            and hasattr(context, "resource_config")
            and context.resource_config
        ):
            schema = context.resource_config.get("schema")

        # 3. Use default schema
        if not schema:
            schema = self.default_schema

        # Generate table name from asset key
        table_name = "_".join(context.asset_key.path)

        logger.info(f"Output - Using schema: {schema}, table: {table_name}")
        return schema, table_name

    def _get_schema_and_table_for_input(
        self, context: OutputContext
    ) -> tuple[str, str]:
        """
        Determine schema and table name for input (loading).

        For inputs, the schema must be determined from the upstream asset!

        Args:
            context: Input context with asset information

        Returns:
            Tuple of (schema_name, table_name)
        """
        schema = None

        # 1. Check upstream output metadata
        if hasattr(context, "upstream_output") and context.upstream_output:
            upstream_context = context.upstream_output
            if (
                hasattr(upstream_context, "definition_metadata")
                and upstream_context.definition_metadata
            ):
                schema = upstream_context.definition_metadata.get("schema")

        # 2. Fallback to default schema
        if not schema:
            schema = self.default_schema

        # Generate table name from asset key
        table_name = "_".join(context.asset_key.path)

        logger.info(f"Input - Using schema: {schema}, table: {table_name}")
        return schema, table_name

    def _ensure_schema_exists(self, engine: sa.Engine, schema: str) -> None:
        """
        Ensure that the schema exists, create if necessary.

        Args:
            engine: SQLAlchemy engine
            schema: Schema name to create

        Raises:
            PostgresIOManagerError: If schema creation fails
        """
        if schema == "public":  # the public schema always exists
            return

        try:
            with engine.connect() as conn:
                conn.execute(sa.text(f"CREATE SCHEMA IF NOT EXISTS {schema}"))
                conn.commit()
                logger.info(f"Ensured schema exists: {schema}")
        except Exception as e:
            raise PostgresIOManagerError(
                f"Failed to create schema {schema}: {str(e)}"
            ) from e

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

        except SchemaNotFoundError:
            raise
        except Exception as e:
            error_msg = f"Error loading DataFrame from {schema}.{table_name}: {str(e)}"
            logger.error(error_msg)
            raise PostgresIOManagerError(error_msg) from e
        finally:
            engine.dispose()
