from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
from dagster import InputContext, OutputContext

from dagster_postgres_pandas.exceptions import (
    PostgresIOManagerError,
    SchemaNotFoundError,
)
from dagster_postgres_pandas.io_manager import PostgresPandasIOManager


class TestPostgresPandasIOManager:
    @pytest.fixture
    def io_manager(self):
        """Create a basic IO manager instance for testing."""
        return PostgresPandasIOManager(
            connection_string="postgresql://user:pass@localhost:5432/test",
            default_schema="test_schema",
            if_exists="replace",
        )

    @pytest.fixture
    def mock_engine(self):
        """Create a mock SQLAlchemy engine."""
        mock = MagicMock()
        mock.connect.return_value.__enter__.return_value = MagicMock()
        return mock

    @pytest.fixture
    def sample_df(self):
        """Create a sample DataFrame for testing."""
        return pd.DataFrame({"id": [1, 2, 3], "name": ["test1", "test2", "test3"]})

    @pytest.fixture
    def output_context(self):
        """Create a mock output context."""
        context = MagicMock(spec=OutputContext)
        context.asset_key.path = ["test_table"]
        context.definition_metadata = {"schema": "custom_schema"}
        context.resource_config = MagicMock()
        context.resource_config.get.return_value = None
        return context

    @pytest.fixture
    def input_context(self):
        """Create mock input context."""
        context = MagicMock(spec=InputContext)
        context.asset_key.path = ["test_table"]
        context.upstream_output = MagicMock()
        context.upstream_output.definition_metadata = {"schema": "custom_schema"}
        return context

    def test_get_schema_and_table_for_output(self, io_manager, output_context):
        """Test schema and table name determination for output."""
        schema, table = io_manager._get_schema_and_table_for_output(output_context)
        assert schema == "custom_schema"
        assert table == "test_table"

        # Test fallback to default schema
        output_context.definition_metadata = {}
        schema, table = io_manager._get_schema_and_table_for_output(output_context)
        assert schema == "test_schema"
        assert table == "test_table"

        # Test resource config fallback
        output_context.resource_config.get.return_value = "resource_schema"
        schema, table = io_manager._get_schema_and_table_for_output(output_context)
        assert schema == "resource_schema"
        assert table == "test_table"

    def test_get_schema_and_table_for_input(self, io_manager, input_context):
        """Test schema and table name determination for input."""
        schema, table = io_manager._get_schema_and_table_for_input(input_context)
        assert schema == "custom_schema"
        assert table == "test_table"

        # Test fallback to default schema
        input_context.upstream_output.definition_metadata = {}
        schema, table = io_manager._get_schema_and_table_for_input(input_context)
        assert schema == "test_schema"
        assert table == "test_table"

    def test_get_engine(self, io_manager):
        """Test engine creation"""
        with patch("sqlalchemy.create_engine") as mock_create_engine:
            mock_engine = MagicMock()
            mock_create_engine.return_value = mock_engine

            engine = io_manager._get_engine()

            # Verify engine was created with correct parameters
            mock_create_engine.assert_called_once_with(
                io_manager.connection_string,
                connect_args={"connect_timeout": io_manager.timeout},
                pool_pre_ping=True,
                pool_recycle=500,
            )
            assert engine == mock_engine

    def test_get_engine_error(self, io_manager):
        """Test engine creation error handling."""
        with patch("sqlalchemy.create_engine") as mock_create_engine:
            mock_create_engine.side_effect = Exception("Connection error")

            with pytest.raises(ConnectionError) as excinfo:
                io_manager._get_engine()

            assert "Failed to connect to PostgreSQL" in str(excinfo.value)

    def test_ensure_schema_exists(self, io_manager, mock_engine):
        """Test schema creation."""
        # Test with non-public schema
        io_manager._ensure_schema_exists(mock_engine, "test_schema")

        mock_conn = mock_engine.connect.return_value.__enter__.return_value
        mock_conn.execute.assert_called_once()
        mock_conn.commit.assert_called_once()

        # Reset mock
        mock_engine.reset_mock()

        # Test with public schema (should do nothing)
        io_manager._ensure_schema_exists(mock_engine, "public")
        mock_engine.connect.assert_not_called()

    def test_ensure_schema_exists_error(self, io_manager, mock_engine):
        """Test schema creation error handling."""
        mock_conn = mock_engine.connect.return_value.__enter__.return_value
        mock_conn.execute.side_effect = Exception("Schema creation error")

        with pytest.raises(PostgresIOManagerError) as excinfo:
            io_manager._ensure_schema_exists(mock_engine, "test_schema")

        assert "Failed to create schema" in str(excinfo.value)

    def test_table_exists(self, io_manager, mock_engine):
        """Test table existence check."""
        mock_conn = mock_engine.connect.return_value.__enter__.return_value
        mock_result = mock_conn.execute.return_value
        mock_result.scalar.return_value = True

        result = io_manager._table_exists(mock_engine, "test_schema", "test_table")
        assert result is True

        # Test when table doesn't exist
        mock_result.scalar.return_value = False
        result = io_manager._table_exists(mock_engine, "test_schema", "test_table")
        assert result is False

    def test_table_exists_error(self, io_manager, mock_engine):
        """Test table existence check error handling."""
        mock_conn = mock_engine.connect.return_value.__enter__.return_value
        mock_conn.execute.side_effect = Exception("Table check error")

        with patch("dagster_postgres_pandas.io_manager.logger.warning") as mock_warning:
            result = io_manager._table_exists(mock_engine, "test_schema", "test_table")
            assert result is False
            mock_warning.assert_called_once()

    def test_handle_output(self, io_manager, sample_df, output_context):
        """Test storing DataFrame."""
        with patch(
            "dagster_postgres_pandas.io_manager.PostgresPandasIOManager._get_engine"
        ) as mock_get_engine:
            mock_engine = MagicMock()
            mock_get_engine.return_value = mock_engine

            with patch(
                "dagster_postgres_pandas.io_manager.PostgresPandasIOManager._get_schema_and_table_for_output"
            ) as mock_get_schema:
                mock_get_schema.return_value = ("test_schema", "test_table")

                with patch(
                    "dagster_postgres_pandas.io_manager.PostgresPandasIOManager._ensure_schema_exists"
                ) as mock_ensure_schema:
                    with patch.object(sample_df, "to_sql") as mock_to_sql:
                        io_manager.handle_output(output_context, sample_df)

                        # Verify all methods were called correctly
                        mock_get_engine.assert_called_once()
                        mock_get_schema.assert_called_once_with(output_context)
                        mock_ensure_schema.assert_called_once_with(
                            mock_engine, "test_schema"
                        )
                        mock_to_sql.assert_called_once()
                        mock_engine.dispose.assert_called_once()

    def test_handle_output_not_dataframe(self, io_manager, output_context):
        """Test storing non-DataFrame."""
        with pytest.raises(ValueError) as excinfo:
            io_manager.handle_output(output_context, "not a dataframe")

        assert "Expected Pandas DataFrame" in str(excinfo.value)

    def test_handle_output_empty_dataframe(self, io_manager, output_context):
        """Test storing empty DataFrame."""
        empty_df = pd.DataFrame()

        with patch("dagster_postgres_pandas.io_manager.logger.warning") as mock_warning:
            io_manager.handle_output(output_context, empty_df)
            mock_warning.assert_called_once()

    def test_handle_output_error(self, io_manager, sample_df, output_context):
        """Test storing DataFrame error handling."""
        with patch(
            "dagster_postgres_pandas.io_manager.PostgresPandasIOManager._get_engine"
        ) as mock_get_engine:
            mock_engine = MagicMock()
            mock_get_engine.return_value = mock_engine

            with patch(
                "dagster_postgres_pandas.io_manager.PostgresPandasIOManager._get_schema_and_table_for_output"
            ) as mock_get_schema:
                mock_get_schema.return_value = ("test_schema", "test_table")

                with patch(
                    "dagster_postgres_pandas.io_manager.PostgresPandasIOManager._ensure_schema_exists"
                ) as mock_ensure_schema:
                    mock_ensure_schema.side_effect = Exception("Storage error")

                    with pytest.raises(PostgresIOManagerError) as excinfo:
                        io_manager.handle_output(output_context, sample_df)

                    assert "Error saving DataFrame" in str(excinfo.value)
                    mock_engine.dispose.assert_called_once()

    def test_load_input(self, io_manager, input_context):
        """Test loading DataFrame."""
        with patch(
            "dagster_postgres_pandas.io_manager.PostgresPandasIOManager._get_engine"
        ) as mock_get_engine:
            mock_engine = MagicMock()
            mock_get_engine.return_value = mock_engine

            with patch(
                "dagster_postgres_pandas.io_manager.PostgresPandasIOManager._get_schema_and_table_for_input"
            ) as mock_get_schema:
                mock_get_schema.return_value = ("test_schema", "test_table")

                with patch(
                    "dagster_postgres_pandas.io_manager.PostgresPandasIOManager._table_exists"
                ) as mock_table_exists:
                    mock_table_exists.return_value = True

                    # Create mock connection
                    mock_conn = MagicMock()
                    mock_engine.connect.return_value.__enter__.return_value = mock_conn

                    with patch("pandas.read_sql_table") as mock_read_sql:
                        mock_df = MagicMock()
                        mock_read_sql.return_value = mock_df

                        result = io_manager.load_input(input_context)

                        # Verify all methods were called correctly
                        mock_get_engine.assert_called_once()
                        mock_get_schema.assert_called_once_with(input_context)
                        mock_table_exists.assert_called_once_with(
                            mock_engine, "test_schema", "test_table"
                        )
                        mock_read_sql.assert_called_once_with(
                            table_name="test_table",
                            con=mock_conn,
                            schema="test_schema",
                        )
                        assert result == mock_df
                        mock_engine.dispose.assert_called_once()

    def test_load_input_table_not_found(self, io_manager, input_context):
        """Test loading DataFrame when table doesn't exist."""
        with patch(
            "dagster_postgres_pandas.io_manager.PostgresPandasIOManager._get_engine"
        ) as mock_get_engine:
            mock_engine = MagicMock()
            mock_get_engine.return_value = mock_engine

            with patch(
                "dagster_postgres_pandas.io_manager.PostgresPandasIOManager._get_schema_and_table_for_input"
            ) as mock_get_schema:
                mock_get_schema.return_value = ("test_schema", "test_table")

                with patch(
                    "dagster_postgres_pandas.io_manager.PostgresPandasIOManager._table_exists"
                ) as mock_table_exists:
                    mock_table_exists.return_value = False

                    with pytest.raises(SchemaNotFoundError) as excinfo:
                        io_manager.load_input(input_context)

                    assert "Table test_schema.test_table does not exist" in str(
                        excinfo.value
                    )
                    mock_engine.dispose.assert_called_once()

    def test_load_input_error(self, io_manager, input_context):
        """Test loading DataFrame error handling."""
        with patch(
            "dagster_postgres_pandas.io_manager.PostgresPandasIOManager._get_engine"
        ) as mock_get_engine:
            mock_engine = MagicMock()
            mock_get_engine.return_value = mock_engine

            with patch(
                "dagster_postgres_pandas.io_manager.PostgresPandasIOManager._get_schema_and_table_for_input"
            ) as mock_get_schema:
                mock_get_schema.return_value = ("test_schema", "test_table")

                with patch(
                    "dagster_postgres_pandas.io_manager.PostgresPandasIOManager._table_exists"
                ) as mock_table_exists:
                    mock_table_exists.return_value = True

                    with patch("pandas.read_sql_table") as mock_read_sql:
                        mock_read_sql.side_effect = Exception("Loading error")

                        with pytest.raises(PostgresIOManagerError) as excinfo:
                            io_manager.load_input(input_context)

                        assert "Error loading DataFrame" in str(excinfo.value)
                        mock_engine.dispose.assert_called_once()
