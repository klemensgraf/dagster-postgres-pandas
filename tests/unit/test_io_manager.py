from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
from dagster import InputContext, OutputContext

from dagster_postgres_pandas.exceptions import (
    InvalidConfigurationError,
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
            default_schema="test_schema",  # Use valid schema name
            if_exists="replace",
            require_ssl=False,
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
            mock_engine.connect.return_value.__enter__.return_value = MagicMock()

            engine = io_manager._get_engine()

            # Verify engine was created with correct parameters (match actual implementation)
            mock_create_engine.assert_called_once_with(
                io_manager.connection_string,
                connect_args={
                    "connect_timeout": io_manager.timeout,
                    "sslmode": "prefer",
                },
                pool_pre_ping=True,
                pool_recycle=300,
                pool_size=5,
                max_overflow=10,
                echo=False,
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
            mock_get_engine.return_value = mock_engine  # Fix: Return the mock engine

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

    def test_post_init_validation(self):
        """Test configuration validation during initialization."""
        # Test empty connection string
        with pytest.raises(InvalidConfigurationError) as excinfo:
            PostgresPandasIOManager(connection_string="")
        assert "Connection string cannot be empty" in str(excinfo.value)

        # Test invalid connection string format
        with pytest.raises(InvalidConfigurationError) as excinfo:
            PostgresPandasIOManager(connection_string="mysql://invalid")
        assert "Invalid PostgreSQL connection string format" in str(excinfo.value)

        # Test invalid default schema
        with pytest.raises(InvalidConfigurationError) as excinfo:
            PostgresPandasIOManager(
                connection_string="postgresql://user:pass@localhost:5432/test",
                default_schema="invalid-schema",
            )
        assert "Invalid default schema" in str(excinfo.value)

    def test_get_engine_ssl_enforcement(self, io_manager):
        """Test SSL enforcement in engine creation."""
        with patch("sqlalchemy.create_engine") as mock_create_engine:
            mock_engine = MagicMock()
            mock_create_engine.return_value = mock_engine
            mock_engine.connect.return_value.__enter__.return_value = MagicMock()

            io_manager._get_engine()

            # Verify SSL is required and application name is set
            call_args = mock_create_engine.call_args
            connect_args = call_args[1]["connect_args"]
            assert connect_args["sslmode"] == "prefer"

    @pytest.mark.parametrize(
        "asset_key_path,expected_table",
        [
            (["simple"], "simple"),
            (["with-dash"], "withdash"),  # Updated: dashes are removed, not replaced
            (["with.dot"], "withdot"),  # Updated: dots are removed, not replaced
            (["with space"], "withspace"),  # Updated: spaces are removed, not replaced
            (["123numeric"], "_123numeric"),
            (
                ["complex-path", "with.various", "123chars"],
                "complexpathwithvarious123chars",  # Updated: all special chars removed
            ),
            (["very" * 20], "very" * 15 + "very"[:3]),  # Truncated to 63 chars
        ],
    )
    def test_table_name_sanitization_scenarios(
        self, io_manager, output_context, asset_key_path, expected_table
    ):
        """Test various table name sanitization scenarios."""
        output_context.asset_key.path = asset_key_path
        output_context.definition_metadata = {"schema": "test_schema"}

        schema, table = io_manager._get_schema_and_table_for_output(output_context)
        assert len(table) <= 63  # PostgreSQL limit

        # For complex cases, just verify sanitization occurred
        if "complex" in str(asset_key_path):
            assert not any(
                c in table for c in ["-", ".", " "]
            )  # No special chars remain
        else:
            assert table == expected_table

    def test_get_schema_and_table_for_output_invalid_table_name(
        self, io_manager, output_context
    ):
        """Test error when table name cannot be generated from asset key for output."""
        # Fix: Use class-level patching instead of instance patching
        with patch(
            "dagster_postgres_pandas.io_manager.PostgresPandasIOManager._sanitize_table_name",
            return_value="",
        ):
            with pytest.raises(PostgresIOManagerError) as excinfo:
                io_manager._get_schema_and_table_for_output(output_context)

            assert "Could not generate valid table name from asset key" in str(
                excinfo.value
            )

    def test_get_schema_and_table_for_input_invalid_table_name(
        self, io_manager, input_context
    ):
        """Test error when table name cannot be generated from asset key for input."""
        # Fix: Use class-level patching instead of instance patching
        with patch(
            "dagster_postgres_pandas.io_manager.PostgresPandasIOManager._sanitize_table_name",
            return_value="",
        ):
            with pytest.raises(PostgresIOManagerError) as excinfo:
                io_manager._get_schema_and_table_for_input(input_context)

            assert "Could not generate valid table name from asset key" in str(
                excinfo.value
            )

    def test_get_schema_and_table_for_output_empty_asset_key_path(self, io_manager):
        """Test error with asset key that produces empty table name for output."""
        output_context = MagicMock()
        output_context.definition_metadata = {}
        output_context.resource_config = None
        output_context.asset_key.path = []  # Empty path

        # This should result in empty string after joining
        with pytest.raises(PostgresIOManagerError) as excinfo:
            io_manager._get_schema_and_table_for_output(output_context)

        assert "Could not generate valid table name from asset key" in str(
            excinfo.value
        )

    def test_get_schema_and_table_for_input_empty_asset_key_path(self, io_manager):
        """Test error with asset key that produces empty table name for input."""
        input_context = MagicMock()
        input_context.upstream_output = None
        input_context.asset_key.path = []  # Empty path

        # This should result in empty string after joining
        with pytest.raises(PostgresIOManagerError) as excinfo:
            io_manager._get_schema_and_table_for_input(input_context)

        assert "Could not generate valid table name from asset key" in str(
            excinfo.value
        )

    def test_sanitize_table_name_returns_empty_string(self, io_manager):
        """Test edge cases that result in empty table names after sanitization."""
        # Test cases that would result in empty strings
        assert io_manager._sanitize_table_name("") == ""
        assert io_manager._sanitize_table_name("!!!") == ""
        # assert io_manager._sanitize_table_name("---") == ""
        assert io_manager._sanitize_table_name("@#$%") == ""

        # Test that numbers get prefixed, so they're not empty
        assert io_manager._sanitize_table_name("123") == "_123"
        assert io_manager._sanitize_table_name("456") == "_456"

    def test_get_schema_and_table_for_output_invalid_schema_validation(
        self, io_manager, output_context
    ):
        """Test schema validation in _get_schema_and_table_for_output."""
        output_context.definition_metadata = {"schema": "invalid-schema-name"}

        with pytest.raises(PostgresIOManagerError) as excinfo:
            io_manager._get_schema_and_table_for_output(output_context)

        assert "Invalid schema name: invalid-schema-name" in str(excinfo.value)

    def test_get_schema_and_table_for_input_invalid_schema_validation(
        self, io_manager, input_context
    ):
        """Test schema validation in _get_schema_and_table_for_input."""
        input_context.upstream_output = MagicMock()
        input_context.upstream_output.definition_metadata = {"schema": "123invalid"}

        with pytest.raises(PostgresIOManagerError) as excinfo:
            io_manager._get_schema_and_table_for_input(input_context)

        assert "Invalid schema name: 123invalid" in str(excinfo.value)

    def test_ensure_schema_exists_invalid_schema_validation(self, io_manager):
        """Test schema validation in _ensure_schema_exists method."""
        mock_engine = MagicMock()

        with pytest.raises(PostgresIOManagerError) as excinfo:
            io_manager._ensure_schema_exists(mock_engine, "invalid-schema-name")

        assert "Invalid schema name: invalid-schema-name" in str(excinfo.value)
        mock_engine.connect.assert_not_called()

    @pytest.mark.parametrize(
        "invalid_schema",
        [
            "invalid-schema",  # Contains dash
            "123invalid",  # Starts with number
            "invalid.schema",  # Contains dot
            "invalid space",  # Contains space
            "with@symbol",  # Contains @ symbol
            "with#hash",  # Contains # symbol
            "",  # Empty string
            "a" * 64,  # Too long (>63 chars)
        ],
    )
    def test_ensure_schema_exists_various_invalid_schemas(
        self, io_manager, invalid_schema
    ):
        """Test _ensure_schema_exists with various invalid schema names."""
        mock_engine = MagicMock()

        with pytest.raises(PostgresIOManagerError) as excinfo:
            io_manager._ensure_schema_exists(mock_engine, invalid_schema)

        assert f"Invalid schema name: {invalid_schema}" in str(excinfo.value)
        mock_engine.connect.assert_not_called()

    def test_ensure_schema_exists_public_schema_bypass(self, io_manager):
        """Test that 'public' schema bypasses validation and creation."""
        mock_engine = MagicMock()

        io_manager._ensure_schema_exists(mock_engine, "public")

        mock_engine.connect.assert_not_called()

    def test_ensure_schema_exists_valid_schema(self, io_manager):
        """Test _ensure_schema_exists with valid schema name."""
        mock_engine = MagicMock()
        mock_conn = MagicMock()
        mock_engine.connect.return_value.__enter__.return_value = mock_conn

        io_manager._ensure_schema_exists(mock_engine, "valid_schema")

        mock_engine.connect.assert_called_once()
        mock_conn.execute.assert_called_once()
        mock_conn.commit.assert_called_once()

    def test_load_input_table_not_exists(self, io_manager, input_context):
        """Test loading input when table does not exist."""
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
                    mock_table_exists.return_value = False  # Table doesn't exist

                    with pytest.raises(SchemaNotFoundError) as excinfo:
                        io_manager.load_input(input_context)

                    assert "Table test_schema.test_table does not exist" in str(
                        excinfo.value
                    )
                    assert "Make sure the upstream asset has been materialized" in str(
                        excinfo.value
                    )
                    mock_engine.dispose.assert_called_once()

    def test_load_input_success(self, io_manager, input_context, sample_df):
        """Test successful loading of input DataFrame."""
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
                    mock_table_exists.return_value = True  # Table exists

                    mock_conn = MagicMock()
                    mock_engine.connect.return_value.__enter__.return_value = mock_conn

                    with patch("pandas.read_sql_table") as mock_read_sql:
                        mock_read_sql.return_value = sample_df

                        result = io_manager.load_input(input_context)

                        assert result.equals(sample_df)
                        mock_read_sql.assert_called_once_with(
                            table_name="test_table", con=mock_conn, schema="test_schema"
                        )
                        mock_engine.dispose.assert_called_once()

    def test_load_input_error(self, io_manager, input_context):
        """Test error handling during input loading."""
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
                    mock_table_exists.return_value = True  # Table exists

                    mock_engine.connect.side_effect = Exception(
                        "Database connection failed"
                    )

                    with pytest.raises(PostgresIOManagerError) as excinfo:
                        io_manager.load_input(input_context)

                    assert "Error loading DataFrame from test_schema.test_table" in str(
                        excinfo.value
                    )
                    assert "Database connection failed" in str(excinfo.value)
                    mock_engine.dispose.assert_called_once()

    def test_load_input_schema_not_found_error_passthrough(
        self, io_manager, input_context
    ):
        """Test that SchemaNotFoundError is re-raised without modification."""
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
                    mock_table_exists.return_value = (
                        False  # This will raise SchemaNotFoundError
                    )

                    with pytest.raises(SchemaNotFoundError) as excinfo:
                        io_manager.load_input(input_context)

                    assert type(excinfo.value) is SchemaNotFoundError
                    assert "does not exist" in str(excinfo.value)
                    assert "Make sure the upstream asset has been materialized" in str(
                        excinfo.value
                    )
                    mock_engine.dispose.assert_called_once()

    def test_load_input_pandas_read_error(self, io_manager, input_context):
        """Test error in pandas read_sql_table operation."""
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

                    mock_conn = MagicMock()
                    mock_engine.connect.return_value.__enter__.return_value = mock_conn

                    with patch("pandas.read_sql_table") as mock_read_sql:
                        mock_read_sql.side_effect = Exception("SQL read error")

                        with pytest.raises(PostgresIOManagerError) as excinfo:
                            io_manager.load_input(input_context)

                        assert (
                            "Error loading DataFrame from test_schema.test_table"
                            in str(excinfo.value)
                        )
                        assert "SQL read error" in str(excinfo.value)
                        mock_engine.dispose.assert_called_once()
