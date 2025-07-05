from unittest.mock import MagicMock

import pandas as pd
import pytest
from dagster import InputContext, OutputContext

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
