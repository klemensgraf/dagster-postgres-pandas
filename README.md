# Dagster Postgres Pandas I/O Manager

[![PyPI version](https://badge.fury.io/py/dagster-postgres-pandas.svg)](https://badge.fury.io/py/dagster-postgres-pandas)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A robust PostgreSQL I/O manager for [Dagster](https://dagster.io/) with Pandas DataFrame support. This package provides seamless integration between Dagster assets and PostgreSQL databases, featuring dynamic schema selection, automatic schema creation, and comprehensive error handling.

## Features

-   🚀 **Easy Integration**: Drop-in replacement for Dagster's built-in I/O managers
-   🎯 **Dynamic Schema Selection**: Flexible schema assignment per asset using metadata
-   📊 **Pandas Native**: Optimized for Pandas DataFrames with chunked operations
-   🔧 **Auto Schema Creation**: Automatically creates schemas when they don't exist
-   🛡️ **Robust Error Handling**: Comprehensive error messages and connection management
-   ⚡ **Performance Optimized**: Connection pooling and efficient bulk operations
-   🔒 **Production Ready**: Timeout handling, connection retries, and detailed logging

## Installation

```bash
uv add dagster-postgres-pandas
```

### Requirements

-   Python 3.10+
-   Dagster 1.8.0+
-   Pandas 2.1.0+
-   PostgreSQL database

## Quick Start

### 1. Set up your environment

```bash
export POSTGRES_CONNECTION_STRING="postgresql://user:password@localhost:5432/database"
```

### 2. Basic usage

```python
import dagster as dg
import pandas as pd
from dagster_postgres_pandas import PostgresPandasIOManager

# Define your assets
@dg.asset
def raw_data() -> pd.DataFrame:
    """Load raw data."""
    return pd.DataFrame({
        'id': [1, 2, 3],
        'name': ['Alice', 'Bob', 'Charlie'],
        'value': [100, 200, 300]
    })

@dg.asset
def processed_data(raw_data: pd.DataFrame) -> pd.DataFrame:
    """Process the raw data."""
    return raw_data.assign(value_doubled=raw_data['value'] * 2)

# Configure Dagster
defs = dg.Definitions(
    assets=[raw_data, processed_data],
    resources={
        "io_manager": PostgresPandasIOManager(
            connection_string=dg.EnvVar("POSTGRES_CONNECTION_STRING")
        )
    }
)
```

### 3. Run your pipeline

```bash
dg dev
```

or

```bash
dagster dev
```

Your DataFrames will be automatically stored in PostgreSQL tables and loaded when needed by downstream assets.

## Configuration

### Basic Configuration

```python
from dagster_postgres_pandas import PostgresPandasIOManager

io_manager = PostgresPandasIOManager(
    connection_string="postgresql://user:password@localhost:5432/database",
    default_schema="analytics",
    if_exists="replace",
    index=False,
    timeout=30
)
```

### Configuration Options

| Parameter           | Type   | Default     | Description                                              |
| ------------------- | ------ | ----------- | -------------------------------------------------------- |
| `connection_string` | `str`  | Required    | PostgreSQL connection string                             |
| `default_schema`    | `str`  | `"public"`  | Default schema for assets                                |
| `if_exists`         | `str`  | `"replace"` | Behavior when table exists (`fail`, `replace`, `append`) |
| `index`             | `bool` | `False`     | Whether to store DataFrame index                         |
| `chunk_size`        | `int`  | `None`      | Number of rows to insert at once (None for all at once)  |
| `timeout`           | `int`  | `30`        | Connection timeout in seconds                            |

### Using Environment Variables (Recommended)

```python
from dagster_postgres_pandas import PostgresPandasIOManager
import dagster as dg

# Recommended approach for production
io_manager = PostgresPandasIOManager(
    connection_string=dg.EnvVar("POSTGRES_CONNECTION_STRING"),
    default_schema="analytics"
)
```

## Advanced Usage

### Schema Management

#### Per-Asset Schema Configuration

```python
@dg.asset(
    metadata={"schema": "analytics"}
)
def sales_data() -> pd.DataFrame:
    """This asset will be stored in the 'analytics' schema."""
    return pd.DataFrame({"sales": [100, 200, 300]})

@dg.asset(
    metadata={"schema": "raw"}
)
def raw_sales_data() -> pd.DataFrame:
    """This asset will be stored in the 'raw' schema."""
    return pd.DataFrame({"raw_sales": [95, 205, 295]})
```

#### Schema Priority

The I/O manager determines the schema in this order:

1. **Asset metadata**: `metadata={"schema": "schema_name"}`
2. **Resource configuration**: `schema` parameter in resource config
3. **Default schema**: `default_schema` parameter

### Large DataFrames

For large DataFrames, use chunked operations:

```python
io_manager = PostgresPandasIOManager(
    connection_string=dg.EnvVar("POSTGRES_CONNECTION_STRING"),
    chunk_size=10000,  # Insert 10k rows at a time
    timeout=120  # Longer timeout for large operations
)
```

## Connection String Format

PostgreSQL connection strings can be formatted in several ways:

```python
# Basic format
"postgresql://username:password@host:port/database"

# With SSL
"postgresql://username:password@host:port/database?sslmode=require"

# With additional parameters
"postgresql://username:password@host:port/database?sslmode=require&connect_timeout=30"

# Environment variable (recommended for production)
connection_string=dg.EnvVar("POSTGRES_CONNECTION_STRING")
```

## Error Handling

The package provides specific exceptions for different error conditions:

```python
from dagster_postgres_pandas import (
    PostgresIOManagerError,
    SchemaNotFoundError,
    ConnectionError,
    InvalidConfigurationError
)

try:
    # Your Dagster code
    pass
except SchemaNotFoundError:
    # Handle missing table/schema
    print("Required table doesn't exist. Make sure upstream assets are materialized.")
except ConnectionError:
    # Handle database connection issues
    print("Could not connect to PostgreSQL database.")
except PostgresIOManagerError:
    # Handle other I/O manager errors
    print("General I/O manager error occurred.")
```

## Examples

### Multi-Schema Pipeline

```python
import dagster as dg
import pandas as pd
from dagster_postgres_pandas import PostgresPandasIOManager

@dg.asset(metadata={"schema": "raw"})
def raw_users() -> pd.DataFrame:
    """Load raw user data."""
    return pd.DataFrame({
        'user_id': [1, 2, 3],
        'name': ['Alice', 'Bob', 'Charlie'],
        'email': ['alice@example.com', 'bob@example.com', 'charlie@example.com']
    })

@dg.asset(metadata={"schema": "staging"})
def staged_users(raw_users: pd.DataFrame) -> pd.DataFrame:
    """Clean and validate user data."""
    return raw_users.dropna().copy()

@dg.asset(metadata={"schema": "analytics"})
def user_analytics(staged_users: pd.DataFrame) -> pd.DataFrame:
    """Generate user analytics."""
    return staged_users.assign(
        name_length=staged_users['name'].str.len(),
        email_domain=staged_users['email'].str.split('@').str[1]
    )

defs = dg.Definitions(
    assets=[raw_users, staged_users, user_analytics],
    resources={
        "io_manager": PostgresPandasIOManager(
            connection_string=dg.EnvVar("POSTGRES_CONNECTION_STRING"),
            default_schema="public"
        )
    }
)
```

### Time Series Data with Append Mode

```python
from datetime import datetime, timedelta
import pandas as pd
import dagster as dg
from dagster_postgres_pandas import PostgresPandasIOManager

@dg.asset(metadata={"schema": "timeseries"})
def daily_metrics() -> pd.DataFrame:
    """Generate daily metrics that should be appended, not replaced."""
    today = datetime.now().date()
    return pd.DataFrame({
        'date': [today - timedelta(days=i) for i in range(3)],
        'metric_value': [100, 110, 95],
        'metric_name': ['sales', 'sales', 'sales']
    })

# Configure I/O manager for append mode
defs = dg.Definitions(
    assets=[daily_metrics],
    resources={
        "io_manager": PostgresPandasIOManager(
            connection_string=dg.EnvVar("POSTGRES_CONNECTION_STRING"),
            if_exists="append"
        )
    }
)
```

### Different I/O Managers per Asset Group

```python
from dagster_postgres_pandas import PostgresPandasIOManager

# Different configurations for different asset groups
raw_io_manager = PostgresPandasIOManager(
    connection_string=dg.EnvVar("POSTGRES_CONNECTION_STRING"),
    default_schema="raw",
    if_exists="replace"
)

analytics_io_manager = PostgresPandasIOManager(
    connection_string=dg.EnvVar("POSTGRES_CONNECTION_STRING"),
    default_schema="analytics",
    if_exists="replace",
    index=True
)

@dg.asset(io_manager_key="raw_io_manager")
def raw_data() -> pd.DataFrame:
    return pd.DataFrame({"value": [1, 2, 3]})

@dg.asset(io_manager_key="analytics_io_manager")
def processed_data(raw_data: pd.DataFrame) -> pd.DataFrame:
    return raw_data * 2

defs = dg.Definitions(
    assets=[raw_data, processed_data],
    resources={
        "raw_io_manager": raw_io_manager,
        "analytics_io_manager": analytics_io_manager
    }
)
```

## Development

### Setting up Development Environment

```bash
# Clone the repository
git clone https://github.com/klemensgraf/dagster-postgres-pandas.git
cd dagster-postgres-pandas

# Create virtual environment & install dev and test dependencies
uv sync --extra dev --extra test

# Run linting
ruff check .
ruff format .
```

### Running Tests

This project uses pytest for testing. To run the tests:

```bash
# Install test dependencies
uv sync --extra test
# or
uv sync --extra dev

# Run all tests
pytest

# Run with coverage report
pytest --cov=dagster_postgres_pandas

# Run only unit tests
pytest tests/unit/
```

### Code Quality

This project uses several tools to ensure code quality:

-   **Ruff**: Linting and formatting (replaces Black, isort, flake8, and mypy)
-   **Pytest**: Unit tests

```bash
# Run all quality checks
ruff check .
ruff format --check .

# Fix linting issues automatically
ruff check --fix .

# Run all tests
pytest
```

## Troubleshooting

### Common Issues

**Connection refused errors:**

-   Ensure PostgreSQL is running
-   Check connection string format
-   Verify network connectivity and firewall settings

**Schema not found errors:**

-   The I/O manager automatically creates schemas, but ensure you have CREATE privileges
-   Check that the upstream asset has been materialized

**Large DataFrame performance:**

-   Use `chunk_size` parameter for large DataFrames
-   Increase `timeout` for long-running operations
-   Consider using connection pooling parameters

**Import errors:**

-   Ensure all dependencies are installed: `pip install -e ".[dev]"`
-   Check Python version compatibility (3.10+)

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

### Development Process

1. Create an issue describing the bug or new feature, mention if you're open to be assigned to it
1. Wait for getting assigned to the issue
1. Fork the repository
1. Create your feature branch (`git checkout -b feature/amazing-feature`)
1. Make your changes
1. Add tests for your changes
1. Run the test suite (`pytest`)
1. Run code quality checks
1. Commit your changes (`git commit -m 'Add amazing feature'`)
1. Push to the branch (`git push origin feature/amazing-feature`)
1. Open a Pull Request

### Reporting Issues

When reporting issues, please include:

-   Python version
-   Dagster version
-   Database version
-   Complete error traceback
-   Minimal example to reproduce the issue

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Support

If you encounter any issues or have questions:

-   **GitHub Issues**: [Open an issue](https://github.com/klemensgraf/dagster-postgres-pandas/issues)
-   **Dagster Community**: Join the [Dagster Slack](https://dagster.io/slack)
-   **Documentation**: Check the [Dagster documentation](https://docs.dagster.io/)

## Acknowledgments

-   Built on top of the excellent [Dagster](https://dagster.io/) framework
-   Powered by [Pandas](https://pandas.pydata.org/) and [SQLAlchemy](https://www.sqlalchemy.org/)
-   Inspired by the Dagster community's need for robust database I/O solutions

---

Made by [Klemens Graf](https://github.com/klemensgraf)
