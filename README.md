# Dagster Postgres Pandas I/O Manager

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

### Code Quality

This project uses several tools to ensure code quality:

-   **Ruff**: Linting and formatting (replaces Black, isort, flake8, and mypy)

```bash
# Run all quality checks
ruff check .
ruff format --check .

# Fix linting issues automatically
ruff check --fix .
```
