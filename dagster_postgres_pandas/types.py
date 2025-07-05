"""Type definitions for PostgreSQL I/O Manager."""

from dataclasses import dataclass
from typing import Literal, Optional

PostgresIfExists = Literal["fail", "replace", "append"]


@dataclass
class PostgresConfig:
    """Configuration for PostgreSQL I/O Manager."""

    connection_string: str
    default_schema: str = "public"
    if_exists: PostgresIfExists = "replace"
    index: bool = False
    chunk_size: Optional[int] = None
    timeout: int = 30
