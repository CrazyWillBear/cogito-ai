from typing import TypedDict

from dbs.QueryAndFilterSchemas import QueryAndFilters


class QueryResult(TypedDict):
    """Schema for a source query and its result."""

    query: str | QueryAndFilters
    source: str
    result: tuple[str, str] | str | None