from typing import TypedDict

from ai.research_agent.schemas.Citation import Citation
from dbs.QueryAndFilterSchemas import QueryAndFilters


class QueryResult(TypedDict):
    """Schema for a source query and its result."""

    id: int
    query: str | QueryAndFilters
    source: str
    result: tuple[str, Citation] | str | None