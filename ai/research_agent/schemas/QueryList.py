from pydantic import BaseModel

from dbs.QueryAndFilterSchemas import QueryAndFilters


class QueryList(BaseModel):
    """Output schema for LLM query generation."""

    vector_db_queries: list[QueryAndFilters] | None
    stanford_encyclopedia_queries: list[str] | None
