from pydantic import BaseModel


class Filters(BaseModel):
    """
    Optional filters for the vector DB query.
    Supports filtering by author name and source title.
    """

    author: str | None = None
    source_title: str | None = None


class QueryAndFilters(BaseModel):
    """
    Schema for the vector DB query and filters.
    Combines a search query string with optional filter criteria.
    """

    query: str
    filters: Filters