from typing_extensions import TypedDict


class Filters(TypedDict):
    """
    Optional filters for the vector DB query.
    Supports filtering by author name and source title.
    """

    author: str | None
    source_title: str | None


class QueryAndFilters(TypedDict):
    """
    Schema for the vector DB query and filters.
    Combines a search query string with optional filter.
    """

    query: str
    filters: Filters | None
