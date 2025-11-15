from pydantic import BaseModel


class Filters(BaseModel):
    """Optional filters for the vector DB query."""
    author: str | None = None
    source_title: str | None = None

class QueryAndFilters(BaseModel):
    """Schema for the vector DB query and filters."""
    query: str
    filters: Filters