from typing import TypedDict


class Citation(TypedDict):
    """Schema for a citation of a source used in research."""

    title: str
    authors: list[str]
    section: str
    source: str