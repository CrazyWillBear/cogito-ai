from typing import TypedDict

from langchain_core.messages import AnyMessage

from ai.research_agent.schemas.QueryResult import QueryResult


class ResearchAgentState(TypedDict):
    """State schema for the Research Agent subgraph."""

    conversation: list[AnyMessage]          # Conversation object

    response: str                           # Final response generated

    research_iterations: int                # Number of research iterations performed
    vector_db_queries: list                 # Queries for vector db
    sep_queries: list                       # Queries for Stanford Encyclopedia of Philosophy
    completed: bool                         # If the query results were satisfactory
    research_needed: bool                   # If the user question is too broad for research

    query_results: list[QueryResult]        # Result status per query
    all_raw_results: set                    # Raw results collected so far (to avoid duplicates)
