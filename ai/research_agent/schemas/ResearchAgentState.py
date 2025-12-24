from typing import TypedDict

from ai.research_agent.schemas.Conversation import Conversation


class ResearchAgentState(TypedDict):
    """State schema for the Research Agent subgraph."""

    messages: list                          # Conversation messages
    conversation: Conversation              # Contains final user message + summarized context

    response: str                           # Final response generated

    vector_db_queries: list                 # Queries for vector db
    sep_queries: list                       # Queries for Stanford Encyclopedia of Philosophy
    queries_feedback: str                   # Feedback for research queries
    query_satisfied: bool                   # If the query results were satisfactory
    research_needed: bool                   # If the user question is too broad for research

    resources: list                         # Resources
