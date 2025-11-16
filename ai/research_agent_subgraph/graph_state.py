from typing import TypedDict, Annotated, List

from langchain_core.messages import BaseMessage
from langgraph.graph import add_messages

from ai.research_agent_subgraph.query_filter_schemas import QueryAndFilters


class ResearchAgentState(TypedDict):
    """
    State schema for the Research Agent subgraph.
    Tracks conversation history, queries, resources, and evaluation flags.
    """

    messages: Annotated[List[BaseMessage], add_messages]  # Message/chat history
    response: str  # Final response generated
    response_feedback: str  # Feedback on the response
    response_satisfied: bool  # If the feedback is satisfactory
    query: dict  # Query to vector db (contains 'query' and 'filters')
    query_filters: dict  # Filters for the vector db query
    queries_made: list  # List of all queries made
    resources: list  # Retrieved resources from vector db
    query_satisfied: bool  # If the query results were satisfactory
