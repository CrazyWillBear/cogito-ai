from typing import TypedDict, Annotated, List

from langchain_core.messages import BaseMessage
from langgraph.graph import add_messages


class ResearchAgentState(TypedDict):
    """
    State schema for the Research Agent subgraph.
    Tracks conversation history, queries, resources, and evaluation flags.
    """

    messages: Annotated[List[BaseMessage], add_messages]    # Message/chat history

    response: str                                           # Final response generated
    response_feedback: str                                  # Feedback on the response
    response_satisfied: bool                                # If the feedback is satisfactory

    queries: list                                           # Queries for vector db (contains list of 'query' and 'filters' pairs)
    queries_made: list                                      # List of all queries made
    query_satisfied: bool                                   # If the query results were satisfactory

    resources: list                                         # Retrieved resources from vector db
