from typing import TypedDict, Annotated, List

from langchain_core.messages import BaseMessage
from langgraph.graph import add_messages

from agent.research_agent_subgraph.query_filter_schemas import QueryAndFilters


class ResearchAgentState(TypedDict):
    """State schema for the Research Agent subgraph."""
    messages: Annotated[List[BaseMessage], add_messages]  # message/chat history
    response: str  # final response generated
    response_feedback: str  # feedback on the response
    response_satisfied: bool  # if the feedback is satisfactory
    query: dict  # query to vector db (contains 'query' and 'filters')
    query_filters: dict  # filters for the vector db query
    queries_made: int  # total number of queries made
    resources: list  # retrieved resources from vector db
    query_satisfied: bool  # if the query results were satisfactory