from typing import TypedDict, Annotated, List

from langchain_core.messages import BaseMessage
from langgraph.constants import START, END
from langgraph.graph import add_messages, StateGraph


class ResearchAgentState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]
    summary: str
    query: str
    queries_made: int
    resources: list
    satisfied: bool

# Import nodes after state is defined to avoid circular import issues
from agent.research_agent_subgraph.nodes.query_vector_db import query_vector_db
from agent.research_agent_subgraph.nodes.check_statisfaction import check_satisfaction
from agent.research_agent_subgraph.nodes.rewrite_query import rewrite_query
from agent.research_agent_subgraph.nodes.summarize import summarize
from agent.research_agent_subgraph.nodes.write_query import write_query


def build_recursive_retriever():
    g = StateGraph(ResearchAgentState)

    g.add_node("write_query", write_query)
    g.add_node("query_vector_db", query_vector_db)
    g.add_node("check_satisfaction", check_satisfaction)
    g.add_node("refine_query", rewrite_query)
    g.add_node("summarize", summarize)

    g.add_edge(START, "write_query")
    g.add_edge("write_query", "query_vector_db")
    g.add_edge("query_vector_db", "check_satisfaction")

    # Conditional branch:
    g.add_conditional_edges(
        "check_satisfaction",
        lambda state: "summarize" if state["satisfied"] else "refine_query",
        {"summarize": "summarize", "refine_query": "refine_query"}
    )

    g.add_edge("refine_query", "query_vector_db")
    g.add_edge("summarize", END)

    return g.compile()