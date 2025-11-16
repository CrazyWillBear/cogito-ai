from langgraph.constants import START, END
from langgraph.graph import StateGraph

from ai.research_agent_subgraph.graph_state import ResearchAgentState
from ai.research_agent_subgraph.nodes.assess_summary import assess_summary
from ai.research_agent_subgraph.nodes.check_statisfaction import check_satisfaction
from ai.research_agent_subgraph.nodes.entry import entry
from ai.research_agent_subgraph.nodes.query_vector_db import query_vector_db
from ai.research_agent_subgraph.nodes.summarize import summarize
from ai.research_agent_subgraph.nodes.write_query import write_query


def build_research_agent():
    """
    Build the Research Agent subgraph.
    Constructs a state graph that queries vector databases, evaluates results,
    and generates summaries until satisfaction criteria are met.
    """

    g = StateGraph(ResearchAgentState)

    # --- Add nodes ---
    g.add_node("entry", entry)
    g.add_node("write_query", write_query)
    g.add_node("query_vector_db", query_vector_db)
    g.add_node("check_satisfaction", check_satisfaction)
    g.add_node("assess_summary", assess_summary)
    g.add_node("summarize", summarize)

    # --- Add linear edges ---
    g.add_edge(START, "entry")
    g.add_edge("entry", "write_query")
    g.add_edge("write_query", "query_vector_db")
    g.add_edge("query_vector_db", "check_satisfaction")

    # --- Add conditional edges ---
    g.add_conditional_edges(
        "check_satisfaction",
        lambda state: "summarize" if state["query_satisfied"] else "write_query"
    )

    g.add_edge("summarize", "assess_summary")
    g.add_conditional_edges(
        "assess_summary",
        lambda state: END if state["response_satisfied"] else "summarize"
    )

    return g.compile()
