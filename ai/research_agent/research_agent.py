from langgraph.constants import START, END
from langgraph.graph import StateGraph

from ai.research_agent.schemas.graph_state import ResearchAgentState
from ai.research_agent.nodes.assess_summary import assess_summary
from ai.research_agent.nodes.assess_resources import assess_resources
from ai.research_agent.nodes.entry import entry
from ai.research_agent.nodes.query_vector_db import query_vector_db
from ai.research_agent.nodes.summarize import summarize
from ai.research_agent.nodes.write_queries import write_queries


class ResearchAgent:
    """Research Agent subgraph for querying vector DBs and summarizing results."""

    def __init__(self):
        """Initialize the Research Agent subgraph."""
        self.graph = self.build()

    def run(self, conversation: dict) -> str:
        """Invoke the Research Agent subgraph with a conversation."""
        res = self.graph.invoke(conversation)
        return res.get('response', 'No response available')

    @staticmethod
    def build():
        """
        Build the Research Agent subgraph.
        Constructs a state graph that queries vector databases, evaluates results,
        and generates summaries until satisfaction criteria are met.
        """

        # --- Initialize graph ---
        g = StateGraph(ResearchAgentState)

        # --- Add nodes ---
        g.add_node("entry", entry)
        g.add_node("write_queries", write_queries)
        g.add_node("query_vector_db", query_vector_db)
        g.add_node("assess_resources", assess_resources)
        g.add_node("assess_summary", assess_summary)
        g.add_node("summarize", summarize)

        # --- Add edges ---
        g.add_edge(START, "entry")
        g.add_edge("entry", "write_queries")
        g.add_edge("write_queries", "query_vector_db")
        g.add_edge("query_vector_db", "assess_resources")
        g.add_edge("summarize", "assess_summary")

        # --- Add conditional edges ---
        g.add_conditional_edges(
            "assess_resources",
            lambda state: "summarize" if state["query_satisfied"] else "write_queries"
        )
        g.add_conditional_edges(
            "assess_summary",
            lambda state: END if state["response_satisfied"] else "summarize"
        )

        return g.compile()
