from typing import Callable

from langgraph.constants import START, END
from langgraph.graph import StateGraph

from ai.research_agent.nodes.assess_resources import assess_resources
from ai.research_agent.nodes.classify_research_needed import classify_research_needed
from ai.research_agent.nodes.create_conversation import create_conversation
from ai.research_agent.nodes.query_sources import query_sources
from ai.research_agent.nodes.write_queries import write_queries
from ai.research_agent.nodes.write_response import write_response
from ai.research_agent.schemas.ResearchAgentState import ResearchAgentState
from dbs.Postgres import Postgres
from dbs.Qdrant import Qdrant


class ResearchAgent:
    """Research Agent subgraph for querying vector DBs and summarizing results."""

    # --- Methods ---
    def __init__(self, qdrant = None, postgres_filters = None):
        """Initialize the Research Agent subgraph."""

        self.graph = None
        self.qdrant = qdrant if qdrant is not None else Qdrant()
        self.postgres_filters = postgres_filters if postgres_filters is not None else Postgres()
        self.resources = []

    def run(self, conversation: dict) -> str:
        """Invoke the Research Agent subgraph with a conversation."""

        res = self.graph.invoke(conversation)
        return res.get('response', 'No response available')

    def build(self) -> None:
        """
        Build the Research Agent subgraph.
        Constructs a state graph that queries vector databases, evaluates results,
        and generates summaries until satisfaction criteria are met.
        """

        # --- Initialize graph ---
        g = StateGraph(ResearchAgentState)

        # --- Add agent_assigner ---
        g.add_node("create_conversation", create_conversation)
        g.add_node("classify_research_needed", classify_research_needed)
        g.add_node("write_queries", write_queries)
        g.add_node("query_sources", self._wrap(query_sources, self.qdrant))
        g.add_node("assess_resources", assess_resources)
        g.add_node("write_response", write_response)

        # --- Add edges ---
        g.add_edge(START, "create_conversation")
        g.add_edge("create_conversation", "classify_research_needed")
        g.add_edge("write_queries", "query_sources")
        g.add_edge("query_sources", "assess_resources")
        g.add_edge("write_response", END)

        # --- Add conditional edges ---
        g.add_conditional_edges(
            "assess_resources",
            lambda state: "write_response" if state["query_satisfied"] else "write_queries"
        )
        g.add_conditional_edges(
            "classify_research_needed",
            lambda state: "write_queries" if state["research_needed"] else "write_response"
        )

        self.graph = g.compile()

    def close(self):
        """Close any database connections used by the Research Agent."""

        self.qdrant.close()
        self.postgres_filters.close()

    @staticmethod
    def _wrap(func: Callable, *args, **kwargs) -> Callable:
        """Wrap a node so it receives `state` plus any extra args/kwargs."""

        def wrapped(state):
            return func(state, *args, **kwargs)

        return wrapped
