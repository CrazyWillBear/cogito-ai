from typing import Callable

from langchain_core.messages import AnyMessage
from langgraph.constants import START, END
from langgraph.graph import StateGraph
from rich.status import Status

from ai.research_agent.nodes.classify_research_needed import classify_research_needed
from ai.research_agent.nodes.create_conversation import create_conversation
from ai.research_agent.nodes.execute_queries import execute_queries
from ai.research_agent.nodes.plan_research import plan_research
from ai.research_agent.nodes.write_response import write_response
from ai.research_agent.schemas.ResearchAgentState import ResearchAgentState
from ai.research_agent.schemas.ResearchEffort import ResearchEffort
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
        self.status = None

    def run(self, conversation: list[AnyMessage], status: Status | None) -> ResearchAgentState:
        """Invoke the Research Agent subgraph with a conversation."""

        init_state = {"conversation": conversation}
        self.status = status
        res = self.graph.invoke(init_state)
        return res

    def build(self) -> None:
        """
        Build the Research Agent subgraph.
        Constructs a state graph that queries vector databases, evaluates results,
        and generates summaries until satisfaction criteria are met.
        """

        # --- Initialize graph ---
        g = StateGraph(ResearchAgentState)

        # --- Add agent_assigner ---
        g.add_node(
            "create_conversation", self._wrap(create_conversation)
        )
        g.add_node(
            "classify_research_needed", self._wrap(classify_research_needed)
        )
        g.add_node(
            "plan_research", self._wrap(plan_research)
        )
        g.add_node(
            "execute_queries", self._wrap(execute_queries, self.qdrant)
        )
        g.add_node(
            "write_response", self._wrap(write_response)
        )

        # --- Add edges ---
        g.add_edge(START, "create_conversation")
        g.add_edge("create_conversation", "classify_research_needed")
        g.add_edge("execute_queries", "plan_research")
        g.add_edge("write_response", END)

        # --- Add conditional edges ---
        g.add_conditional_edges(
            "plan_research",
            lambda state: "write_response" if state["completed"] else "execute_queries"
        )
        g.add_conditional_edges(
            "classify_research_needed",
            lambda state: "plan_research" if state["research_effort"] is not ResearchEffort.NONE else "write_response"
        )

        self.graph = g.compile()

    def close(self):
        """Close any database connections used by the Research Agent."""

        self.qdrant.close()
        self.postgres_filters.close()

    def _wrap(self, func: Callable, *args, **kwargs) -> Callable:
        """Wrap a node so it receives `state` plus any extra args/kwargs.

        This is an instance method (not static) so the returned wrapper can
        read the current value of `self.status` at invocation time. That
        allows callers to call `agent.build()` before `agent.run(status=...)`
        and still have the nodes receive the Status object passed to run().
        """

        def wrapped(state):
            return func(state, *args, status=self.status, **kwargs)

        return wrapped
