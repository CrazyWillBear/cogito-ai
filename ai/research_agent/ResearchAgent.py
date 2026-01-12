from typing import Callable

from langchain_core.messages import AnyMessage
from langgraph.constants import START, END
from langgraph.graph import StateGraph

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
    def __init__(self, qdrant = None, postgres_filters = None, spinner_controller = None):
        """Initialize the Research Agent subgraph."""

        self.graph = None
        self.qdrant = qdrant if qdrant is not None else Qdrant()
        self.postgres_filters = postgres_filters if postgres_filters is not None else Postgres()
        self.spinner_controller = spinner_controller

    def run(self, conversation: list[AnyMessage]) -> ResearchAgentState:
        """Invoke the Research Agent subgraph with a conversation."""

        init_state = {"conversation": conversation}
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
            "create_conversation",
            create_conversation if self.spinner_controller is None
            else self._wrap(create_conversation, self.spinner_controller)
        )
        g.add_node(
            "classify_research_needed",
            classify_research_needed if self.spinner_controller is None
            else self._wrap(classify_research_needed, self.spinner_controller)
        )
        g.add_node(
            "plan_research",
            plan_research if self.spinner_controller is None
            else self._wrap(plan_research, self.spinner_controller)
        )
        g.add_node(
            "execute_queries",
            self._wrap(execute_queries, self.qdrant) if self.spinner_controller is None
            else self._wrap(execute_queries, self.qdrant, self.spinner_controller)
        )
        g.add_node(
            "write_response",
            write_response if self.spinner_controller is None
            else self._wrap(write_response, self.spinner_controller)
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

    @staticmethod
    def _wrap(func: Callable, *args, **kwargs) -> Callable:
        """Wrap a node so it receives `state` plus any extra args/kwargs."""

        def wrapped(state):
            return func(state, *args, **kwargs)

        return wrapped
