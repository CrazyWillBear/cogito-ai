from agent.research_agent_subgraph.graph_state import ResearchAgentState


def entry(state: ResearchAgentState):
    """Initialize all state variables on first entry."""
    return {
        "response": None,
        "response_feedback": None,
        "response_satisfied": False,
        "query": None,
        "query_filters": None,
        "queries_made": 0,
        "resources": [],
        "query_satisfied": False
    }
