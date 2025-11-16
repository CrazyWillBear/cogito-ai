from ai.research_agent_subgraph.graph_state import ResearchAgentState


def entry(state: ResearchAgentState):
    """
    Initialize all state variables on first entry.
    Sets default values for tracking research progress and results.
    """

    return {
        "response": None,
        "response_feedback": None,
        "response_satisfied": False,
        "query": None,
        "query_filters": None,
        "queries_made":[],
        "resources": [],
        "query_satisfied": False
    }
