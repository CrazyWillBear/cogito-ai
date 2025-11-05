from agent.models.llama import llama_low_temp
from agent.research_agent_subgraph.graph import ResearchAgentState


def rewrite_query(state: ResearchAgentState):
    """Rewrite a vector DB query based on the user's message and previous research results."""
    query = state.get("query", "")
    resources = state.get("resources", [])
    prompt = (
        f"Refine and improve the semantic search query based on prior results.\n"
        f"Original query: {query}\n"
        f"Resources (snippets or titles): {resources}\n"
        "Return ONLY the refined query text."
    )
    result = llama_low_temp.invoke(prompt)
    content = getattr(result, "content", str(result)) or ""
    return {"query": content.strip()}