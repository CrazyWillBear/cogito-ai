from agent.models.llama import llama_low_temp
from agent.research_agent_subgraph.graph import ResearchAgentState


def write_query(state: ResearchAgentState):
    """Write an initial vector DB query based on the user's last message."""
    # This is the first node in the graph always, so we set queries_made to 0
    state["queries_made"] = 0
    messages = state["messages"]
    prompt = (
        "Write a short, specific semantic search query based on the user's last message. "
        "Respond with ONLY the query text.\n"
        f"Here are the messages: {messages}"
    )
    result = llama_low_temp.invoke(prompt)
    content = getattr(result, "content", str(result)) or ""
    return {"query": content.strip()}
