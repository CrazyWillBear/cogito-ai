from agent.models.llama import llama_low_temp
from agent.research_agent_subgraph.graph import ResearchAgentState


def check_satisfaction(state: ResearchAgentState):
    """Check if the research results are sufficient to answer the user's query."""
    prompt = (
        "Are these research results sufficient to answer the user's query adequately if summarized?"
        "Respond with ONLY 'Yes' or 'No'.\n"
        f"Here are the messages: {state['messages']}\n"
        f"Here are the research results: {state.get('resources', [])}"
    )
    result = llama_low_temp.invoke(prompt)
    content = getattr(result, "content", str(result)) or ""
    satisfied = ("yes" in content.lower().strip()) or (state["queries_made"] >= 5)
    return {"satisfied": satisfied}
