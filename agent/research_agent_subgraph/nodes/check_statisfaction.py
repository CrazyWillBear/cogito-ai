from langchain_core.messages import SystemMessage, HumanMessage

from agent.models.gpt import gpt_low_temp
from agent.research_agent_subgraph.graph_state import ResearchAgentState


def check_satisfaction(state: ResearchAgentState):
    """Check if the research results are sufficient to answer the user's query."""

    # Build structured prompt
    system_msg = SystemMessage(content=(
        "You are a reasoning assistant that evaluates whether the provided research "
        "is sufficient to answer the user's query adequately."
        " Think using chain-of-thought reasoning before deciding."
        " Include your reasoning in <thinking>...</thinking> tags."
        " Outside of those tags, at the very end, respond with ONLY 'Yes' if the research is sufficient "
        "to answer the user's query, or 'No' if more research is needed."
    ))

    user_msg = HumanMessage(content=(
        f"Here are the conversation messages:\n{state['messages']}\n\n"
        f"Here are the research results:\n{state.get('resources', [])}"
    ))

    # Invoke model with reasoning allowed
    result = gpt_low_temp.invoke([system_msg, user_msg])

    text = getattr(result, "content", str(result)) or ""

    # Strip out reasoning if present
    if "<thinking>" in text:
        text = text.split("</thinking>")[-1].strip()

    # Normalize output
    answer = text.lower().strip()
    satisfied = ("yes" in answer) or (state["queries_made"] >= 5)

    return {"query_satisfied": satisfied}
