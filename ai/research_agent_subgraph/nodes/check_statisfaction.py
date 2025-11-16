from langchain_core.messages import SystemMessage, HumanMessage

from ai.models.gpt import gpt_low_temp
from ai.research_agent_subgraph.graph_state import ResearchAgentState


def check_satisfaction(state: ResearchAgentState):
    """
    Check if the research results are sufficient to answer the user's query.
    Returns True if results are adequate or max query limit is reached; False otherwise.
    """

    # --- Build prompt ---
    system_msg = SystemMessage(content=(
        "You are a reasoning assistant that evaluates whether the provided research "
        "is sufficient to answer the user's query adequately.\n\n"
        "Think using chain-of-thought reasoning before deciding. "
        "Include your reasoning in <thinking>...</thinking> tags.\n"
        "Outside of those tags, at the very end, respond with ONLY 'Yes' if the research is sufficient "
        "to answer the user's query, or 'No' if more research is needed."
    ))

    user_msg = HumanMessage(content=(
        f"Here are the conversation messages:\n{state['messages']}\n\n"
        f"Here are the research results:\n{state.get('resources', [])}"
    ))

    # --- Invoke LLM ---
    result = gpt_low_temp.invoke([system_msg, user_msg])

    # --- Extract and clean output ---
    text = getattr(result, "content", str(result)) or ""
    if "<thinking>" in text:
        text = text.split("</thinking>")[-1].strip()

    answer = text.lower().strip()
    satisfied = ("yes" in answer) or (len(state["queries_made"]) >= 5)

    return {"query_satisfied": satisfied}
