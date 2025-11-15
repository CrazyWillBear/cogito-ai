from langchain_core.messages import SystemMessage, HumanMessage

from agent.models.gpt import gpt_low_temp
from agent.research_agent_subgraph.graph_state import ResearchAgentState


def assess_summary(state: ResearchAgentState):
    """
    Evaluate whether the response is accurate, faithful, and free from hallucinations or false citations.
    Returns a boolean flag: True if the response is good; False otherwise.
    """

    resources = state.get("resources", [])
    summary = state.get("response", "")

    # --- Build prompt ---
    system_msg = SystemMessage(content=(
        "You are a critical reasoning assistant. "
        "Your job is to determine whether the provided response is accurate, "
        "faithful to the research results, and free of hallucinations or false quotes.\n\n"
        "Reason step by step before deciding. Include your reasoning inside <thinking>...</thinking> tags.\n"
        "At the very end, output ONLY one of: 'Yes' (if the response is accurate and faithful) or 'No' (if it contains hallucinations, errors, or misleading content). It does not need to be perfect, merely free of significant issues."
    ))

    user_msg = HumanMessage(content=(
        f"Here is the response to assess:\n---\n{summary}\n---\n\n"
        f"Here are the research resources:\n---\n{resources}\n---"
    ))

    # --- Invoke LLM ---
    result = gpt_low_temp.invoke([system_msg, user_msg])

    # --- Extract and clean output ---
    text = getattr(result, "content", str(result)) or ""
    if "<thinking>" in text:
        text = text.split("</thinking>")[-1].strip()

    decision = text.lower().strip()
    is_good_summary = "yes" in decision

    return {"response_satisfied": is_good_summary}