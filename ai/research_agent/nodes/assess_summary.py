from langchain_core.messages import SystemMessage, HumanMessage

from ai.research_agent.model_config import MODEL_CONFIG
from ai.research_agent.schemas.graph_state import ResearchAgentState


def assess_summary(state: ResearchAgentState):
    """
    Evaluate whether the response is accurate, faithful, and free from hallucinations or false citations.
    Returns a boolean flag: True if the response is good; False otherwise.
    """
    # --- Get model ---
    model = MODEL_CONFIG["assess_summary"]

    # --- Extract state variables ---
    resources = state.get("resources", [])
    summary = state.get("response", "")

    # --- Build prompts ---
    system_msg = SystemMessage(content=(
        "You are a critical reasoning assistant. Your job is to determine whether the provided response is accurate, "
        "faithful to the research results, and free of hallucinations or false citations.\n\n"
        "Reason step by step in <thinking>...</thinking> tags. Consider:\n"
        "- Are all claims supported by the research?\n"
        "- Are quotes accurate (not fabricated)?\n"
        "- Is the response misleading in any way?\n\n"
        "After reasoning, output ONLY 'Yes' (accurate and faithful) or 'No' (contains errors/hallucinations)."
    ))

    user_msg = HumanMessage(content=(
        f"Response to assess:\n---\n{summary}\n---\n\n"
        f"Research resources:\n---\n{resources}\n---"
    ))
    # --- Invoke LLM ---
    result = model.invoke([system_msg, user_msg])

    # --- Extract and clean output ---
    text = getattr(result, "content", str(result)) or ""
    if "<thinking>" in text:
        text = text.split("</thinking>")[-1].strip()

    decision = text.lower().strip()
    is_good_summary = decision.startswith("yes")

    # --- Update state ---
    return {"response_satisfied": is_good_summary}