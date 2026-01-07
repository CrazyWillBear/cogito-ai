from langchain_core.messages import SystemMessage

from ai.models.extract_content import extract_content, safe_invoke
from ai.research_agent.model_config import RESEARCH_AGENT_MODEL_CONFIG
from ai.research_agent.schemas.ResearchAgentState import ResearchAgentState
from util.SpinnerController import SpinnerController


def classify_research_needed(state: ResearchAgentState, spinner_controller: SpinnerController = None):
    """Classify whether the user's last message would benefit from philosophical research."""

    if spinner_controller:
        spinner_controller.set_text("::Classifying research")

    # Extract graph state variables
    conversation = state.get("conversation", {})

    # Get configured model
    classifier_model, classifier_reasoning = RESEARCH_AGENT_MODEL_CONFIG["research_classifier"]

    # Build prompt (system and user message)
    system_msg = SystemMessage(content=(
        "You are a router agent that determines whether your response to the user's message would benefit from academic "
        "philosophical research. Questions that don't need further research or are unrelated to philosophy should be "
        "answered with NOTHING but 'No' (be strict with this). Respond with NOTHING but 'Yes' otherwise.\n\n"
        "Respond with 'Yes' or 'No' ONLY.\n"
    ))

    # Invoke model and extract output
    result = safe_invoke(classifier_model, [*conversation, system_msg], classifier_reasoning)
    classification = "yes" in extract_content(result).strip().lower()  # yes = True, otherwise False

    return {"research_needed": classification}
