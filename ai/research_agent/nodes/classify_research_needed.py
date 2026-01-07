from langchain_core.messages import SystemMessage

from ai.models.util import extract_content, safe_invoke
from ai.research_agent.model_config import RESEARCH_AGENT_MODEL_CONFIG
from ai.research_agent.schemas.ResearchAgentState import ResearchAgentState
from ai.research_agent.schemas.ResearchEffort import ResearchEffort
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
        "## YOUR ROLE\n"
        "You are a router agent that determines whether your response to the user's message would benefit from academic "
        "philosophical research. You will answer with NOTHING but a number (be strict with this). These are your options:\n"
        "0. No research needed: The agent will perform no research and respond to the user's message. Choose this if "
        "the question is irrelevant to philosophy or implies research isn't needed.\n"
        "1. Research needed: The agent will perform research to gather relevant info before responding. These are "
        "questions that require relatively simple research to answer and aren't multifaceted. Such questions include "
        "asking for definitions of terms, high level explanations of concepts, etc.\n"
        "2. Deep research needed: The agent will perform deep research to gather relevant info before responding. This "
        "is for multifaceted or otherwise complex questions. If a question mentions multiple philosophers, theories, or topics, "
        "this is likely needed. Any questions that would benefit from reasoning should be routed here as well.\n\n"

        "## INSTRUCTIONS\n"
        "RESPOND WITH NOTHING BUT THE NUMBER. Do not be afraid to choose '2' for questions you're on the fence about. "
        "This option is barely slower than '1' and ensures a more comprehensive answer.\n"
    ))

    # Invoke model and extract output
    result = extract_content(safe_invoke(classifier_model, [*conversation, system_msg], classifier_reasoning))

    if "1" in result:
        return {"research_effort": ResearchEffort.SIMPLE}
    elif "2" in result:
        return {"research_effort": ResearchEffort.DEEP}
    else:
        return {"research_effort": ResearchEffort.NONE}
