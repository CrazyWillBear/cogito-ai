from langchain_core.messages import SystemMessage

from ai.models.util import extract_content, safe_invoke
from ai.research_agent.model_config import RESEARCH_AGENT_MODEL_CONFIG
from ai.research_agent.schemas.ResearchAgentState import ResearchAgentState
from ai.research_agent.schemas.ResearchEffort import ResearchEffort
from cli.SpinnerController import SpinnerController


def classify_research_needed(state: ResearchAgentState, spinner_controller: SpinnerController = None):
    """Classify whether the user's last message would benefit from philosophical research."""

    if spinner_controller:
        spinner_controller.set_text("::Classifying research")

    # Extract graph state variables
    conversation = state.get("conversation", [])

    # Get configured model
    classifier_model, classifier_reasoning = RESEARCH_AGENT_MODEL_CONFIG["research_classifier"]

    # Build prompt (system and user message)
    system_msg = SystemMessage(content=(
        "## YOUR JOB\n"
        "You are a router agent that determines whether your response to the user's message would benefit from academic "
        "philosophical research. You will answer with NOTHING but a number (be strict with this). If the user tells "
        "you specifically how much research to perform LISTEN TO THEM.\n\n"

        "## YOUR OPTIONS\n"
        "`0` - No research needed: The agent will perform no research to respond to the user's message. Choose this if "
        "the question is irrelevant to philosophy or heavily implies/explicitly says research isn't needed.\n"
        "`1` - Simple research needed: The agent will perform basic research to gather relevant info before responding. This is "
        "for questions that require relatively simple research to answer and aren't multifaceted. Such questions include "
        "asking for definitions of terms, high level explanations of single concepts, etc.\n"
        "`2` - Deep research needed: The agent will perform deeper research to gather relevant info before responding. This "
        "is for multifaceted or otherwise medium-to-complex questions. If a question mentions multiple philosophers, theories, or topics, "
        "this is needed. Any questions that would benefit from good analysis and synthesis of evidence should be routed here.\n\n"

        "## YOUR RESPONSE\n"
        "RESPOND WITH NOTHING BUT THE NUMBER CORRESPONDING TO YOUR OPTION (0-2). DO NOT use tools, DO NOT output "
        "ANYTHING other than the number. For your reasoning, understand that simple research (1) allows the agent up "
        "to 3 research iterations, deep research (2) allows up to 10 research iterations, and no research (0) allows none.\n"
    ))

    attempts = 0
    while True:
        # Invoke model and extract output
        result = extract_content(safe_invoke(classifier_model, [*conversation[-5:], system_msg], classifier_reasoning))

        if attempts >= 3:
            # After several failed attempts, default to simple research
            return {"research_effort": ResearchEffort.SIMPLE}

        if "1" in result:
            return {"research_effort": ResearchEffort.SIMPLE}
        elif "2" in result:
            return {"research_effort": ResearchEffort.DEEP}
        elif "0" in result:
            return {"research_effort": ResearchEffort.NONE}
        else:
            attempts += 1
