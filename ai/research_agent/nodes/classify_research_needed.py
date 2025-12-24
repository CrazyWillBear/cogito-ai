import time

from langchain_core.messages import SystemMessage, HumanMessage

from ai.models.gpt import gpt_extract_content
from ai.models.model_config import MODEL_CONFIG
from ai.research_agent.schemas.ResearchAgentState import ResearchAgentState


def classify_research_needed(state: ResearchAgentState):
    """Classify whether the user's last message would benefit from philosophical research."""

    # Start timing and log
    print("::Classifying research...", end="", flush=True)
    start = time.perf_counter()

    # Extract graph state variables
    conversation = state.get("conversation", {})

    # Get configured model
    classifier_model, classifier_reasoning = MODEL_CONFIG["research_classifier"]

    # Build prompt (system and user message)
    system_msg = SystemMessage(content=(
        "You are a router agent that determines whether the question being asked would benefit from philosophical "
        "research. Questions that don't need further research or are unrelated to philosophy, should be answered "
        "with NOTHING but 'No' (be strict with this). Respond with NOTHING but 'Yes' otherwise.\n\n"
        "Respond with 'Yes' or 'No' ONLY. Most questions require research."
    ))

    conversation_context = conversation.get("context", "No previous conversation context")
    last_message = conversation.get("last_user_message", "No last user message found")
    user_msg = HumanMessage(content=(
        f"Here is the conversation context:\n{conversation_context}\n\n"
        f"Here is the user's last message:\n{last_message}\n\n"
    ))

    # Invoke model and extract output
    result = classifier_model.invoke([system_msg, user_msg], reasoning={"effort": classifier_reasoning})
    classification = "yes" in gpt_extract_content(result).strip().lower()  # yes = True, otherwise False

    # End timing and log
    end = time.perf_counter()
    print(f"\r\033[K::Research classified in {end - start:.2f}s")

    return {"research_needed": classification}
