import time

from langchain_core.messages import SystemMessage, HumanMessage

from ai.models.gpt import gpt_extract_content
from ai.models.model_config import MODEL_CONFIG
from ai.research_agent.schemas.ResearchAgentState import ResearchAgentState


# Max queries allowed
MAX_SOURCES = 4

def get_feedback(last_message, resources):
    """Get feedback on why the current research resources are insufficient to answer the user's query."""

    # Get configured feedback model
    feedback_model, feedback_reasoning = MODEL_CONFIG["assess_resources_feedback"]

    # Build feedback prompt (system and user message)
    feedback_system_msg = SystemMessage(content=(
        "You are an assistant that provides feedback on why the current research resources are insufficient to answer "
        "the user's query. Provide specific guidelines for further queries, not criticisms of current results. Limit "
        "your response to 100-200 tokens, in other words, be concise.\n"
    ))

    feedback_user_msg = HumanMessage(content=(
        f"Here is the user's last message:\n{last_message}\n\n"
        f"Here are the resources obtained so far:\n{resources}\n"
        "Provide guidance for further queries, what sources they may need to be from, what they should be about, etc."
    ))

    # Invoke feedback model and extract output
    feedback_result = feedback_model.invoke([feedback_system_msg, feedback_user_msg], reasoning={"effort": feedback_reasoning})
    feedback = gpt_extract_content(feedback_result)

    return feedback

def assess_resources(state: ResearchAgentState):
    """Assess whether the collected research resources are sufficient to answer the user's query."""

    # Start timing and log
    print("::Assessing resources...", end="", flush=True)
    start = time.perf_counter()

    # Extract graph state variables
    conversation = state.get("conversation", {})
    resources = state.get("resources") or "No research resources collected yet."

    # Get configured model
    classifier_model, classifier_reasoning = MODEL_CONFIG["assess_resources_classifier"]

    # Build prompt (system and user message)
    system_msg = SystemMessage(content=(
        "You are a classifier assistant that classified the research as either sufficient or insufficient to answer "
        "the user's query. Some things to consider:\n"
        "- Does the research cover the main concept(s) and argument(s) implied by the user's query?\n"
        "- If the user named specific authors or sources, they ALL MUST BE included in the resources.\n"
        "- Most of the time, one resource is sufficient, but sometimes multiple are needed for complex queries.\n"
        "- Are the resources too specific or too general to address the user's question? Are they focusing on an "
        "irrelevant or unimportant aspect of the question? Are they covering all aspects of the answer?\n\n"
        "Respond with NOTHING BUT a single word: 'YES' if the research is sufficient to answer the user's query, "
        "or 'NO' if it is not sufficient."
    ))

    last_message = conversation.get("last_user_message", "No last user message found")
    user_msg = HumanMessage(content=(
        f"Here is the user's last message:\n{last_message}\n\n"
        f"Here are research results obtained so far:\n{resources}\n"
    ))

    # Invoke model and extract output
    result = classifier_model.invoke([system_msg, user_msg], reasoning={"effort": classifier_reasoning})
    query_satisfied = "yes" in gpt_extract_content(result).strip().lower()  # yes = True, otherwise False

    # If not satisfied, get feedback on what additional research is needed
    if not query_satisfied:
        feedback = get_feedback(last_message, resources)
    else:
        feedback = ""

    # End timing and log
    end = time.perf_counter()
    print(f"\r\033[K::Resources assessed in {end - start:.2f}s")

    return {"query_satisfied": query_satisfied or len(resources) >= MAX_SOURCES, "queries_feedback": feedback}

