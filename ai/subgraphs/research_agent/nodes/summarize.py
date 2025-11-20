import time

from langchain_core.messages import SystemMessage, HumanMessage

from ai.models.gpt import gpt_extract_content
from ai.subgraphs.research_agent.model_config import MODEL_CONFIG
from ai.subgraphs.research_agent.schemas.graph_state import ResearchAgentState


def summarize(state: ResearchAgentState):
    """
    Summarize the most relevant sources from the research results to inform a response.
    Combines conversation history with research resources to generate a cited answer.
    """
    print("::Reasoning through and writing final response...", end="", flush=True)
    start = time.perf_counter()

    # --- Get model ---
    model = MODEL_CONFIG["summarize"]

    # --- Extract state variables ---
    conversation = state.get("conversation", {})
    resource_summaries = state.get("resource_summaries", "No research resources collected yet.")
    conv_summary = conversation.get("summarized_context", "No prior context needed.")
    last_message = conversation.get("last_user_message", "No last user message found")

    # --- Define prompt ---
    system_msg = SystemMessage(content=(
        "You must now respond to the user's last message given the following resources that you've "
         "'researched'. Use specific quotes, respond in a conversational yet academic tone, and cite all sources at the "
         "end using this format: '(author last, author first; title)'\n\n"
         "Consider:\n"
         "- What resources are most relevant to the question?\n"
         "- Based on those resources, what's the answer?\n"
         "- How do the resources support that answer?\n\n"
        f"Here is a summary of the conversation previous to the user's message:\n{conv_summary}\n\n"
        f"Here are summaries of the research resources you've gathered so far:\n{resource_summaries}\n"
         "After reasoning, output your final response."
    ))

    user_msg = HumanMessage(content=last_message)

    # --- Invoke LLM ---
    result = model.invoke([system_msg, user_msg], reasoning={"effort": "low"})

    # --- Extract and clean output ---
    text = gpt_extract_content(result)  # Extract main response text

    end = time.perf_counter()
    print(f"\r\033[K::Reasoned about and wrote final response in {end - start:.2f}s")

    # --- Update state ---
    return {"response": text}
