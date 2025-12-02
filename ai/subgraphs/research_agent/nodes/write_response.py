import time

from langchain_core.messages import SystemMessage, HumanMessage

from ai.models.gpt import gpt_extract_content
from ai.subgraphs.research_agent.model_config import MODEL_CONFIG
from ai.subgraphs.research_agent.schemas.graph_state import ResearchAgentState


def write_response(state: ResearchAgentState):
    """Compose the assistant's final answer by synthesizing conversation context and gathered research, using quoted
    evidence and formatted citations."""

    # Start timing and log
    print("::Reasoning through and writing final response...", end="", flush=True)
    start = time.perf_counter()

    # Get configured model
    model = MODEL_CONFIG["write_response"]

    # Extract graph state variables
    resources = state.get("resources", "No research resources collected yet.")
    conversation = state.get("conversation", {})
    conv_summary = conversation.get("summarized_context", "No prior context needed.")
    last_message = conversation.get("last_user_message", "No last user message found")

    # Construct prompt (system message and user message)
    system_msg = SystemMessage(content=(
        "Respond to the user's last message given the following resources that you've 'researched'. Use specific cited "
        "quotes, respond in a conversational yet academic tone, and cite all sources using the `citation` given. "
        "You don't need to copy the citation exactly, use what seems right, but don't use information not given and "
        "also make sure to include the full range of section(s)/chapter(s)/etc., the author, and the source in your "
        "citations. Respond directly to the message in a tightly-organized manner.\n\n"
        "Consider:\n"
        "- What is the user's main question or message?\n"
        "- What parts of the resources are most relevant to the message/question?\n"
        "- What is the answer and how do those parts support it?\n"
        "- How can I best structure my response to be compact and direct yet with specific evidence?\n"
        f"Here is a summary of the conversation previous to the user's message:\n{conv_summary}\n\n"
        f"Here are the research resources you've gathered so far:\n{resources}\n"
    ))

    user_msg = HumanMessage(content=last_message)

    # Invoke LLM and extract output
    result = model.invoke([system_msg, user_msg], reasoning={"effort": "low"})
    text = gpt_extract_content(result)  # Extract main response text

    # End timing and log
    end = time.perf_counter()
    print(f"\r\033[K::Reasoned about and wrote final response in {end - start:.2f}s")

    return {"response": text}
