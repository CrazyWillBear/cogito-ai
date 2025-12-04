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
        "Respond to the user’s latest message using only the provided research resources.\n"
        "Write in a clear, conversational academic tone.\n"
        "Use specific quoted evidence with citations at each quote containing:\n"
        "- author\n"
        "- source title\n"
        "- section/chapter range provided in the resource\n"
        "...and NOT containing §pgepubid tags or any other kinds of §id tags.\n"
        "Cite ALL quotes used in the response.\n"
        "In the following format: (\"quoted text\" (Author, Source Title, Sections/Chapter/etc. X-Y)).\n"
        
        "Guidelines:\n"
        "- Answer the user’s question directly.\n"
        "- Select only the most relevant parts of the resources.\n"
        "- Organize the response tightly (clean structure, minimal fluff).\n"
        "- Do not use information outside the resources.\n"
        
        "Context summary:\n"
        f"{conv_summary}\n"
        
        "Resources:"
        f"{resources}"
    ))

    user_msg = HumanMessage(content=last_message)

    # Invoke LLM and extract output
    result = model.invoke([system_msg, user_msg], reasoning={"effort": "low"})
    text = gpt_extract_content(result)  # Extract main response text

    # End timing and log
    end = time.perf_counter()
    print(f"\r\033[K::Reasoned about and wrote final response in {end - start:.2f}s")

    return {"response": text}
