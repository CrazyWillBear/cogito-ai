from langchain_core.messages import SystemMessage, HumanMessage

from ai.models.gpt import gpt_low_temp
from ai.research_agent_subgraph.graph_state import ResearchAgentState


def summarize(state: ResearchAgentState):
    """
    Summarize the most relevant sources from the research results to inform a response.
    Combines conversation history with research resources to generate a cited answer.
    """

    messages = state["messages"]
    resources = state["resources"]

    # --- Build prompt with chat history and resources ---
    chat_history = messages + [
        SystemMessage(content="Use the research resources below to answer, citing sources."),
        HumanMessage(content=str(resources))
    ]

    # --- Invoke LLM ---
    summary = gpt_low_temp.invoke(chat_history)

    return {"response": summary.content}
