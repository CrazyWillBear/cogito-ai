from langchain_core.messages import SystemMessage, HumanMessage

from agent.models.gpt import gpt_low_temp
from agent.research_agent_subgraph.graph_state import ResearchAgentState


def summarize(state: ResearchAgentState):
    """Summarize the most relevant sources from the research results to inform a response."""
    messages = state["messages"]
    resources = state["resources"]

    chat_history = messages + [
        SystemMessage(content="Use the research resources below to answer, citing sources."),
        HumanMessage(content=str(resources))
    ]
    summary = gpt_low_temp.invoke(chat_history)
    return { "response": summary.content }