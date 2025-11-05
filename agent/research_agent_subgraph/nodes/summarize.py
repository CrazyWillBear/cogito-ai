from agent.models.llama import llama_low_temp
from agent.research_agent_subgraph.graph import ResearchAgentState


def summarize(state: ResearchAgentState):
    """Summarize the most relevant sources from the research results to inform a response."""
    messages = state["messages"]
    resources = state["resources"]
    prompt = f"Based on the user's last message, please list the sources that are most relevant and helpful to inform \
               a response. You should list them as quotes and cite your source. Here are the messages: {messages}\n \
               Here are the research results: {resources}"

    summary = llama_low_temp.invoke(prompt)
    return { "summary": summary.content }