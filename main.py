from langchain_core.messages import HumanMessage, SystemMessage

from agent.research_agent_subgraph.graph import build_research_agent

inputs = {
    "messages": [
        SystemMessage(content="You are a helpful research assistant."),
        HumanMessage(content="What makes an act moral according to Hobbes?")
    ]
}

agent = build_research_agent()
result = agent.invoke(inputs)
print(result.get('response', 'No response available'))