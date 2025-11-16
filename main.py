from langchain_core.messages import HumanMessage, SystemMessage

from ai.research_agent_subgraph.graph import build_research_agent


# Sample input to test the research agent. You can make this whatever you want, I'd leave the system message as-is.
inputs = {
    "messages": [
        SystemMessage(content="You are a helpful research assistant."),
        HumanMessage(content="What is the value of the common man according to Hobbes?")
    ]
}

agent = build_research_agent()
result = agent.invoke(inputs)
print(result.get('response', 'No response available'))