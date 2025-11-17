from langchain_core.messages import HumanMessage, SystemMessage

from ai.research_agent.research_agent import ResearchAgent

# Sample input to test the research agent. You can make this whatever you want, I'd leave the system message as-is.
inputs = {
    "messages": [
        SystemMessage(content="You are a helpful philosophical research assistant."),
        HumanMessage(content="How does Hobbes reconcile the universal right to self-preservation in the state of nature (ch. 14) with the duty to obey the sovereign once a commonwealth is established (chs. 17–18)? Are there any explicit limits or exceptions to this duty in Leviathan?")
    ]
}

agent = ResearchAgent()
output = agent.run(inputs)
print(output)