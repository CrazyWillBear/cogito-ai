from agent.research_agent_subgraph.graph import build_recursive_retriever

inputs = {
    "messages": [
        {"role": "user", "content": "Where does the sovereign get its authority from?"}
    ]
}

agent = build_recursive_retriever()
result = agent.invoke(inputs)
print("Summary:", result.get('summary', 'No summary available'))
print("\nMessages:")
for i in result.get('messages', []):
    print(i)