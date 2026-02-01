from langchain_ollama import ChatOllama

gemma3_4b = ChatOllama(
    model="gemma3:4b",
    temperature=0.3
)