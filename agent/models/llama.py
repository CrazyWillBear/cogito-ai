from langchain_ollama import ChatOllama

llama_low_temp = ChatOllama(
    model="llama3.2:3b",
    temperature=0.0
)