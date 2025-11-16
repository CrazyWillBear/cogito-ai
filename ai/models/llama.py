from langchain_ollama import ChatOllama


# Create a Llama model
llama_low_temp = ChatOllama(
    model="llama3.2:3b",
    temperature=0.0
)