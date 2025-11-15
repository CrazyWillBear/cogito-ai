from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI


# Create a GPT model
gpt_low_temp = ChatOpenAI(
    model="gpt-4.1-mini",
    temperature=0.0
)