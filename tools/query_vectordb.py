from langchain.tools import tool
from langchain_core.messages import HumanMessage
from langchain_ollama import ChatOllama
from qdrant_client import QdrantClient

from tools.embed import embed

DB_URL = "http://localhost:6333"

def rewrite_query(query: str) -> str:
    llm_rewriter = ChatOllama(model="llama3.2:3b", temperature=0.2)
    prompt = f"Rewrite as a question (example: 'covenant' -> 'What is a convenant?'). If already a question just rewrite it. Respond with NOTHING but the rewritten question. Here is what you need to rewrite: {query}"
    return llm_rewriter.generate([[HumanMessage(content=prompt)]]).generations[0][0].text
