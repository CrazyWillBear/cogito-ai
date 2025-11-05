from typing import TypedDict, List, Annotated

from langchain.agents import create_agent
from langchain_core.messages import BaseMessage
from langchain_ollama import ChatOllama
from langgraph.graph import add_messages

from tools.query_vectordb import query_vectordb

class State(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]

class Agent:
    def __init__(self):
        # This will later be replaced by a larger model
        self.model = ChatOllama(
            model="llama3.2:3b",
            temperature=0.3
        )

        self.tools = [query_vectordb]

        self.agent = create_agent(
            model=self.model,
            tools=self.tools,
            system_prompt=self.SYS_PROMPT
        )