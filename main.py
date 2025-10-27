from typing import TypedDict
from langchain_core.messages import BaseMessage


class GraphState(TypedDict):
    output_txt:         str
    chat_history:       list[BaseMessage]
    research_needed:    bool
