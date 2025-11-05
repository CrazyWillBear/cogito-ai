# python
import json
from typing import List, Dict, Any, Union
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage


class ChatHistory:
    def __init__(self):
        self.messages: List[Dict[str, Any]] = []
        self.history: List[Any] = []

    def _add_message(self, role: str, content: str) -> None:
        self.messages.append({"role": role, "content": content})

    def from_json(self, json_data: Union[str, List[Dict[str, Any]]]) -> None:
        """
        Accepts either:
          - a Python list of message dicts, or
          - a JSON string representing that list.
        Each message must be a dict with 'role' and 'content'.
        """
        if isinstance(json_data, str):
            json_data = json.loads(json_data)

        if not isinstance(json_data, list):
            raise ValueError("json_data must be a list of message objects")

        for i, msg in enumerate(json_data):
            if not isinstance(msg, dict) or "role" not in msg or "content" not in msg:
                raise ValueError(f"message at index {i} must be an object with 'role' and 'content'")
            role = msg["role"]
            content = "" if msg["content"] is None else str(msg["content"])
            self._add_message(role, content)

    def get_history_langchain(self):
        """
        Convert stored messages to LangChain message objects.
        Unknown roles are skipped.
        """
        mapping = {
            "user": HumanMessage,
            "human": HumanMessage,
            "assistant": AIMessage,
            "ai": AIMessage,
            "system": SystemMessage,
            "bot": AIMessage,
        }
        history = []
        for msg in self.messages:
            cls = mapping.get(msg["role"])
            if cls is None:
                continue
            history.append(cls(content=msg["content"]))
        return history
