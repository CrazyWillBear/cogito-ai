import json
from pathlib import Path
from typing import TypedDict

import questionary
from langchain_core.messages import AnyMessage, HumanMessage, AIMessage, SystemMessage
from rich.console import Console

CONVERSATIONS_DIR = Path.home() / Path(".cogito/conversations")
CONVERSATIONS_DIR.mkdir(parents=True, exist_ok=True)

class Conversation(TypedDict):
    id: int             # unique ID for conversation
    name: str           # name given by user or LLM
    conversation: list  # conversation dict


def get_conversations() -> list[Conversation]:
    """Retrieve saved conversation logs from disk."""

    conversations = []
    for file in CONVERSATIONS_DIR.glob("conversation-*.json"):
        conversation_str = file.read_text(encoding="utf-8")
        conversation: Conversation = json.loads(conversation_str)
        conversation["conversation"] = _messages_dict_to_messages(conversation["conversation"])
        conversations.append(conversation)

    return conversations

def get_conversation_by_id(conversation_id: int) -> Conversation | None:
    """Load a conversation by ID."""

    conversations = get_conversations()
    for conv in conversations:
        if conv["id"] == conversation_id:
            return conv
    return None

def get_new_conversation_id() -> int:
    """Generate a new unique conversation ID."""

    conversations: list[Conversation] = get_conversations()
    if not conversations:
        return 1
    else:
        max_id = max(c.get("id") for c in conversations)
        return max_id + 1

def save_conversation(conversation: list[dict], conversation_id: int, conversation_name: str):
    """Write conversation logs to disk."""

    p = CONVERSATIONS_DIR / Path(f"conversation-{conversation_id}.json")
    p.parent.mkdir(parents=True, exist_ok=True)

    conversation_dict: Conversation = {
        "id": conversation_id,
        "name": conversation_name,
        "conversation": conversation
    }
    conversation_dict_str = json.dumps(conversation_dict)

    p.write_text(conversation_dict_str, encoding="utf-8")

def user_select_conversation(console: Console) -> Conversation | None:
    """Load a conversation by ID."""

    conversations = get_conversations()

    # Use Questionary for the interaction
    console.print("Select a conversation to resume or start a new one:", style="bold gold3")
    answer = questionary.select(
        "",
        choices=["New conversation", *(f"{c['name']} - {c['id']}" for c in conversations)],
        style=questionary.Style([
            ('qmark', 'fg:cyan bold'),
            ('question', 'bold fg:gold'),
            ('pointer', 'fg:cyan bold'),
            ('highlighted', 'fg:cyan'),
            ('selected', 'fg:green'),
        ])
    ).ask()

    if answer is None or answer == "New conversation":
        return None

    # Extract the conversation ID from the selected answer
    selected_id = int(answer.split(" - ")[-1])

    # Find and return the selected conversation
    for conv in conversations:
        if conv["id"] == selected_id:
            return conv
    return None

def _messages_dict_to_messages(messages_dict: list[dict]) -> list[AnyMessage]:
    """Convert a list of message dicts to LangChain message objects."""

    messages_res = []
    for msg_dict in messages_dict:
        if msg_dict["type"] == "human":
            messages_res.append(HumanMessage(msg_dict.get("data").get("content")))
        elif msg_dict["type"] == "ai":
            messages_res.append(AIMessage(msg_dict.get("data").get("content")))
        elif msg_dict["type"] == "system":
            messages_res.append(SystemMessage(msg_dict.get("data").get("content")))
    return messages_res
