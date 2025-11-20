from typing import TypedDict


class Conversation(TypedDict):
    """Schema for conversation data used by the Research Agent."""

    last_user_message: str                     # The last message from the user
    summarized_context: str                    # Summarized context from all but last user message