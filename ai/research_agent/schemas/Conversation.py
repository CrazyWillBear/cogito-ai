from pydantic import BaseModel


class Conversation(BaseModel):
    """Schema for conversation data used by the Research Agent."""

    last_user_message: str                     # The last message from the user
    context: str                              # Context from all but last user message