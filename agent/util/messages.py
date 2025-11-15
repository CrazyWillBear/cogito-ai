from langchain_core.messages import HumanMessage


def get_last_user_message(messages):
    """Retrieve the content of the last HumanMessage from a list of messages."""
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage):
            return msg.content
    return None  # if no user message found
