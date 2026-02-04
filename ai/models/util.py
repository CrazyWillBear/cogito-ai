def extract_content(result):
    """Extract the main text content from a model.invoke() result, ignoring any 'reasoning' or auxiliary objects."""

    content_list = getattr(result, "content", result)

    # If it's already a string, return it
    if isinstance(content_list, str):
        return content_list.strip()

    # If it's a list of messages, find the first text message
    if isinstance(content_list, list):
        for msg in content_list:
            if isinstance(msg, dict) and msg.get("type") == "text":
                return msg.get("text", "").strip()

    # Fallback: convert to string
    return str(result).strip()

def safe_invoke(model, messages):
    """Invoke a model with optional reasoning parameters, handling models that may not support reasoning.

    Also ensures no tool calls are made by unbinding any tools from the model.
    """

    model = model.bind_tools([], tool_choice="none")
    result = model.invoke(messages)
    return result
