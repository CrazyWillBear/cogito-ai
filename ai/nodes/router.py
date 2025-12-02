"""
This file is NOT BEING USED YET. This will be used later on to route messages to either a research or normal chat agent.
"""
from ai.models.llama import llama_low_temp


def router(state):
    """Routes messages to research or chat based on whether the last message requires research."""
    # --- Extract state variables ---
    messages = state['messages']

    # --- Build and invoke prompt ---
    prompt = f"You are a router that determines whether the last message sent by the user/human would benefit from \
              research to inform the response. Respond with ONLY 'Yes' (if it requires research) or 'No' (if it \
              doesn't). Here are the messages: {messages}"
    res = llama_low_temp.invoke(prompt)

    # --- Determine route ---
    if 'yes' in res.lower():
        return 'research'
    else:
        return 'chat'