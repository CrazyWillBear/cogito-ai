from ai.models.llama import llama_low_temp


def router(state):
    messages = state['messages']
    prompt = f"You are a router that determines whether the last message sent by the user/human would benefit from \
              research to inform the response. Respond with ONLY 'Yes' (if it requires research) or 'No' (if it \
              doesn't). Here are the messages: {messages}"
    res = llama_low_temp.invoke(prompt)

    if 'yes' in res.lower():
        return 'research'
    else:
        return 'chat'