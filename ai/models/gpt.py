from langchain_openai import ChatOpenAI


# GPT 5 low temperature model
gpt5 = ChatOpenAI(
    model="gpt-5",
    temperature=0.0
)

# GPT 5 mini low temperature model
gpt5_mini = ChatOpenAI(
    model="gpt-5-mini",
    temperature=0.0
)

# GPT 5 nano low temperature model
gpt5_nano = ChatOpenAI(
    model="gpt-5-nano",
    temperature=0.0
)
