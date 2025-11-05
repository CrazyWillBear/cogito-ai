from langchain_openai import ChatOpenAI

# Create a GPT model (e.g. GPT-4o)
gpt_low_temp = ChatOpenAI(
    model="gpt-4o",
    temperature=0.2
)