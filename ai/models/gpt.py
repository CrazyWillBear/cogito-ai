from langchain_openai import ChatOpenAI


# Create a GPT model
gpt_low_temp = ChatOpenAI(
    model="gpt-5",
    temperature=0.0
)