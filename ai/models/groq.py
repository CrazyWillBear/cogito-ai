from dotenv import load_dotenv
from langchain_groq import ChatGroq

load_dotenv()  # needed for langchain_groq to find GROQ_API_KEY in environment

oss_120b = ChatGroq(
    model="openai/gpt-oss-120b",
    temperature=0.2,
    reasoning_effort="medium",
    reasoning_format="parsed"
)

oss_20b = ChatGroq(
    model="openai/gpt-oss-20b",
    temperature=0.0,
    reasoning_effort="low",
    reasoning_format="parsed"
)

llama_8b_instant = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0.1
)

llama_4_maverick = ChatGroq(
    model="meta-llama/llama-4-maverick-17b-128e-instruct",
    temperature=0.1
)
