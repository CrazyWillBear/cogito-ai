from dotenv import load_dotenv
from langchain_groq import ChatGroq

load_dotenv()  # needed for langchain_groq to find GROQ_API_KEY in environment

oss_120b_low = ChatGroq(
    model="openai/gpt-oss-120b",
    temperature=0.3,
    reasoning_effort="low",
    reasoning_format="parsed"
)

oss_120b_med = ChatGroq(
    model="openai/gpt-oss-120b",
    temperature=0.3,
    reasoning_effort="medium",
    reasoning_format="parsed"
)

oss_20b_low_temp = ChatGroq(
    model="openai/gpt-oss-20b",
    temperature=0.0,
    reasoning_effort="low",
    reasoning_format="parsed",
)

oss_20b_med_temp_med_reasoning = ChatGroq(
    model="openai/gpt-oss-20b",
    temperature=0.4,
    reasoning_effort="medium",
    reasoning_format="parsed",
)

llama_8b_instant = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0.1
)

llama_4_scout = ChatGroq(
    model="meta-llama/llama-4-scout-17b-16e-instruct",
    temperature=0.0
)
