from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
from parsers.boolean import BooleanOutputParser

# A small model is perfectly capable of basic classification like this
model = ChatOllama(
    model="llama3.2:3b",
    temperature=0.0
)

prompt = ChatPromptTemplate.from_messages([
    SystemMessage(content="You are a classifier that determines if the last message sent by the user requires external research to respond accurately."),
    HumanMessage(content="Here is the conversation so far:\n{chat_history}\nDoes the very last user message need research? Answer only with True or False.")
])

parser = BooleanOutputParser(true_val="True", false_val="False")

chain = prompt | model | parser

def research_classifier_node(state):
    # Extract chat history from state
    history = state["chat_history"]
    conversation_text = "\n".join([f"{msg.role}: {msg.content}" for msg in history])

    # Invoke the chain with the conversation text
    needs_research = chain.invoke({"chat_history": conversation_text})
    return {"needs_research": needs_research}