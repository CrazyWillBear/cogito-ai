from langchain_core.messages import SystemMessage, HumanMessage
from pydantic import BaseModel

from ai.research_agent.model_config import MODEL_CONFIG
from ai.research_agent.schemas.graph_state import ResearchAgentState
from ai.research_agent.schemas.query import QueryAndFilters


class QueryAndFiltersList(BaseModel):
    """
    Schema for a list of QueryAndFilters objects.
    Used when multiple queries with filters are needed.
    """

    queries: list[QueryAndFilters]


def reason_about_queries(messages: list):
    """Generate reasoning about how to break down a user's question into multiple targeted search queries."""
    # Get model
    model = MODEL_CONFIG["write_queries"]

    # --- Build prompts ---
    system_prompt = (
        "You are a reasoning agent. Your job is to reason about how to break down the user's question into multiple "
        "targeted search queries that will find the best information.\n\n"
        "First, think through your strategy:\n"
        "- What are the key aspects or sub-questions to address?\n"
        "- Should you search broadly or use specific filters?\n"
        "- What different angles or perspectives are needed? (Try a different angle for each query)\n\n"
        "Do not actually output any queries, just reason about how you would approach writing them.\n"
        "Respond with nothing but the reasoning itself, no additional text."
    )

    user_prompt = f"Here is the chat history:\n{messages}"

    # --- Invoke LLM ---
    chat_history = [system_prompt, user_prompt]
    result = model.invoke(chat_history)

    return result.content


def write_queries(state: ResearchAgentState):
    """
    Write a vector DB query based on the user's message and previous research.
    Generates a structured query with optional filters for author and source title.
    """
    # Get model
    model = MODEL_CONFIG["write_queries"]
    structured_model = model.with_structured_output(QueryAndFiltersList)

    # --- Extract state variables ---
    resources = state.get("resources", [])
    queries = state.get("queries_made", [])
    messages = state.get("messages", [])

    # --- Reason about queries ---
    reasoning = reason_about_queries(messages)

    # --- Build prompts ---
    system_prompt = (
        "You are a helpful semantic search assistant. Your job is to break down the user's question into multiple "
        "targeted search queries that will find the best information.\n\n"
        "Here is your previous reasoning:\n"
        f"\"\"\"\n{reasoning}\n\"\"\"\n\n"
        "Generate 2-5 queries depending on complexity. DO NOT PUT THE WORK OR AUTHOR IN THE SEARCH STRING, PUT IT IN "
        "THE FILTERS!!! Each query should have:\n"
        "- A focused search string for semantic search across a vector db\n"
        "- Optional filters for 'author' and/or 'source_title'\n"
    )

    user_prompt = f"Here is the chat history:\n{messages}"

    # Include previous research results if available
    if len(queries) != 0:
        user_prompt += (
            f"\n\nPrevious research results: {resources}"
            f"\nPrevious queries made: {queries}"
            "\n\nConsider what information gaps remain and avoid redundant queries."
        )

    # --- Invoke LLM ---
    chat_history = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt),
    ]
    result = structured_model.invoke(chat_history)

    # --- Update state with new query ---
    return {"queries": result.queries, "queries_made": [*state["queries_made"], *result.queries]}
