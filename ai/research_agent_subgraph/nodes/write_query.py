from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.output_parsers import JsonOutputParser

from ai.models.gpt import gpt_low_temp
from ai.research_agent_subgraph.graph_state import ResearchAgentState
from ai.research_agent_subgraph.query_filter_schemas import QueryAndFilters


def write_query(state: ResearchAgentState):
    """
    Write a vector DB query based on the user's message and previous research.
    Generates a structured query with optional filters for author and source title.
    """

    # --- Initialize parser ---
    parser = JsonOutputParser(pydantic_object=QueryAndFilters)

    # --- Extract state variables ---
    messages = state.get("messages", [])
    resources = state.get("resources", [])
    queries = state.get("queries_made", [])

    # --- Build prompt ---
    system_prompt = (
        "You are a helpful semantic search assistant. "
        "Write a concise JSON object with these fields: "
        "'query' (string) and 'filters' (object with optional 'author' and 'source_title'). "
        "Do NOT include any extra text. Author and source_title will use fuzzy search to match, so "
        "feel free to use them even if they are not exact matches. "
        "Base the 'query' on the user's last message only."
    )
    user_prompt = f"Here are the recent chat messages:\n{state['messages']}"

    # Include previous research results if available
    if len(queries) != 0:
        user_prompt += f"\n\nThese are the previous research results: {resources}\nThese are the previous queries made: {queries}"

    # --- Invoke LLM ---
    chat_history = messages + [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt),
    ]
    result = gpt_low_temp.invoke(chat_history)

    # --- Parse and return output ---
    content = getattr(result, "content", str(result)) or ""
    parsed_output = parser.parse(content)

    return {"query": parsed_output, "queries_made": [*state["queries_made"], parsed_output]}
