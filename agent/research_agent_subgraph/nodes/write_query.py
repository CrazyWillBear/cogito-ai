from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.output_parsers import JsonOutputParser

from agent.models.gpt import gpt_low_temp
from agent.research_agent_subgraph.graph_state import ResearchAgentState
from agent.research_agent_subgraph.query_filter_schemas import QueryAndFilters


def write_query(state: ResearchAgentState):
    """Write an initial vector DB query based on the user's last message."""
    # Create output parser
    parser = JsonOutputParser(pydantic_object=QueryAndFilters)

    # Get state variables
    messages = state.get("messages", [])
    resources = state.get("resources", [])

    # Prompts
    system_prompt = (
        "You are a helpful semantic search assistant. "
        "Write a concise JSON object with these fields: "
        "'query' (string) and 'filters' (object with optional 'author' and 'source_title'). "
        "Do NOT include any extra text. Author and source_title will use fuzzy search to match, so "
        "feel free to use them even if they are not exact matches. "
        "Base the 'query' on the user's last message only. "
    )
    user_prompt = f"Here are the recent chat messages:\n{state['messages']}"

    # If there are previous research results, include them
    if state["queries_made"] is None:
        # Initialize query count if not present
        state["queries_made"] = 0
    else:
        user_prompt += f"\n\nThese are the previous research results: {resources}"

    # Invoke as structured message list
    chat_history = messages + [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt),
    ]
    result = gpt_low_temp.invoke(chat_history)

    # Get the model output content, parse, and return
    content = getattr(result, "content", str(result)) or ""
    parsed_output = parser.parse(content)
    return {"query": parsed_output}
