import time

from langchain_core.messages import SystemMessage, HumanMessage
from pydantic import BaseModel

from ai.subgraphs.research_agent.model_config import MODEL_CONFIG
from ai.subgraphs.research_agent.schemas.graph_state import ResearchAgentState
from dbs.query import QueryAndFilters


class QueryAndFiltersList(BaseModel):
    """Output schema for LLM query generation."""

    queries: list[QueryAndFilters]


# Sample response to fit schema
SAMPLE_RESPONSE = \
"""
[
    {
        "query": "...",
        "filters": {
            "author": "...",
            "source_title": "..."
        }
    },
    {
        "query": "..."
    },
    ...
]
"""

def write_queries(state: ResearchAgentState):
    """
    Write a vector DB query based on the user's message and previous research.
    Generates a structured query with optional filters for author and source title.
    """

    # Start timing and log
    print("::Writing queries...", end="", flush=True)
    start = time.perf_counter()

    # Get configured model
    model = MODEL_CONFIG["write_queries"]
    structured_model = model.with_structured_output(QueryAndFiltersList)

    # Extract graph state variables
    feedback = state.get("queries_feedback", "No feedback yet.")
    conversation = state.get("conversation", {})

    # Construct prompt (system message and user message)
    system_msg = SystemMessage(content=(
        "You are a semantic-search assistant for philosophical research.\n" 
        "Your output is a JSON object containing a list of vector-DB search queries.\n\n"
        
        "Your task for each user message:\n"
        "1. Identify the key philosophical concepts, debates, and authors implied.\n"
        "2. Plan what evidence would be needed to answer the question fully.\n"
        "3. During planning, think about what philosophers and primary sources should be included to give a full genealogy of the idea/concept.\n"
        "4. If the user names authors/sources, put them ONLY in 'filters', not in the query text.\n"
        "5. Use short, dense query strings (keywords, not questions).\n"
        "6. Generate as little as 1 query for simple prompts or up to 6 for complex ones.\n"
        "7. Avoid redundancy.\n\n"
        
        "Output strictly in this structure:\n"
        f"{SAMPLE_RESPONSE}\n"
    ))

    conv_summary = conversation.get("summarized_context", "No prior context.")
    last_message = conversation.get("last_user_message", "No last user message")
    user_msg = HumanMessage(content=(
        f"Conversation summary:\n{conv_summary}\n\n"
        f"User's last message:\n{last_message}\n\n"
        f"Previous queries feedback:\n{feedback}"
    ))

    # Invoke LLM with structured output
    result = structured_model.invoke([system_msg, user_msg], reasoning={"effort": "medium"})

    # Stop timing and log
    end = time.perf_counter()
    print(f"\r\033[K::Wrote queries in {end - start:.2f}s")

    return {"queries": result.queries}
