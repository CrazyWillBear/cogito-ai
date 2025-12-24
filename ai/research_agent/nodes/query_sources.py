import time

from ai.research_agent.sources.sep import query_sep
from ai.research_agent.sources.vector_db import query_vector_db
from ai.research_agent.schemas.ResearchAgentState import ResearchAgentState
from dbs.Qdrant import Qdrant


def query_sources(state: ResearchAgentState, qdrant: Qdrant):
    """Query the vector database with the queries and filters from the graph state. Set resources to old resources +
    new ones."""

    # Start timing and log
    print("::Searching sources...", end="", flush=True)
    start = time.perf_counter()

    # Extract graph state variables
    vector_db_queries = state.get("vector_db_queries", None)
    sep_queries = state.get("sep_queries", None)

    user_query = state.get("conversation", {}).get("last_user_message", "no last user message found")
    resources = state.get("resources", [])

    # Query vector db and add to resources
    if vector_db_queries:
        resources.extend(query_vector_db(vector_db_queries, user_query, qdrant))
    if sep_queries:
        resources.extend(query_sep(sep_queries, user_query))

    # End timing and log
    end = time.perf_counter()
    print(f"\r\033[K::Sources searched in {end - start:.2f}s")

    return {"resources": resources}
