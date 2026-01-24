from ai.research_agent.sources.sep import query_sep
from ai.research_agent.sources.vector_db import query_vector_db
from ai.research_agent.schemas.ResearchAgentState import ResearchAgentState
from dbs.Qdrant import Qdrant
from cli.SpinnerController import SpinnerController
from concurrent.futures import ThreadPoolExecutor


def execute_queries(state: ResearchAgentState, qdrant: Qdrant, spinner_controller: SpinnerController = None):
    """Query the vector database with the queries and filters from the graph state. Set resources to old resources +
    new ones."""

    if spinner_controller:
        spinner_controller.set_text("::Pulling from sources")

    # Extract graph state variables
    vector_db_queries = state.get("vector_db_queries", None)
    sep_queries = state.get("sep_queries", None)

    conversation = state.get("conversation", [])
    query_results = state.get("query_results", [])
    all_results = state.get("all_raw_results", set())

    # Run vector DB and SEP queries concurrently (only if present)
    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = []
        if vector_db_queries:
            futures.append(executor.submit(query_vector_db, vector_db_queries, qdrant))
        if sep_queries:
            futures.append(executor.submit(query_sep, sep_queries, conversation))

        for future in futures:
            try:
                results = future.result() or []
            except Exception:
                results = []
            for result in results:
                raw_result = result.get("result", None)
                if raw_result in all_results:
                    result["result"] = "[Duplicate Result Omitted, Already Retrieved In Previous Queries]"
                else:
                    all_results.add(raw_result)
                query_results.append(result)

    return {"query_results": query_results, "all_raw_results": all_results}
