import uuid
from concurrent.futures import ThreadPoolExecutor

from rich.status import Status

from ai.research_agent.schemas.QueryResult import QueryResult
from ai.research_agent.schemas.ResearchAgentState import ResearchAgentState
from ai.research_agent.sources.sep import query_sep
from ai.research_agent.sources.vector_db import query_vector_db
from dbs.Qdrant import Qdrant


def execute_queries(state: ResearchAgentState, qdrant: Qdrant, status: Status | None):
    """Query the vector database with the queries and filters from the graph state. Set resources to old resources +
    new ones."""

    # Extract graph state variables
    vector_db_queries = state.get("vector_db_queries", None)
    sep_queries = state.get("sep_queries", None)
    query_results = state.get("query_results", [])
    conversation = state.get("conversation", [])
    all_results = state.get("all_raw_results", set())

    # Deduplicate
    if vector_db_queries:
        for query in vector_db_queries:
            if query in [q["query"] for q in query_results if q.get("source") == "Project Gutenberg Vector DB"]:
                vector_db_queries.remove(query)
                query_result: QueryResult = {
                    "id": int(uuid.uuid4()),
                    "query": query,
                    "source": "Project Gutenberg Vector DB",
                    "result": "[Duplicate Query Omitted, Already Retrieved In Previous Queries]"
                }
                query_results.append(query_result)
    if sep_queries:
        for query in sep_queries:
            if query in [q["query"] for q in query_results if q.get("source") == "SEP"]:
                sep_queries.remove(query)
                query_result: QueryResult = {
                    "id": int(uuid.uuid4()),
                    "query": query,
                    "source": "SEP",
                    "result": "[Duplicate Query Omitted, Already Retrieved In Previous Queries]"
                }
                query_results.append(query_result)

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
                raw_result = result.get("result")
                if type(raw_result) == tuple:
                    raw_result = raw_result[0]
                if raw_result in all_results:
                    result["result"] = "[Duplicate Result Omitted, Already Retrieved In Previous Queries]"
                else:
                    all_results.add(raw_result)
                query_results.append(result)

    return {"query_results": query_results, "all_raw_results": all_results}
