from qdrant_client import QdrantClient
from agent.research_agent_subgraph.graph import ResearchAgentState
from tools.embed import embed

DB_URL = "http://localhost:6333"
COLLECTION = "philosophy"

def query_vector_db(state: ResearchAgentState):
    """Query the vector database with the given query and filters."""
    client = QdrantClient(DB_URL)

    query = state["query"]
    vector = embed(query)

    results = client.query_points(
        collection_name="philosophy",
        query=vector,
        limit=2
    )

    # Append new results to existing resources without overwriting
    old_resources = state.get("resources", [])
    new_resources = old_resources + results.points

    # Increment the queries_made count
    queries_made = state.get("queries_made", 0) + 1

    return {"resources": new_resources, "queries_made": queries_made}
