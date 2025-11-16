import psycopg2
from qdrant_client import QdrantClient
from qdrant_client.http.models import Filter, MatchValue, FieldCondition
from rapidfuzz import process

from ai.research_agent_subgraph.graph_state import ResearchAgentState
from embed.embed import embed

VEC_DB_URL = "http://localhost:6333"
VEC_COLLECTION = "philosophy"


def query_vector_db(state: ResearchAgentState):
    """
    Query the vector database with the given query and filters.
    Uses fuzzy matching to find best-matching authors and sources from PostgreSQL metadata.
    Returns accumulated resources and increments query count.
    """

    # --- Initialize database clients ---
    qdrant_client = QdrantClient(VEC_DB_URL)

    conn = psycopg2.connect(
        host="localhost",
        port=5432,
        dbname="filters",
        user="munir",
        password="123"
    )
    cur = conn.cursor()

    # --- Extract state variables ---
    query = state.get("query", None)
    vector = embed(query.get("query", ""))
    filters = query.get("filters", None)

    # --- Build filter conditions ---
    _filter_conditions = []
    if filters is not None:
        author = filters.get("author")
        source_title = filters.get("source_title")

        # Fuzzy match author
        if author is not None:
            cur.execute("SELECT DISTINCT authors FROM filters;")
            authors = [row[0] for row in cur.fetchall()]
            best_author = process.extractOne(author, authors)
            if best_author:
                _filter_conditions.append(FieldCondition(key="author", match=MatchValue(value=best_author[0])))

        # Fuzzy match source
        if source_title is not None:
            cur.execute("SELECT DISTINCT sources FROM filters;")
            sources = [row[0] for row in cur.fetchall()]
            best_source = process.extractOne(source_title, sources)
            if best_source:
                _filter_conditions.append(FieldCondition(key="source", match=MatchValue(value=best_source[0])))

    # Build filter only if we have conditions
    _filter = Filter(must=_filter_conditions) if _filter_conditions else None

    # --- Query vector database ---
    results = qdrant_client.query_points(
        collection_name=VEC_COLLECTION,
        query=vector,
        limit=2,
        query_filter=_filter
    )

    # --- Accumulate results ---
    old_resources = state.get("resources", [])
    new_resources = old_resources + results.points

    # --- Cleanup connections ---
    cur.close()
    conn.close()
    qdrant_client.close()

    return {"resources": new_resources}
