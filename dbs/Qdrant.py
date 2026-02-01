import os
import uuid
import warnings

from qdrant_client import QdrantClient, models
from qdrant_client.http.models import MatchValue, FieldCondition, Filter
from rapidfuzz import process

from ai.research_agent.schemas.Citation import Citation
from ai.research_agent.schemas.QueryResult import QueryResult
from dbs.Postgres import Postgres
from dbs.QueryAndFilterSchemas import QueryAndFilters
from embed.Embedder import Embedder

# Cogito is meant to only be run in Docker compositions where Qdrant ports isn't publicly exposed or on local systems
# without HTTPS. Change this and line #37 accordingly.
warnings.filterwarnings(
 "ignore",
 message=r"Api key is used with an insecure connection\.",
 category=UserWarning,
 module=r"dbs\.Qdrant",
)

class Qdrant:
    """Qdrant vector database client with fuzzy-matched filtering."""

    # --- Methods ---
    def __init__(self):
        """Initialize Qdrant database client."""

        url = os.getenv("COGITO_QDRANT_URL")
        port = int(os.getenv("COGITO_QDRANT_PORT", "6334"))
        api_key = os.getenv("COGITO_QDRANT_API_KEY")
        self.collection = os.getenv("COGITO_QDRANT_COLLECTION")

        # --- Initialize database clients ---
        self.client = QdrantClient(url=url, grpc_port=port, prefer_grpc=True, https=False, api_key=api_key)
        self.postgres_client = Postgres()
        self.embedder = Embedder()

    def close(self):
        """Close Qdrant client connection."""

        self.client.close()

    def batch_query(self, queries: list[QueryAndFilters]) -> list[QueryResult]:
        """Batch query Qdrant with per-query fuzzy filters."""

        author_sources = self.postgres_client.author_sources
        all_authors = list(author_sources.keys())
        all_sources = self.postgres_client.all_sources

        # --- Batch embed all query texts ---
        query_texts = [q.get("query") for q in queries]
        vectors = self.embedder.embed_batch(query_texts)

        # --- Build SearchRequest and results lists ---
        search_requests = []
        results_out = []

        for q, vector in zip(queries, vectors):
            filter_obj = None

            if q.get("filters"):
                conditions = []
                f = q.get("filters")

                selected_author = None

                # fuzzy match author (if provided)
                if f.get("author"):
                    best_author = process.extractOne(f.get("author"), all_authors)
                    if best_author:
                        selected_author = best_author[0]
                        score = best_author[1]

                        if score <= 80:
                            r = {
                                "query": q,
                                "source": "Project Gutenberg Vector DB",
                                "result": f"'{f.get('author')}' not found in author list. This author is not in the database. "
                                          f"Closest match: '{selected_author}'."
                            }
                            results_out.append(r)
                            continue  # skip this query if author match is too low

                        conditions.append(
                            FieldCondition(
                                key="author",
                                match=MatchValue(value=selected_author)
                            )
                        )

                # fuzzy match source with scoped candidate set
                if f.get("source_title"):
                    if selected_author and author_sources.get(selected_author):
                        candidate_sources = author_sources[selected_author]
                    else:
                        candidate_sources = all_sources

                    best_source = process.extractOne(f.get("source_title"), candidate_sources)
                    if best_source:
                        selected_source = best_source[0]
                        score = best_source[1]

                        if score <= 80:
                            r = {
                                "query": q,
                                "source": "Project Gutenberg Vector DB",
                                "result": f"'{f.get('source_title')}' not found in source list. This source is either not "
                                          f"written by the author '{selected_author}' or is not in the database. Best "
                                          f"match: '{selected_source}'."
                            }
                            results_out.append(r)
                            continue

                        conditions.append(
                            FieldCondition(
                                key="title",
                                match=MatchValue(value=selected_source)
                            )
                        )

                if conditions:
                    filter_obj = Filter(must=conditions)

            # Add search request
            search_requests.append(
                models.QueryRequest(
                    query=vector,
                    limit=1,
                    filter=filter_obj,
                    with_payload=True,
                    with_vector=False
                )
            )

        # --- Execute all queries in a single batch ---
        batch_results = self.client.query_batch_points(
            collection_name=self.collection,
            requests=search_requests
        )

        # --- Convert Qdrant results into your desired payload lists ---
        seen_ids = set()
        for query, response in zip(queries, batch_results):
            for point in response.points:
                if point.id not in seen_ids:
                    seen_ids.add(point.id)
                    payload = point.payload

                    content = payload.get("text", "null")
                    author = payload.get("author", "null")
                    source_title = payload.get("title", "null")
                    section = payload.get("section", "null")
                    citation: Citation = {"title": source_title, "authors": [author], "source": "Project Gutenberg", "section": section}

                    result = (content, citation)
                    r: QueryResult = {"id": int(uuid.uuid4()), "query": query, "source": "Project Gutenberg Vector DB", "result": result}
                    results_out.append(r)

        return results_out
