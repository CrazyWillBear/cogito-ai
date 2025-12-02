import os

from dotenv import load_dotenv
from qdrant_client import QdrantClient, models
from qdrant_client.http.models import MatchValue, FieldCondition, Filter
from rapidfuzz import process

from dbs.postgres_filters import PostgresFilters
from dbs.query import QueryAndFilters
from embed.embed import Embeder


class Qdrant:
    """Qdrant vector database client with fuzzy-matched filtering."""

    # --- Methods ---
    def __init__(self):
        """Initialize Qdrant database client."""

        load_dotenv()
        url = os.getenv("COGITO_QDRANT_URL")
        port = int(os.getenv("COGITO_QDRANT_PORT", "6334"))
        api_key = os.getenv("COGITO_QDRANT_API_KEY")
        self.collection = os.getenv("COGITO_QDRANT_COLLECTION")

        # --- Initialize database clients ---
        self.client = QdrantClient(url=url, grpc_port=port, api_key=api_key, prefer_grpc=True, https=False)
        self.postgres_client = PostgresFilters()
        self.embedder = Embeder()

    def close(self):
        """Close Qdrant client connection."""

        self.client.close()

    def batch_query(self, queries: list[QueryAndFilters]):
        """Batch query Qdrant with per-query fuzzy filters."""

        author_sources = self.postgres_client.author_sources
        all_authors = list(author_sources.keys())
        all_sources = self.postgres_client.all_sources

        # --- Batch embed all query texts ---
        query_texts = [q.query for q in queries]
        vectors = self.embedder.embed_batch(query_texts)

        # --- Build SearchRequest list ---
        search_requests = []

        for q, vector in zip(queries, vectors):
            filter_obj = None

            if q.filters:
                conditions = []
                f = q.filters

                selected_author = None

                # fuzzy match author (if provided)
                if f.author:
                    best_author = process.extractOne(f.author, all_authors)
                    if best_author:
                        selected_author = best_author[0]
                        conditions.append(
                            FieldCondition(
                                key="author",
                                match=MatchValue(value=selected_author)
                            )
                        )

                # fuzzy match source with scoped candidate set
                if f.source_title:
                    if selected_author and author_sources.get(selected_author):
                        candidate_sources = author_sources[selected_author]
                    else:
                        candidate_sources = all_sources

                    best_source = process.extractOne(f.source_title, candidate_sources)
                    if best_source:
                        conditions.append(
                            FieldCondition(
                                key="source_title",
                                match=MatchValue(value=best_source[0])
                            )
                        )

                if conditions:
                    filter_obj = Filter(must=conditions)

            # Add search request
            search_requests.append(
                models.QueryRequest(
                    query=vector,
                    limit=2,
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
        seen_ids, results_out = set(), []
        for response in batch_results:
            for point in response.points:
                if point.id not in seen_ids:
                    seen_ids.add(point.id)
                    results_out.append(point.payload)

        return results_out
