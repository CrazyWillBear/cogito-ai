from dbs.Qdrant import Qdrant
from dbs.query import QueryAndFilters


def query_vector_db(queries: list[QueryAndFilters], last_user_msg: str, qdrant: Qdrant):
    """Query the vector db and extract text in parallel."""

    try:
        responses = qdrant.batch_query(queries)
    except Exception as e:
        print(f"\r\033[k::vector db batch query failed: {e}")
        return None

    resources = []
    for payload in responses:
        content = payload.get("text", "null")

        author = payload.get("author", "null")
        source_title = payload.get("title", "null")
        section = payload.get("section", "null")  # label may be different

        citation = f"Author: {author}\nSource Title: {source_title}\nSection: {section}\n"
        resources.append((content, citation))

    return resources
