from dbs.Qdrant import Qdrant
from dbs.QueryAndFilterSchemas import QueryAndFilters


def query_vector_db(queries: list[QueryAndFilters], last_user_msg: str, qdrant: Qdrant):
    """Query the vector db and extract text in parallel."""


    try:
        response = qdrant.batch_query(queries)
    except Exception as e:
        print(f"\r\033[k::vector db batch query failed: {e}")
        return None

    return response
