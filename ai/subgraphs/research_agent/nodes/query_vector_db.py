import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from langchain_core.messages import SystemMessage, HumanMessage

from ai.models.gpt import gpt_extract_content
from ai.subgraphs.research_agent.model_config import MODEL_CONFIG
from ai.subgraphs.research_agent.schemas.graph_state import ResearchAgentState
from dbs.qdrant import Qdrant


def extract_text(resource_text, user_query):
    """Subnode to extract relevant text from the resource based on the user's query using the given model."""

    # Get configured model
    model = MODEL_CONFIG["query_vector_db"]

    # Construct prompt (system and user message)
    system_msg = SystemMessage(content=(
        "You are a text extraction agent. Extract text relevant to the user's query given the following guidelines:\n"
        "- Total text extracted can range from the entire source text to half the length of the source text.\n"
        "- Focus on relevant arguments, concepts, and ideas presented.\n"
        "- Only EXTRACT TEXT, do not SUMMARIZE, do not FORMULATE ARGUMENTS, do not address parts of the question "
        "unrelated to the source you've been given (for example, if the question addresses multiple philosophers, only "
        "extract text relevant to the philosopher from which the source is written by or about). You should only extract sections of text.\n\n"
        "User query:\n"
        f"{user_query}\n"
    ))

    user_msg = HumanMessage(content=(
        "Here is the source text:\n"
        f"{resource_text}"
    ))

    # Invoke model and return extracted output
    res = model.invoke([system_msg, user_msg], reasoning={"effort": "minimal"})
    citation = '; '.join(resource_text.split('\n')[-2:])  # Extract citation from resource text
    return (gpt_extract_content(res) + '\n' + citation).strip()

def query_vector_db(state: ResearchAgentState, qdrant: Qdrant):
    """
    Query the vector database with the given query and filters.
    Uses fuzzy matching to find best-matching authors and sources from PostgreSQL metadata.
    Returns accumulated resources and increments query count.
    """

    # Start timing and log
    print("::Querying vector database and extracting text...", end="", flush=True)
    start = time.perf_counter()

    # Extract graph state variables
    queries = state.get("queries")
    user_query = state.get("conversation", {}).get("last_user_message", "No last user message found")
    old_resources = state.get("resources", [])
    new_resources = old_resources.copy()

    # Query vector DB
    try:
        responses = qdrant.batch_query(queries)
    except Exception as e:
        print(f"\r\033[K::Vector DB batch query failed: {e}")
        return {"resources": new_resources}

    resources = []
    for payload in responses:
        content = payload.get("text", "")
        author = payload.get("author", "Unknown Author")
        source_title = payload.get("source_title", "Unknown Source")
        citation = payload.get("citations", "No Citation Provided")

        resource_text = f'"""\n{content}\n"""\n- From: ({author}, {source_title})\nCitation: "{citation}"'
        resources.append(resource_text)

    # Summarize new resources in parallel
    with ThreadPoolExecutor(max_workers=6) as executor:
        future_to_resource = {executor.submit(extract_text, r, user_query): r for r in resources}
        for future in as_completed(future_to_resource):
            summary = future.result()
            new_resources.append(gpt_extract_content(summary))

    # End timing and log
    end = time.perf_counter()
    print(f"\r\033[K::Vector database queried and sources extracted in {end - start:.2f}s")

    return {"resources": new_resources}
