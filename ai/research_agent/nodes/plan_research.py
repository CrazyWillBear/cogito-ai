from langchain_core.messages import SystemMessage
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel

from ai.models.extract_content import safe_invoke, extract_content
from ai.research_agent.model_config import RESEARCH_AGENT_MODEL_CONFIG
from ai.research_agent.schemas.ResearchAgentState import ResearchAgentState
from ai.research_agent.sources.stringify import stringify_query_results
from dbs.QueryAndFilterSchemas import QueryAndFilters
from util.SpinnerController import SpinnerController


class QueryList(BaseModel):
    """Output schema for LLM query generation."""

    vector_db_queries: list[QueryAndFilters] | None
    stanford_encyclopedia_queries: list[str] | None


MAX_ITERATIONS = 5

# Sample response to fit schema
SAMPLE_RESPONSE = \
"""
# Standard query (don't put ANY comments in your response)
{
  "vector_db_queries": [  # must be a list of query + filter objects or null
    {
      "query": "sample query",  # must be a string
      "filters": {
        "author": "sample author",  # string or null
        "source_title": "sample source title"  # string or null
      }
    },
    {
      "query": "sample query",
      "filters": null  # this can be null
    }
  ],
  "stanford_encyclopedia_queries": [  # must be a list of strings or null
    "free will",  # must be a string
    "determinism"  # you can have multiple
  ]
}
# To end research
{
    "vector_db_queries": null,              # you can also just set one of these to null
    "stanford_encyclopedia_queries": null   # if you want to use one instead of both
}
"""

SEP_SEARCH_RULES = \
"""
- Fuzzy match: append `~` to a word
  Example: `leibnitz~`

- Required terms: prefix with `+`
  Example: `+leibniz +locke`

- Excluded terms: prefix with `-`
  Example: `+leibniz -locke`

- Boolean operators: use `AND`, `OR`, `NOT` (uppercase)
  Example: `(leibniz OR newton) NOT locke`

- Exact phrase: wrap in double quotes
  Example: `"the world is all that is the case"`

- Proximity search: quoted words + `~N`
  Example: `"world case"~5`

- Title search: `title:word`
  Example: `title:Descartes`

- Author search: `author:name`
  Example: `author:smith`

- Wildcard: use `*` for partial matches
  Example: `logic*`, `title:contract*`

- Case-insensitive: `leibniz` == `Leibniz`

- Rules can be combined
  Example: `+semantics +logic -title:logic*`
"""

# Shared tokenizer and structured model for reuse
MODEL, DEFAULT_REASONING = RESEARCH_AGENT_MODEL_CONFIG["plan_research"]
PARSER = JsonOutputParser()


def plan_research(state: ResearchAgentState, spinner_controller: SpinnerController = None):
    """Write a vector DB query based on the user's message and previous research.
    Generates a structured query with optional filters for author and source title.
    """

    if spinner_controller:
        spinner_controller.set_text("::Planning next step")

    # Extract graph state variables
    conversation = state.get("conversation", [])
    query_results = state.get("query_results", [])
    research_iterations = state.get("research_iterations", 1)

    # Hard stop to avoid infinite loops in the graph
    if research_iterations >= MAX_ITERATIONS:
        if spinner_controller:
            spinner_controller.set_text("::Reached max research iterations; finalizing")
        return {"completed": True}

    # Construct prompt (system message and user message)
    system_msg = SystemMessage(
        content=(
            "You are a research query planner. Analyze the user's question and generate search queries for two sources, "
            "or stop research by returning null for all fields.\n\n"

            f"**Iteration {research_iterations}**: For simple questions, keep iterations below 2. Complex questions can "
            "continue until satisfied.\n\n"

            "## SOURCES\n"
            "1. **Vector DB**: Primary source chunks from Project Gutenberg philosophy texts\n"
            "2. **SEP**: Stanford Encyclopedia articles for conceptual overviews\n\n"

            "## OUTPUT FORMAT (CRITICAL)\n"
            "Respond ONLY with the following JSON structure, no other text. It uses `#` comments which don't exist in "
            "real JSON, so don't add any comments of your own.\n"
            f"```json{SAMPLE_RESPONSE}```\n\n"

            "## SOURCE SELECTION\n"
            "**Vector DB**: Named philosophers, specific passages, textual evidence, author's development of ideas\n"
            "**SEP**: General concepts, movements, debates, overviews, multiple philosophers, scholarly interpretation\n"
            "**Both**: Idea genealogy, comparing sources with concepts, evidence + interpretation\n\n"

            "## QUERY RULES\n"
            "- MAX 2 queries per source\n"
            "- NEVER repeat past queries\n"
            "- Vector DB: One concept per query, broad enough for large chunks, author names in 'filters' only, semantic search works\n"
            "- SEP: Concise encyclopedia terms, usually 1 query unless genuinely distinct facets\n"
            f"- SEP-specific rules:\n\"\"\"{SEP_SEARCH_RULES}\"\"\"\n\n"

            "## END RESEARCH WHEN:\n"
            "- You have 2-3 relevant sources with relevant content, OR\n"
            "- Past queries show sources are unavailable/irrelevant, OR\n"
            "- You have surpassed 3 iterations for a simple question (not multifaceted, can be Googled, etc.)\n\n"
            
            "## HOW TO END RESEARCH:\n"
            "Set both `stanford_encyclopedia_queries` and `vector_db_queries` to `null`\n\n"
            
            "## IN YOUR REASONING/OUTPUT:\n"
            "Do NOT formulate the final response. Reason about the comprehensiveness of prior research and what is needed "
            "next, if anything. Consider how many iterations have occurred and the complexity of the question.\n\n"
            
            "## ADVICE:\n"
            "- If a user asks about previous research that you don't have, re-query for those sources.\n"
            "- If prior queries yielded no results, adjust your approach\n"
            "- If a resource is missing from the database, note it and avoid re-querying it\n\n"

            f"## PREVIOUS QUERIES + RESULTS\n```\n{stringify_query_results(query_results)}\n```"
        )
    )

    # Invoke LLM with structured output and retry parsing on invalid JSON
    max_parse_attempts = 3
    attempt = 0
    result = None

    while attempt < max_parse_attempts:
        try:
            # Optional spinner feedback per attempt
            if spinner_controller:
                spinner_controller.set_text(f"::Planning next step")

            llm_output = safe_invoke(MODEL, [*conversation, system_msg], DEFAULT_REASONING)
            content = extract_content(llm_output)
            result = PARSER.parse(content)
            break
        except Exception as e:
            attempt += 1
            if spinner_controller:
                spinner_controller.set_text(f"::Retrying JSON parse ({attempt}/{max_parse_attempts})")

    # If parsing failed after retries, end gracefully
    if result is None:
        if spinner_controller:
            spinner_controller.set_text("::Planning failed due to invalid JSON; ending research")
        return {"completed": True}

    if not result.get("vector_db_queries") and not result.get("stanford_encyclopedia_queries"):
        return {"completed": True}

    return {
        "vector_db_queries": result.get("vector_db_queries"),
        "sep_queries": result.get("stanford_encyclopedia_queries"),
        "research_iterations": research_iterations + 1,
    }
