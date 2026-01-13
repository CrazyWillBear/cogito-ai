from langchain_core.messages import SystemMessage, AIMessage
from langchain_core.output_parsers import JsonOutputParser

from ai.models.util import safe_invoke, extract_content
from ai.research_agent.model_config import RESEARCH_AGENT_MODEL_CONFIG
from ai.research_agent.schemas.ResearchAgentState import ResearchAgentState
from ai.research_agent.schemas.ResearchEffort import ResearchEffort
from ai.research_agent.sources.stringify import stringify_query_results
from util.SpinnerController import SpinnerController

# --- Define constants ---
MAX_ITERATIONS_DEEP = 10
MAX_ITERATIONS_SIMPLE = 4

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
    research_effort = state.get("research_effort", None)

    if research_effort == ResearchEffort.DEEP:
        max_iterations = MAX_ITERATIONS_DEEP
    else:
        max_iterations = MAX_ITERATIONS_SIMPLE

    # Hard stop to avoid infinite loops in the graph
    if research_iterations > max_iterations:
        if spinner_controller:
            spinner_controller.set_text("::Reached max research iterations; finalizing")
        return {"completed": True}

    # Construct prompt (system message and user message)
    system_msg = SystemMessage(
        content=(
            "## YOUR ROLE\n"
            "You are a research query planner. Analyze the user's question and generate search queries for two sources, "
            "or stop research by returning null for all fields.\n\n"

            f"## ITERATION {research_iterations} (1-indexed):\n"
            f"For this task, your hard limit is {max_iterations}, which will be your final iteration. If you finish "
            f"early, that's okay.\n\n"

            "## SOURCES\n"
            "1. **Vector DB**: Primary source chunks from Project Gutenberg philosophy texts\n"
            "2. **SEP**: Stanford Encyclopedia articles for conceptual overviews\n\n"

            "## OUTPUT FORMAT (CRITICAL)\n"
            "Respond ONLY with the following JSON structure, no other text. It uses `#` comments which don't exist in "
            "real JSON, so don't add any comments of your own. DO NOT USE TOOLS!!!\n"
            f"```json{SAMPLE_RESPONSE}```\n\n"

            "## SOURCE SELECTION\n"
            "**Vector DB**: Named philosophers, specific passages, textual evidence, author's development of ideas\n"
            "**SEP**: General concepts, movements, debates, overviews, multiple philosophers, scholarly interpretation\n"
            "**Both**: Idea genealogy, comparing sources with concepts, evidence + interpretation\n\n"

            "## QUERY RULES\n"
            "- MAX 5 queries for Project Gutenberg Vector DB (for this iteration, not total)\n"
            "- Max 2 queries for SEP (for this iteration, not total)\n"
            "- MIN 1 completed query total across all sources\n"
            "- NEVER repeat past queries\n"
            "- Vector DB: Contains both large and medium sized chunks, can be broad or specific. Can be searched both "
            "semantically and normally.\n"
            f"- SEP:\n\"\"\"{SEP_SEARCH_RULES}\"\"\"\n\n"

            "## END RESEARCH WHEN:\n"
            "- You have 1-2 relevant results with relevant content (simple questions)\n"
            "- You have sufficient research (>=3 relevant results) to answer the question (medium-to-complex questions)\n"
            "- Past queries show necessary sources are unavailable/irrelevant\n\n"
            
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
    research_history_message = AIMessage(content=(
        f"PREVIOUS QUERIES + RESULTS:\n```\n{stringify_query_results(query_results)}\n```"
    ))

    # Invoke LLM with structured output and retry parsing on invalid JSON
    model, reasoning_effort = RESEARCH_AGENT_MODEL_CONFIG["plan_research"]
    parser = JsonOutputParser()
    max_parse_attempts = 3
    attempt = 0
    result = None

    while attempt < max_parse_attempts:
        try:
            # Optional spinner feedback per attempt
            if spinner_controller:
                spinner_controller.set_text(f"::Planning next step")

            llm_output = safe_invoke(model, [*conversation, system_msg, research_history_message], reasoning_effort)
            content = extract_content(llm_output)
            result = parser.parse(content)
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

    # Check for research completion
    if not result.get("vector_db_queries") and not result.get("stanford_encyclopedia_queries"):
        return {"completed": True}

    return {
        "vector_db_queries": result.get("vector_db_queries"),
        "sep_queries": result.get("stanford_encyclopedia_queries"),
        "research_iterations": research_iterations + 1,
    }
