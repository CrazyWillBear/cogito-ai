import tiktoken
from langchain_core.messages import SystemMessage, AIMessage
from langchain_core.output_parsers import JsonOutputParser
from rich.status import Status

from ai.models.util import safe_invoke, extract_content
from ai.research_agent.model_config import RESEARCH_AGENT_MODEL_CONFIG
from ai.research_agent.schemas.ResearchAgentState import ResearchAgentState
from ai.research_agent.schemas.ResearchEffort import ResearchEffort
from ai.research_agent.sources.stringify import stringify_query_results

# --- Define constants ---
MAX_ITERATIONS_DEEP = 8
MAX_ITERATIONS_SIMPLE = 4
MAX_TOKENS = 100000

SAMPLE_RESPONSE = \
"""
# Standard query (don't put ANY comments in your response)
{
  "long_term_plan":  "sample plan",  # long term research plan, can be over multiple iterations, can be null to end research
  "short_term_plan": "sample plan",  # short term plan for the current iteration, can be null to end research
  "vector_db_queries": [  # optional field, must be a list of query + filter objects or null
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
  "stanford_encyclopedia_queries": [  # optional field, must be a list of strings or null
    "free will",  # must be a string
    "determinism"  # you can have multiple
  ],
  "ids_to_remove": [  # optional field, list of IDs to remove from future consideration
    "id1",  # must be a string
    "id2"  # must be a string
  ]
}
# To end research
{
    "long_term_plan": null,
    "short_term_plan": null,
    "vector_db_queries": null,
    "stanford_encyclopedia_queries": null,
    "ids_to_remove": null
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

def _prune_research_results(state: ResearchAgentState, ids_to_remove: list[str]):
    """Remove research results with the given IDs from the graph state."""
    if not ids_to_remove:
        return

    query_results = state.get("query_results", [])

    # Filter out results with IDs to remove
    for result in query_results:
        if str(result["id"]) in ids_to_remove:
            result["result"] = "[Removed from future consideration by research planner]"

    state["query_results"] = query_results

def plan_research(state: ResearchAgentState, status: Status | None):
    """Write a vector DB query based on the user's message and previous research.
    Generates a structured query with optional filters for author and source title.
    """

    if status:
        short_term_plan = state.get("short_term_plan", "Planning my next move...")
        status.update(short_term_plan)

    # Extract graph state variables
    conversation = state.get("conversation", [])
    query_results = state.get("query_results", [])
    long_term_plan = state.get("long_term_plan", "No long term plan yet.")
    short_term_plan = state.get("short_term_plan", "No short term plan yet.")
    research_iterations = state.get("research_iterations", 1)
    research_effort = state.get("research_effort", None)

    if research_effort == ResearchEffort.DEEP:
        max_iterations = MAX_ITERATIONS_DEEP
    else:
        max_iterations = MAX_ITERATIONS_SIMPLE

    # Hard stop to avoid infinite loops in the graph
    if research_iterations > max_iterations:
        return {"completed": True}

    # Token limit check
    encoding = tiktoken.get_encoding("cl100k_base")
    tokens = encoding.encode(str([msg.content for msg in conversation]))
    if len(tokens) >= MAX_TOKENS:
        return {"completed": True}

    # Construct prompt (system message and user message)
    system_msg = SystemMessage(
        content=(
            "## YOUR ROLE\n"
            "You are a PLANNER NODE WITHIN AN AGENT. Analyze the user's question and generate a plan + search queries, "
            "or stop research by returning null for all fields. You do not continue the conversation.\n\n"

            f"## ITERATION {research_iterations} (1-indexed):\n"
            f"For this task, your hard limit is {max_iterations}, which will be your final iteration. If you finish "
            f"early, ensure you have at least three successful and relevant query unless nothing's working.\n\n"

            "## SOURCES\n"
            "1. **Vector DB**: Primary source chunks from Project Gutenberg philosophy texts\n"
            "2. **SEP**: Stanford Encyclopedia articles for conceptual overviews\n\n"

            "## PLANS"
            "- `long_term_plan`: Overall research plan, can span multiple iterations. Describes potential queries, "
            "sources and/or philosophers to include, and sequential tool calls.\n"
            "- `short_term_plan`: Specific plan for this iteration only. A one-or-two sentences description of what "
            "you aim to achieve this iteration. Use present continuous tense, such as 'I'm searching for...' or 'Looking "
            "for...' etc. Try to end in '...'.\n\n"

            "## SOURCE SELECTION\n"
            "**Vector DB**: Named philosophers, specific passages, textual evidence, author's development of ideas\n"
            "**SEP**: General concepts, movements, debates, overviews, multiple philosophers, scholarly interpretation\n"
            "**Both**: Idea genealogy, comparing sources with concepts, evidence + interpretation\n\n"

            "## QUERY RULES\n"
            "- MAX 3 queries for Project Gutenberg Vector DB (for this iteration, not total)\n"
            "- Max 1 queries for SEP (for this iteration, not total)\n"
            "- MIN 1 successful query total across all sources\n"
            "- NEVER repeat past queries\n"
            "- Vector DB: Contains both large and medium sized chunks, can be broad or specific. Can be searched both "
            "semantically and normally.\n"
            f"- SEP:\n\"\"\"{SEP_SEARCH_RULES}\"\"\"\n\n"

            "## END RESEARCH WHEN:\n"
            "- You have 3-4 RELEVANT results with relevant content (simple questions)\n"
            "- You have sufficient research (>=3 relevant results) to answer the question (medium-to-complex questions)\n"
            "- Past queries show ABSOLUTELY NECESSARY sources are unavailable/irrelevant\n\n"
            
            "## HOW TO END RESEARCH:\n"
            "Set all fields, `long_term_plan`, `short_term_plan`, `stanford_encyclopedia_queries`, "
            "`vector_db_queries`, and `ids_to_remove` to `null`\n\n"
            
            "## HOW TO REMOVE RESOURCES"
            "If there are specific sources or chunks that are irrelevant or unhelpful from past research, you can "
            "optionally include an `ids_to_remove` field with a list of their IDs to delete them from future consideration."
            "Prune unnecessary sources as needed.\n\n"
            
            "## IN YOUR REASONING/OUTPUT:\n"
            "Do NOT formulate the final response. Reason about the comprehensiveness of prior research and what is needed "
            "next, if anything. Consider how many iterations have occurred and the complexity of the question. You "
            "should rarely revise your long term plan, only doing so when needed. Revise your short term plan every"
            "iteration.\n\n"
            
            "## ADVICE:\n"
            "- If a user asks about previous research that you don't have, re-query for those sources.\n"
            "- Think about what sources (specific books, essays, etc.) would be best to answer the question.\n"
            "- If a resource is missing from the database, avoid re-querying it.\n"
            "- IF A PREVIOUS QUERY RETURNED IRRELEVANT INFORMATION, REWRITE THAT QUERY TO BE MORE FOCUSED / OTHERWISE "
            "FIX IT AND TRY AGAIN.\n\n"
            
            "## OUTPUT FORMAT (CRITICAL, MOST IMPORTANT!!!)\n"
            "CRITICAL: You are NOT allowed to use functions, tools, or XML tags like <tool_call>. Do not use the "
            "string 'repo_search' or any other external tool name. Respond ONLY in the following JSON format, with no "
            "other text as your message's content. It uses `#` comments which don't exist in real JSON, so don't add "
            "any comments of your own.\n"
            f"```json\n{SAMPLE_RESPONSE}```\n"
            "Your output and the JSON response should be as a message to the user, not in a tool call or anything else. "
            "Just respond as described in the normal 'content' section of your message. Do not invoke ANY tool calls, "
            "you don't have access to any.\n\n"
            
            "## STRICT RULES\n"
            "NEVER repeat queries.\n"
        )
    )
    research_history_message = SystemMessage(content=(
        f"YOUR LONG TERM PLAN:\n"
        f"\"{long_term_plan}\"\n\n"
        f"YOUR SHORT TERM PLAN (PREVIOUS ITERATION):\n"
        f"\"{short_term_plan}\"\n\n"
        f"PREVIOUS QUERIES + RESULTS:\n"
        f"```\n{stringify_query_results(query_results)}\n```\n\n"
    ))
    previous_conversation_message = SystemMessage(content=(
        "CONVERSATION HISTORY (for your context):\n```" + str(conversation[:-1]) + "\n```\n^ Previous conversation.\n"
    ))

    # Invoke LLM with structured output and retry parsing on invalid JSON
    model, reasoning_effort = RESEARCH_AGENT_MODEL_CONFIG["plan_research"]
    parser = JsonOutputParser()
    max_parse_attempts = 5
    attempt = 0
    result = None

    while attempt < max_parse_attempts:
        try:
            llm_output = safe_invoke(model, [previous_conversation_message, research_history_message, conversation[-1], system_msg], reasoning_effort)
            content = extract_content(llm_output)
            result = parser.parse(content)
            break
        except Exception as e:
            print(f"Failed to parse research plan: {e}")
            attempt += 1

    # If parsing failed after retries, end gracefully
    if result is None:
        return {"completed": True}

    # Prune research results if needed
    ids_to_remove = result.get("ids_to_remove", None)
    if ids_to_remove:
        _prune_research_results(state, ids_to_remove)

    # Check for research completion
    long_term_plan, short_term_plan, vector_db_queries, sep_queries = (
        result.get("long_term_plan"),
        result.get("short_term_plan"),
        result.get("vector_db_queries"),
        result.get("stanford_encyclopedia_queries")
    )
    if not long_term_plan and not short_term_plan and not vector_db_queries and not sep_queries:
        return {"completed": True}

    return {
        "long_term_plan": long_term_plan,
        "short_term_plan": short_term_plan,
        "vector_db_queries": vector_db_queries,
        "sep_queries": sep_queries,
        "research_iterations": research_iterations + 1,
    }
