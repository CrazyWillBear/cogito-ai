import time

from langchain_core.messages import SystemMessage, HumanMessage
from pydantic import BaseModel

from ai.models.model_config import MODEL_CONFIG
from ai.research_agent.schemas.ResearchAgentState import ResearchAgentState
from dbs.QueryAndFilterSchemas import QueryAndFilters


class QueryList(BaseModel):
    """Output schema for LLM query generation."""

    vector_db_queries: list[QueryAndFilters]
    stanford_encyclopedia_queries: list[str]


# Sample response to fit schema
SAMPLE_RESPONSE = \
"""
{
  "vector_db_queries": [
    # this is a query
    {
      "query": "What is a covenant?",
      "filters": {
        "author": "Thomas Hobbes",
        "source_title": "Leviathan"
      }
    },
    # this is a second query
    {
      "query": "determinism and moral responsibility",
      "filters": null
    }
  ],
  "stanford_encyclopedia_queries": [
    "free will",    # this is a query
    "determinism"   # this is a second query
  ]
}
"""

SEP_SEARCH_RULES = \
"""
Fuzzy search:
If you are not sure how to spell a search term, you can try a fuzzy search, which will find inexact matches. Simply add a tilde (‘~’) after the term.
Example: results for the search Liebnitz~ include documents matching the term “Leibniz”.
Required terms:
By default, search results will contain entries that match any of the search terms. You can put a plus sign in front of each term that must be matched.
Example: results for the search leibniz locke might mention Leibniz or Locke, but not necessarily both. If you want results that must mention Leibniz and Locke, you can use the search +leibniz +locke.
Excluded terms:
If you want to make sure none of the search results include some term, you can put a minus sign in front of that term.
Example: results for the search +leibniz -locke are documents mentioning Leibniz which do not also mention Locke.
Combining terms:
You can formulate a complex query by combining terms using plus signs and minus signs as described above. Alternatively, you can use the expressions “AND”, “OR”, and “NOT” in all caps.
Example: results for the search leibniz NOT locke are documents mentioning Leibniz which do not also mention Locke. Note this is equivalent to the search +leibniz -locke
Note: you can use parentheses to group terms. So if you want to search for entries which mention Leibniz or Newton but don't mention Locke, you could do the search (leibniz OR newton) NOT locke.
Exact phrase:
If you want to search for an exact phrase, put the complete phrase in double quotes.
Example: results for the search "the world is all that is the case" are those documents including that exact string of words.
Note: this will not always work as expected because short, common words are not indexed. For instance, the search "world all case" will find the same results as the given example – but the text shown with the result will be less helpful.
Title search:
If you want to search for a title that contains a word, type “title:”, followed by the word.
Example: results for the search title:Descartes are those documents in which the word “Descartes” occurs in the title.
Author search:
If you want to search for entries by author name, type “author:”, followed by the name.
Example: results for the search author:smith are those documents written by authors named Smith.
Proximity phrase:
If you want to search for words that occur close to each other, put the words in double quotes followed by a tilde and how far apart the words may be.
Example: results for the search "world case"~5 are those documents in which “world” is followed within five words by “case”. If the number equals 1, this is the same as searching for an exact phrase.
Case:
Searches are not case sensitive.
Example: the search leibniz and the search Leibniz will return the same results.
Wildcard searches:
An asterisk (‘*’) can be added as a wildcard symbol in the middle or at the end of a word or partial word. The asterisk will match any letter or series of letters in a single word.
Example: results for the search logic* are those documents in which the word “logic” or the word “logical” or the word “logicism”, etc., occurs.
Example: results for the search title:contract* are those documents in which the word “contract” or the word “contractarianism” or the word “contractualism”, etc., occurs in the title.
More complex searches:
The above search operations can be combined.
Example: results for the search title:social title:political are those documents in which the word “social” or the word “political” occurs in the title.
Example: results for the search +semantics +logic -title:logic* are those documents which mention logic and semantics but whose title does not include the word “logic” nor any word that begins with “logic”.
"""

def write_queries(state: ResearchAgentState):
    """
    Write a vector DB query based on the user's message and previous research.
    Generates a structured query with optional filters for author and source title.
    """

    # Start timing and log
    print("::Writing queries...", end="", flush=True)
    start = time.perf_counter()

    # Get configured model
    model, reasoning = MODEL_CONFIG["write_queries"]
    structured_model = model.with_structured_output(QueryList)

    # Extract graph state variables
    feedback = state.get("queries_feedback", "No feedback yet.")
    conversation = state.get("conversation", {})

    # Construct prompt (system message and user message)
    system_msg = SystemMessage(content=(
        "You are a semantic-search assistant for philosophical research. Your task is to analyze "
        "user questions and generate optimal search queries for two complementary sources:\n\n"

        "## DATA SOURCES\n"
        "1. **Vector DB**: Contains large chunks from Project Gutenberg philosophy primary sources\n"
        "2. **Stanford Encyclopedia of Philosophy (SEP)**: High-level conceptual articles\n\n"

        "## OUTPUT FORMAT\n"
        "Return a JSON object with this exact structure:\n"
        f"{SAMPLE_RESPONSE}\n\n"

        "## DECISION FRAMEWORK\n"
        "Use this logic to determine which sources to query:\n\n"

        "**Use Vector DB when:**\n"
        "- ALWAYS when a user mentions a philosopher or source by name\n"
        "- User asks about specific passages, arguments, or ideas from named philosophers\n"
        "- ALWAYS when the question requires direct textual evidence from primary sources\n"
        "- Tracing how a specific author develops an idea across their work\n"
        "- User explicitly names authors or texts (put author names in 'filters' field, NOT in query text)\n\n"

        "**Use Stanford Encyclopedia when:**\n"
        "- User asks about philosophical concepts, movements, or debates generally\n"
        "- Question needs overview of how multiple philosophers approach a topic\n"
        "- Exploring contemporary scholarly interpretation of ideas\n"
        "- No specific primary source is implied or needed\n\n"

        "**Use BOTH when:**\n"
        "- Question asks for genealogy/history of an idea (Vector DB for primary sources, SEP for overview)\n"
        "- Comparing specific textual passages with broader conceptual understanding\n"
        "- User wants both evidence and interpretation\n\n"

        "## QUERY CRAFTING GUIDELINES\n\n"

        "**For Vector DB queries:**\n"
        "- The chunks are large, so keep queries broad enough to capture relevant passages\n"
        "- Focus on concepts, ideas, and arguments (not just keywords)\n"
        "- Use natural language that would appear in philosophical texts\n"
        "- Never include author names in query text—use 'filters' field instead\n"
        "- Example: 'the social contract and natural rights' not 'Locke contract'\n"
        "- For explorations of philosopher's ideas or definitions, write enough queries to cover all aspects of their "
        "definitions or ideas.\n"

        "**For SEP queries:**\n"
        "- Use concise, encyclopedia-appropriate terminology\n"
        "- Default to ONE query for >85% of questions\n"
        "- Only use multiple queries when the question has genuinely distinct facets\n"
        "- Follow these SEP-specific rules:\n"
        f"{SEP_SEARCH_RULES}\n\n"

        "## LIMITS TO QUERIES:\n"
        "- Unlimited Vector DB queries (try to limit to 4-5 max)\n"
        "- MAX 2 SEP queries\n\n"

        "## IMPORTANT REMINDERS\n"
        "- Set fields to null if you're not using that source (don't omit the field)\n"
        "- Author names belong in 'filters', never in 'query' text\n"
        "- More queries ≠ better results. Be judicious.\n"
        "- Your queries should reflect what a philosophy researcher would search for\n"
        "- Output ONLY valid JSON, no additional text\n"
    ))

    conv_summary = conversation.get("context", "No prior context.")
    last_message = conversation.get("last_user_message", "No last user message")
    user_msg = HumanMessage(content=(
        f"Conversation:\n{conv_summary}\n\n"
        f"User's last message:\n{last_message}\n\n"
        f"Previous queries feedback (if any):\n{feedback}"
    ))

    # Invoke LLM with structured output
    result = structured_model.invoke([system_msg, user_msg], reasoning={"effort": reasoning})

    # Stop timing and log
    end = time.perf_counter()
    print(f"\r\033[K::Wrote queries in {end - start:.2f}s")

    if not result.vector_db_queries and not result.stanford_encyclopedia_queries:
        return {"broad_question": True}

    return {"vector_db_queries": result.vector_db_queries, "sep_queries": result.stanford_encyclopedia_queries}
