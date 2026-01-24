import json

from ai.research_agent.schemas.QueryResult import QueryResult


def stringify_query_results(query_results: list[QueryResult]) -> str:
    """Convert a list of QueryResult objects into a formatted string."""

    output = "```\n\n"

    if query_results:
        for result in query_results:
            output += json.dumps(result, indent=4) + "\n\n"

    return output + "```"
