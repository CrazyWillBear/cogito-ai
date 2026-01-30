import json

from ai.research_agent.schemas.QueryResult import QueryResult


def stringify_query_results(query_results: list[QueryResult]) -> str:
    """Convert a list of QueryResult objects into a formatted string."""

    output = ""

    if query_results:
        for result in query_results:
            output += "```\n" + json.dumps(result, indent=4) + "\n```\n\n"

    return output
