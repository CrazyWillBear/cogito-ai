"""Fair warning, all of the async logic was written by AI and is slightly messy."""

import asyncio
import json
import uuid

import aiohttp
from bs4 import BeautifulSoup
from langchain_core.messages import SystemMessage, AnyMessage

from ai.models.util import extract_content, safe_invoke
from ai.research_agent.model_config import RESEARCH_AGENT_MODEL_CONFIG
from ai.research_agent.schemas.Citation import Citation
from ai.research_agent.schemas.QueryResult import QueryResult


async def _search_sep_async(query, limit=1):
    """Search SEP and return list of results (async)."""
    url = "https://plato.stanford.edu/search/searcher.py"
    params = {"query": query}
    headers = {"User-Agent": "Cogito Research Bot (wbc008@bucknell.edu)"}

    async with aiohttp.ClientSession() as session:
        async with session.get(url, params=params, headers=headers,
                               timeout=aiohttp.ClientTimeout(total=10)) as response:
            text = await response.text()
            soup = BeautifulSoup(text, "html.parser")

            results = []
            i = 0
            for result in soup.find_all("div", class_="result_listing"):
                if i >= limit:
                    break
                i += 1

                title_elem = result.find("div", class_="result_title")
                snippet_elem = result.find("div", class_="result_snippet")

                if title_elem and title_elem.find("a"):
                    link = title_elem.find("a")["href"]
                    title = title_elem.get_text(strip=True)
                    snippet = snippet_elem.get_text(strip=True) if snippet_elem else ""

                    results.append({"title": title, "url": link, "snippet": snippet})

            return results


async def _extract_sections_async(url):
    """Extract all sections from a SEP article with their headers and content (async)."""
    headers = {"User-Agent": "Cogito Research Bot (wbc008@bucknell.edu)"}

    async with aiohttp.ClientSession() as session:
        async with session.get(url, headers=headers, timeout=aiohttp.ClientTimeout(total=10)) as response:
            text = await response.text()
            soup = BeautifulSoup(text, "html.parser")

            # Extract citation metadata
            citation: Citation = {"title": "", "authors": [], "source": ""}
            title_meta = soup.find("meta", property="citation_title")
            if title_meta:
                citation["title"] = title_meta.get("content", "")

            authors = []
            author_metas = soup.find_all("meta", property="citation_author")
            for author_meta in author_metas:
                authors.append(author_meta.get("content", ""))
            citation["authors"] = authors

            date_meta = soup.find("meta", property="citation_publication_date")
            if date_meta:
                citation["source"] = "Stanford Encyclopedia of Philosophy - " + date_meta.get("content", "")

            # Extract sections
            main_content = soup.find("div", id="main-text")
            if not main_content:
                return [], citation

            sections = []
            current_section = None
            current_level = 0

            # Get all children of main-text in order
            for elem in main_content.children:
                # Skip non-tag elements (like NavigableString)
                if not hasattr(elem, "name"):
                    continue

                if elem.name in ["h1", "h2", "h3", "h4", "h5", "h6"]:
                    elem_level = int(elem.name[1])

                    # Only start a new section if this header is same level or higher than current section
                    if current_section is None or elem_level <= current_level:
                        # Save previous section if it exists
                        if current_section:
                            sections.append(current_section)

                        # Start new section
                        current_section = {"header": elem.get_text(strip=True), "level": elem.name, "content": []}
                        current_level = elem_level
                    else:
                        # This is a sub-header, add it as formatted content
                        sub_header_text = f"### {elem.get_text(strip=True)}"
                        current_section["content"].append(sub_header_text)

                elif current_section is not None:
                    # Add any non-header element's text to current section
                    text = elem.get_text(strip=True)
                    if text:
                        current_section["content"].append(text)

            # Add the last section
            if current_section:
                sections.append(current_section)

            return sections, citation


def _select_relevant_sections(sections, conversation, article_title):
    """Use LLM to determine which sections are relevant to the user's query."""
    if not sections:
        return []

    # Create a list of section headers with their identifiers extracted from the header text
    section_headers = []
    for i, section in enumerate(sections):
        header = section["header"]
        # Try to extract section number from header (e.g., "2.1 Reductionist vs. Non-Reductionist")
        # Most SEP articles format sections like "2.1 Title" or "2. Title"
        section_id = header.split()[0] if header and header[0].isdigit() else str(i + 1)
        section_headers.append(f"{section_id}. {header}")

    system_msg = SystemMessage(content=(
        "Determine which sections are most relevant to the user's last message:\n"
        
        f"Article Title: {article_title}\n"
        f"Section Headers:\n" + f"{chr(10).join(section_headers)}\n\n"
        "Respond with ONLY an array of section identifiers as strings that are relevant. Only include at "
        "most 5 section identifiers. Do NOT include any other text and DO NOT make any tool calls.\n\n"
        "This exact format: `[\"1\", \"2.1\", \"3.4\"]`. Nothing other than the brackets and section identifiers in between them.\n"
    ))
    conversation_context_message = SystemMessage(content=(
        "Here is the conversation so far (most recent messages last):\n\n"
        f"{''.join([f'- {msg.content}\n' for msg in conversation[-5:]])}\n"
    ))

    try:
        model = RESEARCH_AGENT_MODEL_CONFIG.get("extract_text")
        content = extract_content(safe_invoke(model, [conversation_context_message, system_msg]))

        # Try to extract JSON from the response
        # Remove markdown code blocks if present
        content = content.replace("```json", "").replace("```", "").strip()

        relevant_identifiers = json.loads(content)

        # Match identifiers to sections
        relevant_sections = []
        for identifier in relevant_identifiers:
            identifier_str = str(identifier).strip()

            # Find sections that match this identifier
            for section in sections:
                header = section["header"]
                # Check if header starts with this identifier followed by space or dot
                if header.startswith(identifier_str + " ") or header.startswith(identifier_str + "."):
                    relevant_sections.append(section)
                    break
                # Also check if it's just the identifier (for cases like "1 Introduction")
                header_first = header.split()[0]
                if header_first == identifier_str or header_first.split(".")[0] == identifier_str.split(".")[0]:
                    relevant_sections.append(section)
                    break

        return relevant_sections

    except Exception as e:
        print(f"Error selecting relevant sections: {e}")
        # Fallback: return first few sections if LLM call fails
        return sections[:3]


def _format_section_text(section):
    """Format a section with its header and content."""
    header = section["header"]
    content = "\n\n".join(section["content"])
    return f"## {header}\n\n{content}"


async def _process_article_async(result, conversation):
    """Process a single article: extract sections, select relevant ones, return list of (text, citation) tuples (async)."""
    sections, base_citation = await _extract_sections_async(result["url"])

    if not sections:
        return []

    # Use LLM to select relevant sections (this is still sync, but runs in executor)
    # We run this in the default executor to avoid blocking the event loop
    loop = asyncio.get_event_loop()
    relevant_sections = await loop.run_in_executor(
        None,
        _select_relevant_sections,
        sections,
        conversation,
        result["title"]
    )

    if not relevant_sections:
        return []

    # Create a tuple for each relevant section
    tuples = []
    for section in relevant_sections:
        section_text = _format_section_text(section)
        # Add section header to citation
        base_citation["section"] = section["header"]
        tuples.append((section_text, base_citation))

    return tuples


async def _query_sep_async(queries: list[str], conversation: list[AnyMessage]) -> list[QueryResult]:
    """Query Stanford Encyclopedia of Philosophy and return list of QueryResult (async)."""
    limit = 1

    # Run all searches concurrently
    search_tasks = [_search_sep_async(query, limit=limit) for query in queries]
    all_search_results = await asyncio.gather(*search_tasks, return_exceptions=True)

    # Build a list of (query, search_result) jobs
    jobs: list[tuple[str, dict]] = []
    for query, search_results in zip(queries, all_search_results):
        # Handle exceptions from search
        if isinstance(search_results, Exception):
            print(f"Error searching SEP for query '{query}': {search_results}")
            continue

        for result in search_results:
            jobs.append((query, result))

    # Process all articles concurrently
    article_tasks = [_process_article_async(result, conversation) for (query, result) in jobs]
    all_article_tuples = await asyncio.gather(*article_tasks, return_exceptions=True)

    # Map from query -> list[(text, citation)]
    per_query_tuples: dict[str, list[tuple[str, str]]] = {}

    for (query, result), article_tuples in zip(jobs, all_article_tuples):
        # Handle exceptions from article processing
        if isinstance(article_tuples, Exception):
            print(f"Error processing article for query '{query}': {article_tuples}")
            continue

        if not article_tuples:
            continue

        per_query_tuples.setdefault(query, []).extend(article_tuples)

    # Flatten into QueryResult objects with correct query association
    results: list[QueryResult] = []
    for query, tuples in per_query_tuples.items():
        for section_tuple in tuples:
            results.append(QueryResult(id=int(uuid.uuid4()), query=query, source="SEP", result=section_tuple))

    return results


def query_sep(queries: list[str], conversation: list[AnyMessage]) -> list[QueryResult]:
    """Query Stanford Encyclopedia of Philosophy and return list of QueryResult.

    Each QueryResult will have:
    - query: the originating query string for this result
    - result: a (text, citation) tuple for a relevant SEP section

    This is a synchronous wrapper around the async implementation.
    """

    return asyncio.run(_query_sep_async(queries, conversation))