import requests
from bs4 import BeautifulSoup

from ai.research_agent.nodes.sources.extract_text import extract_text


def _search_sep(query, limit=100):
    """Search SEP and return list of results."""

    url = "https://plato.stanford.edu/search/searcher.py"
    params = {'query': query}
    headers = {'User-Agent': 'Cogito Research Bot (wbc008@bucknell.edu)'}

    response = requests.get(url, params=params, headers=headers)
    soup = BeautifulSoup(response.text, 'html.parser')

    results = []
    i = 0
    for result in soup.find_all('div', class_='result_listing'):
        if i >= limit:
            break
        i += 1

        title_elem = result.find('div', class_='result_title')
        snippet_elem = result.find('div', class_='result_snippet')

        if title_elem and title_elem.find('a'):
            link = title_elem.find('a')['href']
            title = title_elem.get_text(strip=True)
            snippet = snippet_elem.get_text(strip=True) if snippet_elem else ""

            results.append({
                'title': title,
                'url': link,
                'snippet': snippet
            })

    return results

def _get_article_text(url):
    """Download and parse a SEP article, return (text, citation)."""

    headers = {'User-Agent': 'Cogito Research Bot (wbc008@bucknell.edu)'}
    response = requests.get(url, headers=headers)
    soup = BeautifulSoup(response.text, 'html.parser')

    # Extract citation metadata
    citation = {}

    title_meta = soup.find('meta', property='citation_title')
    if title_meta:
        citation['title'] = title_meta.get('content', '')

    # Get all authors (there can be multiple)
    authors = []
    author_metas = soup.find_all('meta', property='citation_author')
    for author_meta in author_metas:
        authors.append(author_meta.get('content', ''))
    citation['authors'] = authors

    date_meta = soup.find('meta', property='citation_publication_date')
    if date_meta:
        citation['publication_date'] = date_meta.get('content', '')

    # Format citation string
    author_str = ', '.join(authors) if authors else 'Unknown'
    citation_str = f"{author_str}. \"{citation.get('title', 'Unknown')}\" Stanford Encyclopedia of Philosophy ({citation.get('publication_date', 'n.d.')}). {url}"

    # Extract main article text
    main_content = soup.find('div', id='main-text')  # adjust selector as needed

    if main_content:
        pieces = main_content.find_all(['p', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6'])
        text = '\n\n'.join(p.get_text(strip=True) for p in pieces)
        return text, citation_str

    return None, citation_str

def query_sep(queries: list[str], last_user_msg: str):
    """Query Stanford Encyclopedia of Philosophy and return extracted texts with citations."""

    res = []
    limit = 1

    for query in queries:
        search_results = _search_sep(query, limit=limit)

        for result in search_results:
            article_text, citation = _get_article_text(result['url'])
            if article_text:
                article_token_limit = 5000
                if len(article_text.split(" ")) / 3 > article_token_limit:
                    article_text, _ = extract_text(article_text, last_user_msg)

                res.append({
                    'title': result['title'],
                    'article_text': article_text,
                    'citation': "Stanford Encyclopedia of Philosophy; " + citation
                })

    return res
