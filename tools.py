import os
import re

import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from tavily import TavilyClient
from trafilatura import extract as extract_text

load_dotenv()


def web_search(query: str) -> str:
    """Search Tavily and return a readable summary of the results."""
    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key:
        raise RuntimeError("TAVILY_API_KEY is not set in the environment")

    results = TavilyClient(api_key=api_key).search(query=query, max_results=5)
    output = []
    for result in results.get("results", []):
        output.append(
            f"Title: {result.get('title', '')}\n"
            f"URL: {result.get('url', '')}\n"
            f"Snippet: {result.get('content', '')[:300]}"
        )

    return "\n----\n".join(output) or "No results found."


def scrape_url(url: str) -> str:
    """Fetch an article page and return clean text content for analysis."""
    if not url:
        return "No URL provided."

    try:
        response = requests.get(
            url,
            timeout=20,
            headers={
                "User-Agent": "Mozilla/5.0 (compatible; MultiAgentResearchBot/1.0; +https://example.com)"
            },
        )
        response.raise_for_status()
        html = response.text
    except requests.RequestException as exc:
        return f"Unable to fetch the page: {exc}"

    extracted = extract_text(html, include_links=True, include_formatting=False)
    if extracted:
        text = re.sub(r"\s+", " ", extracted).strip()
        return text[:8000]

    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "noscript", "svg"]):
        tag.decompose()

    text = soup.get_text(" ", strip=True)
    cleaned = re.sub(r"\s+", " ", text).strip()
    return cleaned[:8000] if cleaned else "No readable content found on the page."
