"""DuckDuckGo web search tool."""
from __future__ import annotations

from ddgs import DDGS


def web_search(query: str, max_results: int = 5) -> str:
    query = (query or "").strip()
    if not query:
        return "ERROR: empty search query."
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=max_results))
    except Exception as e:
        return f"ERROR: web search failed: {e}"

    if not results:
        return f"No results found for '{query}'."

    lines = []
    for i, r in enumerate(results, start=1):
        title = r.get("title", "(no title)")
        url = r.get("href") or r.get("url", "")
        body = (r.get("body") or "").strip().replace("\n", " ")
        if len(body) > 220:
            body = body[:220] + "…"
        lines.append(f"{i}. {title}\n   {url}\n   {body}")
    return "\n".join(lines)
