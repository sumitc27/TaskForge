"""URL reader tool: fetch a page and return clean, truncated text."""
from __future__ import annotations

import httpx
from bs4 import BeautifulSoup

_MAX_CHARS = 6000
_HEADERS = {
    "User-Agent": "TaskForge/1.0 (autonomous research agent; +https://github.com/taskforge) httpx",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
}


def read_url(url: str, timeout: int = 20) -> str:
    url = (url or "").strip()
    if not url.startswith(("http://", "https://")):
        return f"ERROR: not a valid http(s) URL: '{url}'"
    try:
        with httpx.Client(follow_redirects=True, timeout=timeout, headers=_HEADERS) as c:
            resp = c.get(url)
            resp.raise_for_status()
    except httpx.HTTPStatusError as e:
        return f"ERROR: {url} returned HTTP {e.response.status_code}."
    except Exception as e:
        return f"ERROR: could not fetch {url}: {e}"

    ctype = resp.headers.get("content-type", "")
    if "html" not in ctype and "text" not in ctype:
        return f"ERROR: {url} is not a readable text/HTML page (type: {ctype})."

    soup = BeautifulSoup(resp.text, "html.parser")
    for tag in soup(["script", "style", "nav", "footer", "header", "noscript", "form"]):
        tag.decompose()
    text = " ".join(soup.get_text(separator=" ").split())
    if not text:
        return f"ERROR: no readable text extracted from {url}."

    title = soup.title.string.strip() if soup.title and soup.title.string else url
    truncated = text[:_MAX_CHARS]
    suffix = "" if len(text) <= _MAX_CHARS else f"\n…[truncated, {len(text)} chars total]"
    return f"TITLE: {title}\nURL: {url}\n\n{truncated}{suffix}"
