"""Wikipedia search tool — uses the public Wikipedia REST API, no key required."""

from __future__ import annotations

import logging

from pydantic_ai import RunContext

from loom.deps import LoomDeps

logger = logging.getLogger(__name__)


async def wikipedia_search(ctx: RunContext[LoomDeps], query: str) -> str:
    """Search Wikipedia and return the introductory summary of the best matching article.

    Args:
        query: The topic or question to search for.

    Returns:
        The article title and its introductory text (up to 1500 chars).
    """
    client = ctx.deps.http_client

    try:
        # Step 1: find the best matching article title.
        search_resp = await client.get(
            "https://en.wikipedia.org/w/api.php",
            params={
                "action": "query",
                "list": "search",
                "srsearch": query,
                "srlimit": 1,
                "format": "json",
            },
        )
        search_resp.raise_for_status()
        results = search_resp.json().get("query", {}).get("search", [])

        if not results:
            return f"No Wikipedia article found for '{query}'."

        title = results[0]["title"]

        # Step 2: fetch the plain-text intro extract.
        extract_resp = await client.get(
            "https://en.wikipedia.org/w/api.php",
            params={
                "action": "query",
                "prop": "extracts",
                "exintro": True,
                "explaintext": True,
                "titles": title,
                "format": "json",
            },
        )
        extract_resp.raise_for_status()
        pages = extract_resp.json().get("query", {}).get("pages", {})
        page = next(iter(pages.values()))
        extract: str = page.get("extract", "").strip()

        if not extract:
            return f"Wikipedia article '{title}' exists but has no extract."

        if len(extract) > 1500:
            extract = extract[:1500] + "…"

        logger.info("wikipedia_search | title=%r | chars=%d", title, len(extract))
        return f"[Wikipedia: {title}]\n\n{extract}"

    except Exception as exc:  # noqa: BLE001
        logger.warning("wikipedia_search | failed: %s", exc)
        return f"Wikipedia search failed: {exc}"
