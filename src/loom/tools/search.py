"""Web search tool — swap the body for a real backend (Tavily, SerpAPI, etc.)."""

from __future__ import annotations

import asyncio
import random


async def web_search(query: str) -> str:
    """Search the web for up-to-date information relevant to the query.

    Args:
        query: The search query string.

    Returns:
        A short text snippet with the search result.
    """
    await asyncio.sleep(0.05)  # simulate network latency
    stubs = [
        f"[Result A] According to recent sources, '{query}' relates to topic X.",
        f"[Result B] Studies show that '{query}' is connected to phenomena Y and Z.",
        f"[Result C] Experts define '{query}' as a multi-faceted concept.",
    ]
    return random.choice(stubs)  # noqa: S311 — stub only
