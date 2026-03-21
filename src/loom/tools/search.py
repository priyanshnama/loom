"""Web search tool — swap the body for a real backend (Tavily, SerpAPI, etc.)."""

from __future__ import annotations

import asyncio
import logging
import random

logger = logging.getLogger(__name__)


async def web_search(query: str, thread_id: str = "default") -> str:
    """Search the web for up-to-date information relevant to the query.

    Results are automatically stored in the Neo4j knowledge graph so future
    queries on the same thread can retrieve them without re-fetching.

    Args:
        query: The search query string.
        thread_id: The active session thread ID (injected by the caller).

    Returns:
        A short text snippet with the search result.
    """
    await asyncio.sleep(0.05)  # simulate network latency
    stubs = [
        f"[Result A] According to recent sources, '{query}' relates to topic X.",
        f"[Result B] Studies show that '{query}' is connected to phenomena Y and Z.",
        f"[Result C] Experts define '{query}' as a multi-faceted concept.",
    ]
    result = random.choice(stubs)  # noqa: S311 — stub only

    # Persist to knowledge graph (best-effort; never fail the tool call).
    try:
        from loom.knowledge_graph import store_search_result  # noqa: PLC0415

        await store_search_result(query=query, result=result, thread_id=thread_id)
    except Exception as exc:  # noqa: BLE001
        logger.warning("web_search | kg write failed: %s", exc)

    return result
