"""Dependency injection container for Loom agents."""

from __future__ import annotations

from dataclasses import dataclass

import httpx

_client: httpx.AsyncClient | None = None


def get_http_client() -> httpx.AsyncClient:
    """Return a process-wide shared async HTTP client (created on first call)."""
    global _client
    if _client is None:
        _client = httpx.AsyncClient(timeout=15.0, follow_redirects=True)
    return _client


@dataclass
class LoomDeps:
    """Injected into every agent run — available to all contextual tools via RunContext."""
    thread_id: str
    http_client: httpx.AsyncClient
