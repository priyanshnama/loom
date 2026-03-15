"""LangGraph graph state.

``LoomState`` is a Pydantic ``BaseModel`` used as the state schema for the
``StateGraph``.  LangGraph calls ``model_validate`` on partial update dicts, so
every field must carry a sensible default.

The ``messages`` field uses the built-in ``add_messages`` reducer which appends
new messages and de-duplicates by ``id``, matching standard LangGraph behaviour.
"""

from __future__ import annotations

from typing import Annotated, Sequence

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field

from loom.models import LoomResponse


class LoomState(BaseModel):
    """Full mutable state threaded through every node in the Loom graph."""

    model_config = {"arbitrary_types_allowed": True}

    # Conversation history — the add_messages reducer handles append + dedup.
    messages: Annotated[Sequence[AnyMessage], add_messages] = Field(default_factory=list)

    # The raw user query for this turn.
    query: str = ""

    # Accumulated tool outputs, keyed by tool name.
    # e.g. {"web_search": ["..."], "book_flight": ["PNR: ABC"]}
    tool_outputs: dict[str, list[str]] = Field(default_factory=dict)

    # Latest structured response from the agent node.
    response: LoomResponse | None = None

    # How many agent→tool→agent loops have occurred this session.
    iteration_count: int = 0

    # Ceiling on refinement loops (overridable per-invocation via config).
    max_iterations: int = 3

    # Agent's chain-of-thought / reasoning text from the most recent turn.
    # Populated by agent_node from the LoomResponse.reasoning field.
    internal_monologue: str = ""

    # Cumulative count of tool errors across all turns in this thread.
    error_count: int = 0

    # Most recent tool error message, or None if the last tool call succeeded.
    # Cleared at the start of each agent_node run; re-set if errors occur.
    last_tool_error: str | None = None
