"""Graph node implementations.

Pipeline
--------
  researcher_node  — Agent[LoomDeps, str]
      Uses tools (Wikipedia, calculator) to gather information.
      Stores plain-text findings in state.research_notes.

  synthesizer_node — Agent[LoomDeps, str]
      No tools. Reads state.research_notes and produces a LoomResponse JSON.
      This is the node that loops on low confidence (should_continue routing).

  respond_node
      No-op HITL checkpoint — the interrupt fires *before* this node.

Keeping node logic here (and graph wiring in graph.py) means you can unit-test
nodes by passing plain LoomState objects without constructing the full graph.
"""

from __future__ import annotations

import json
import logging
import re

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from pydantic_ai import Agent
from pydantic_ai.mcp import MCPServerStdio
from pydantic_ai.messages import ModelRequest, ToolReturnPart

from loom.config import settings
from loom.deps import LoomDeps, get_http_client
from loom.models import LoomResponse
from loom.state import LoomState

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# System prompts
# ---------------------------------------------------------------------------

_RESEARCHER_SYSTEM_PROMPT = """\
You are a research specialist. Your job is to gather information using your tools.

- Use wikipedia_search for factual, encyclopedic, or current-events questions.
- Use calculate for any mathematical computation.
- Call tools as many times as needed to collect thorough findings.
- Return comprehensive research notes — raw findings, not a polished answer.
"""

# ---------------------------------------------------------------------------
# Agent singletons — created once, reused across requests.
# ---------------------------------------------------------------------------

_researcher: Agent[LoomDeps, str] | None = None
_synthesizer: Agent[LoomDeps, str] | None = None


def _get_researcher() -> Agent[LoomDeps, str]:
    global _researcher
    if _researcher is None:
        from loom.tools import RESEARCHER_TOOLS  # noqa: PLC0415

        mcp_servers = (
            [MCPServerStdio("npx", args=["-y", "mcp-remote", "https://mcp-server.zomato.com/mcp"])]
            if settings.zomato_mcp_enabled
            else []
        )
        _researcher = Agent(
            settings.loom_model,
            system_prompt=_RESEARCHER_SYSTEM_PROMPT,
            tools=RESEARCHER_TOOLS,
            mcp_servers=mcp_servers,
        )
    return _researcher


def _get_synthesizer() -> Agent[LoomDeps, str]:
    global _synthesizer
    if _synthesizer is None:
        _synthesizer = Agent(
            settings.loom_model,
            system_prompt=settings.loom_system_prompt,
        )
    return _synthesizer


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _extract_thread_id(state: LoomState) -> str:
    for m in reversed(state.messages):
        if isinstance(m, HumanMessage) and m.id:
            return m.id.split(":")[0]
    return "default"


def _parse_response(text: str) -> LoomResponse:
    """Extract and validate a LoomResponse from the synthesizer's text output."""
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)

    try:
        return LoomResponse.model_validate(json.loads(text))
    except (json.JSONDecodeError, Exception):
        pass

    start = text.find("{")
    if start == -1:
        raise ValueError(f"No JSON object found in model output: {text!r}")

    depth, in_string, escape_next = 0, False, False
    for i, ch in enumerate(text[start:], start):
        if escape_next:
            escape_next = False
            continue
        if ch == "\\" and in_string:
            escape_next = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return LoomResponse.model_validate(json.loads(text[start : i + 1]))

    raise ValueError(f"Could not extract JSON from: {text[:200]!r}")


def _is_tool_error(content: str) -> bool:
    stripped = content.strip()
    if not stripped:
        return True
    lower = stripped.lower()
    return any(
        sig in lower
        for sig in ("error:", "[error]", "exception:", "traceback", "failed:", "tool failed", "timeout")
    )


def _collect_tool_results(
    result: object,
    iteration: int,
) -> tuple[list[ToolMessage], dict[str, list[str]], str | None, int]:
    """Walk pydantic-ai result messages and collect tool call data.

    Returns (extra_messages, new_tool_outputs, first_error, error_delta).
    """
    extra_messages: list[ToolMessage] = []
    new_tool_outputs: dict[str, list[str]] = {}
    detected_error: str | None = None
    error_delta = 0

    for msg in result.all_messages():  # type: ignore[union-attr]
        if not isinstance(msg, ModelRequest):
            continue
        for part in msg.parts:
            if not isinstance(part, ToolReturnPart):
                continue
            name = part.tool_name
            content = str(part.content)
            new_tool_outputs.setdefault(name, []).append(content)
            extra_messages.append(
                ToolMessage(content=content, tool_call_id=f"{name}-{iteration}")
            )
            logger.info("tool_called | name=%s", name)
            if detected_error is None and _is_tool_error(content):
                detected_error = f"Tool '{name}' returned: {content[:200]}"
                error_delta += 1

    return extra_messages, new_tool_outputs, detected_error, error_delta


# ---------------------------------------------------------------------------
# Node: researcher
# ---------------------------------------------------------------------------


async def researcher_node(state: LoomState) -> dict:
    """Run the researcher agent: uses tools to gather raw information.

    Returns:
    - ``research_notes``: plain-text findings from tools.
    - ``messages``: ToolMessages + AIMessage (researcher summary).
    - ``tool_outputs``: merged tool results.
    - ``error_count`` / ``last_tool_error``: populated if a tool failed.
    """
    deps = LoomDeps(thread_id=_extract_thread_id(state), http_client=get_http_client())
    prompt = f"Research the following query using your tools.\n\nQuery: {state.query}"

    logger.info("researcher_node | query=%r", state.query[:80])
    researcher = _get_researcher()
    async with researcher.run_mcp_servers():
        result = await researcher.run(prompt, deps=deps)

    extra_messages, new_tool_outputs, detected_error, error_delta = _collect_tool_results(
        result, iteration=0
    )

    merged: dict[str, list[str]] = dict(state.tool_outputs)
    for k, v in new_tool_outputs.items():
        merged.setdefault(k, []).extend(v)

    logger.info("researcher_node | notes_len=%d | tools_called=%d", len(result.output), len(new_tool_outputs))

    return {
        "research_notes": result.output,
        "messages": [*extra_messages, AIMessage(content=result.output)],
        "tool_outputs": merged,
        "last_tool_error": detected_error,
        "error_count": state.error_count + error_delta,
    }


# ---------------------------------------------------------------------------
# Node: synthesizer
# ---------------------------------------------------------------------------


async def synthesizer_node(state: LoomState) -> dict:
    """Synthesize the final answer from research notes.

    Reads ``state.research_notes`` and conversation history, then produces a
    structured ``LoomResponse`` JSON.  Has no tools — pure reasoning only.

    This is the node that loops on low confidence (should_continue routing).

    Returns:
    - ``messages``: final AIMessage.
    - ``response``: validated LoomResponse.
    - ``iteration_count``: incremented by 1.
    - ``internal_monologue``, ``last_tool_error``, ``error_count``.
    """
    deps = LoomDeps(thread_id=_extract_thread_id(state), http_client=get_http_client())

    prompt_parts: list[str] = []

    if state.last_tool_error:
        prompt_parts.append(
            f"⚠️  A research tool returned an error:\n    {state.last_tool_error}\n\n"
            "Work with the information available."
        )

    prior = [m for m in state.messages if isinstance(m, (HumanMessage, AIMessage))]
    if prior:
        history = "\n".join(
            f"{'User' if isinstance(m, HumanMessage) else 'Assistant'}: {m.content}"
            for m in prior
        )
        prompt_parts.append(f"Conversation so far:\n{history}")

    if state.research_notes:
        prompt_parts.append(f"Research findings:\n{state.research_notes}")

    prompt_parts.append(f"Query: {state.query}")
    prompt_parts.append(
        "Respond ONLY with a single valid JSON object — no markdown fences, no extra text:\n"
        '{"answer": "<your response>", "confidence_score": <float 0.0-1.0>, '
        '"sources": ["<source or citation>"], "reasoning": "<brief chain-of-thought>"}'
    )

    prompt = "\n\n".join(prompt_parts)
    logger.info(
        "synthesizer_node | iteration=%d | prompt_len=%d",
        state.iteration_count,
        len(prompt),
    )

    result = await _get_synthesizer().run(prompt, deps=deps)
    response = _parse_response(result.output)

    logger.info(
        "synthesizer_node | confidence=%.2f | needs_refinement=%s",
        response.confidence_score,
        response.needs_refinement,
    )

    return {
        "messages": [AIMessage(content=response.answer)],
        "response": response,
        "iteration_count": state.iteration_count + 1,
        "tool_outputs": state.tool_outputs,
        "internal_monologue": response.reasoning,
        "last_tool_error": None,
        "error_count": state.error_count,
    }


# ---------------------------------------------------------------------------
# Node: respond  (Human-in-the-Loop target)
# ---------------------------------------------------------------------------


async def respond_node(state: LoomState) -> dict:
    """Finalisation node — the HITL interrupt fires *before* this node.

    No-op: the response is already in state.response. This node exists solely
    as a named checkpoint for the interrupt mechanism.
    """
    logger.info(
        "respond_node | confidence=%.2f | iterations=%d",
        state.response.confidence_score if state.response else 0.0,
        state.iteration_count,
    )
    return {}
