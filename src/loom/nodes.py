"""Graph node implementations.

Each function in this module is a LangGraph node: it receives a ``LoomState``
snapshot and returns a *partial* state dict whose fields will be merged back by
the graph runtime.

Keeping node logic here (and graph wiring in ``graph.py``) means you can unit-
test nodes by passing plain ``LoomState`` objects without constructing the full
graph.
"""

from __future__ import annotations

import json
import logging
import re

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from pydantic_ai import Agent
from pydantic_ai.messages import ModelRequest, ToolReturnPart

from loom.config import settings
from loom.models import LoomResponse
from loom.state import LoomState

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Pydantic AI agent — returns plain text; we validate to LoomResponse below.
#
# We intentionally avoid output_type= here because pydantic-ai 1.x's internal
# tool-calling extraction for structured output is unreliable across Claude
# model versions (the model may call the output tool with an empty payload).
# Instead we ask Claude to return JSON directly and validate with Pydantic,
# which is simpler and fully version-agnostic.
# ---------------------------------------------------------------------------

_agent: Agent[None, str] | None = None


def _get_agent() -> Agent[None, str]:
    """Return the singleton pydantic-ai Agent, creating it on first call.

    Tools are registered here so pydantic-ai sends their schemas to the model
    on every run.  Adding a new tool = add it to loom/tools/__init__.py only.
    """
    global _agent
    if _agent is None:
        from loom.tools import TOOLS  # noqa: PLC0415 — deferred to avoid circular import

        _agent = Agent(
            settings.loom_model,
            system_prompt=settings.loom_system_prompt,
            tools=TOOLS,
        )
    return _agent


def _parse_response(text: str) -> LoomResponse:
    """Extract and validate a LoomResponse from the model's text output.

    Strategy (in order):
    1. Strip markdown fences then try a direct ``json.loads``.
    2. Fall back to a balanced-brace scanner that finds the first complete
       ``{...}`` object, correctly ignoring nested braces inside string values
       and inside code examples embedded in the answer.
    """
    text = text.strip()

    # Strip optional ```json ... ``` fences.
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)

    try:
        data = json.loads(text)
        return LoomResponse.model_validate(data)
    except (json.JSONDecodeError, Exception):
        pass

    # Balanced-brace scan: walk character by character tracking depth so we
    # stop at the first *complete* top-level JSON object, not the last '}'.
    start = text.find("{")
    if start == -1:
        raise ValueError(f"No JSON object found in model output: {text!r}")

    depth = 0
    in_string = False
    escape_next = False

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
                candidate = text[start : i + 1]
                data = json.loads(candidate)
                return LoomResponse.model_validate(data)

    raise ValueError(f"Could not extract a complete JSON object from model output: {text[:200]!r}")


def _is_tool_error(content: str) -> bool:
    """Return True when a tool output looks like an error or empty result.

    Convention: tools should raise exceptions for hard errors (pydantic-ai
    converts those to error strings automatically).  This heuristic also
    catches empty returns and common error-prefix patterns.
    """
    stripped = content.strip()
    if not stripped:
        return True
    lower = stripped.lower()
    error_signals = (
        "error:",
        "[error]",
        "exception:",
        "traceback",
        "failed:",
        "tool failed",
        "not found",
        "timeout",
    )
    return any(sig in lower for sig in error_signals)


# ---------------------------------------------------------------------------
# Node: agent
# ---------------------------------------------------------------------------


async def agent_node(state: LoomState) -> dict:
    """Call the Pydantic AI agent and update state with its structured response.

    Self-correction: if ``state.last_tool_error`` is set (i.e. the previous
    turn's tool call returned an error), the prompt is prefixed with
    instructions telling the model to reformulate its query before retrying.

    pydantic-ai handles tool selection and execution internally within a single
    agent.run() call.  After it returns, we inspect all tool return values; if
    any look like an error we populate ``last_tool_error`` so the routing edge
    can loop us back here on the next turn.

    Returns a partial state dict containing:
    - ``messages``: ToolMessages for each tool call + final AIMessage.
    - ``response``: the validated ``LoomResponse``.
    - ``iteration_count``: incremented by 1.
    - ``tool_outputs``: merged dict of all tool results from this turn.
    - ``internal_monologue``: reasoning text from this turn's response.
    - ``last_tool_error``: error string if a tool failed, else None.
    - ``error_count``: incremented if a new tool error was detected.
    """
    # ------------------------------------------------------------------
    # Fetch knowledge-graph context for this query (best-effort).
    # ------------------------------------------------------------------
    kg_snippets: list[str] = []
    if state.query:
        try:
            from loom.knowledge_graph import query_knowledge  # noqa: PLC0415

            # Infer thread_id from the last HumanMessage id or fall back to "default".
            thread_id = "default"
            for m in reversed(state.messages):
                if isinstance(m, HumanMessage) and m.id:
                    thread_id = m.id.split(":")[0]
                    break
            kg_snippets = await query_knowledge(state.query, thread_id=thread_id)
        except Exception as exc:  # noqa: BLE001
            logger.warning("agent_node | kg query failed: %s", exc)

    # ------------------------------------------------------------------
    # Build prompt
    # ------------------------------------------------------------------
    prompt_parts: list[str] = []

    # Self-correction preamble — shown when a previous tool call failed.
    if state.last_tool_error:
        prompt_parts.append(
            f"⚠️  Your previous tool call returned an error:\n"
            f"    {state.last_tool_error}\n\n"
            "Please reformulate your search query or choose a different tool "
            "to gather the information you need, then try again."
        )
        logger.warning(
            "agent_node | self-correction | error_count=%d | error=%s",
            state.error_count,
            state.last_tool_error,
        )

    prior = [m for m in state.messages if isinstance(m, (HumanMessage, AIMessage))]
    if prior:
        history_lines = []
        for msg in prior:
            role = "User" if isinstance(msg, HumanMessage) else "Assistant"
            history_lines.append(f"{role}: {msg.content}")
        prompt_parts.append("Conversation so far:\n" + "\n".join(history_lines))

    if kg_snippets:
        prompt_parts.append(
            "Relevant knowledge from prior searches:\n"
            + "\n".join(f"  - {s}" for s in kg_snippets)
        )

    if state.tool_outputs:
        lines = []
        for tool_name, outputs in state.tool_outputs.items():
            for out in outputs:
                lines.append(f"  [{tool_name}] {out}")
        prompt_parts.append("Tool outputs from previous turns:\n" + "\n".join(lines))

    prompt_parts.append(f"Query: {state.query}")

    # Always append the output-format contract here (not in system.md) so that
    # system.md stays a pure persona/instructions file.
    prompt_parts.append(
        "Respond ONLY with a single valid JSON object — no markdown fences, no extra text:\n"
        '{"answer": "<your response>", "confidence_score": <float 0.0-1.0>, '
        '"sources": ["<source or citation>"], "reasoning": "<brief chain-of-thought>"}'
    )

    prompt = "\n\n".join(prompt_parts)

    logger.info(
        "agent_node | iteration=%d | prompt_len=%d",
        state.iteration_count,
        len(prompt),
    )

    result = await _get_agent().run(prompt)
    response = _parse_response(result.output)

    logger.info(
        "agent_node | confidence=%.2f | needs_refinement=%s",
        response.confidence_score,
        response.needs_refinement,
    )

    # ------------------------------------------------------------------
    # Collect tool outputs; detect any errors in this run.
    # ------------------------------------------------------------------
    new_tool_outputs: dict[str, list[str]] = {}
    extra_messages: list[ToolMessage] = []
    detected_error: str | None = None
    new_error_count = state.error_count

    for msg in result.all_messages():
        if not isinstance(msg, ModelRequest):
            continue
        for part in msg.parts:
            if not isinstance(part, ToolReturnPart):
                continue
            tool_name = part.tool_name
            content = str(part.content)
            new_tool_outputs.setdefault(tool_name, []).append(content)
            extra_messages.append(
                ToolMessage(
                    content=content,
                    tool_call_id=f"{tool_name}-{state.iteration_count}",
                )
            )
            logger.info("agent_node | tool_called=%s", tool_name)

            # Self-correction: flag the first error we encounter.
            if detected_error is None and _is_tool_error(content):
                detected_error = f"Tool '{tool_name}' returned: {content[:200]}"
                new_error_count += 1
                logger.warning(
                    "agent_node | tool_error detected | tool=%s | content=%s",
                    tool_name,
                    content[:100],
                )

    # Merge new outputs into accumulated state.
    merged_outputs: dict[str, list[str]] = dict(state.tool_outputs)
    for k, v in new_tool_outputs.items():
        merged_outputs.setdefault(k, []).extend(v)

    return {
        "messages": [*extra_messages, AIMessage(content=response.answer)],
        "response": response,
        "iteration_count": state.iteration_count + 1,
        "tool_outputs": merged_outputs,
        "internal_monologue": response.reasoning,
        "last_tool_error": detected_error,   # None clears it for next turn
        "error_count": new_error_count,
        "kg_context": kg_snippets,
    }


# ---------------------------------------------------------------------------
# Node: respond  (Human-in-the-Loop target)
# ---------------------------------------------------------------------------


async def respond_node(state: LoomState) -> dict:
    """Finalisation node — the HITL interrupt fires *before* this node runs.

    When ``interrupt_before=["respond"]`` is set in the compiled graph, the
    graph pauses here so an operator can inspect ``state`` and decide whether
    to approve.  On approval the caller resumes with ``graph.ainvoke(None,
    config=config)``, which executes this node and proceeds to END.

    The node itself is intentionally a no-op: the response is already fully
    formed in ``state.response`` by ``agent_node``.  Its sole purpose is to
    act as a named checkpoint for the interrupt mechanism.
    """
    logger.info(
        "respond_node | finalising | confidence=%.2f | iterations=%d",
        state.response.confidence_score if state.response else 0.0,
        state.iteration_count,
    )
    return {}
