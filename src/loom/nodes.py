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


# ---------------------------------------------------------------------------
# Node: agent
# ---------------------------------------------------------------------------


async def agent_node(state: LoomState) -> dict:
    """Call the Pydantic AI agent and update state with its structured response.

    pydantic-ai handles tool selection and execution internally within a single
    agent.run() call.  After it returns, we extract any tool outputs from the
    run's message history and store them in state for the next turn.

    Returns a partial state dict containing:
    - ``messages``: ToolMessages for each tool call + final AIMessage.
    - ``response``: the validated ``LoomResponse``.
    - ``iteration_count``: incremented by 1.
    - ``tool_outputs``: merged dict of all tool results from this turn.
    """
    # Build prompt — include prior conversation and any accumulated tool outputs.
    prompt_parts: list[str] = []

    prior = [m for m in state.messages if isinstance(m, (HumanMessage, AIMessage))]
    if prior:
        history_lines = []
        for msg in prior:
            role = "User" if isinstance(msg, HumanMessage) else "Assistant"
            history_lines.append(f"{role}: {msg.content}")
        prompt_parts.append("Conversation so far:\n" + "\n".join(history_lines))

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

    # Collect tool outputs from this pydantic-ai run (may be zero if no tools called).
    new_tool_outputs: dict[str, list[str]] = {}
    extra_messages: list[ToolMessage] = []

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

    # Merge new outputs into accumulated state.
    merged_outputs: dict[str, list[str]] = dict(state.tool_outputs)
    for k, v in new_tool_outputs.items():
        merged_outputs.setdefault(k, []).extend(v)

    return {
        "messages": [*extra_messages, AIMessage(content=response.answer)],
        "response": response,
        "iteration_count": state.iteration_count + 1,
        "tool_outputs": merged_outputs,
    }
