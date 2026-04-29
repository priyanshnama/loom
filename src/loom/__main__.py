"""CLI entry point: ``python -m loom``.

Usage
-----
    # Ask a question (creates or resumes the session)
    python -m loom --query "What is quantum entanglement?" --thread my-session

    # Follow-up in the same session (conversation resumes from checkpoint)
    python -m loom --query "Elaborate on the EPR paradox" --thread my-session

    # Enable Human-in-the-Loop: graph pauses before the final response
    python -m loom --query "Book me a flight to Mumbai" --thread my-session --hitl

    # Resume after a HITL interrupt (the graph was paused, not yet finished)
    python -m loom --thread my-session --resume

    # Wipe the session and start fresh (no query needed)
    python -m loom --thread my-session --reset

HITL flow
---------
When --hitl is supplied:
1. The graph runs agent_node (possibly multiple times) until it produces a
   high-confidence answer.
2. The graph pauses **before** the ``respond`` node and ainvoke() returns.
3. The CLI prints the internal monologue, proposed answer, tool outputs, and
   error count for inspection.
4. The operator is prompted: "Approve? [y/n/feedback]"
   - y          → resumes the graph (executes respond_node → END).
   - n          → resets the session and exits.
   - <feedback> → appends feedback as a new HumanMessage and re-runs the
                  agent from the current checkpoint.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys

from dotenv import load_dotenv

load_dotenv()

from langchain_core.messages import HumanMessage  # noqa: E402

from loom.config import settings  # noqa: E402
from loom.graph import build_graph  # noqa: E402
from loom.persistence import get_checkpointer  # noqa: E402
from loom.session import reset_thread  # noqa: E402
from loom.state import LoomState  # noqa: E402

logging.basicConfig(
    level=settings.log_level,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    stream=sys.stderr,
)
logger = logging.getLogger("loom.__main__")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _state_from_raw(raw: dict) -> LoomState:
    return LoomState.model_validate(raw)


def _print_result(thread_id: str, state: LoomState) -> None:
    if state.response is None:
        print("No response generated.", file=sys.stderr)
        sys.exit(1)

    result = {
        "thread_id": thread_id,
        "answer": state.response.answer,
        "confidence_score": state.response.confidence_score,
        "sources": state.response.sources,
        "reasoning": state.response.reasoning,
        "iterations": state.iteration_count,
        "error_count": state.error_count,
    }
    print(json.dumps(result, indent=2))


def _print_hitl_summary(state: LoomState) -> None:
    """Print the graph state for operator review during a HITL interrupt."""
    sep = "─" * 60
    print(f"\n{sep}")
    print("  ⏸  HUMAN-IN-THE-LOOP — graph paused before respond node")
    print(sep)
    if state.internal_monologue:
        print(f"\n  Internal monologue:\n    {state.internal_monologue}")
    if state.tool_outputs:
        print("\n  Tool outputs:")
        for tool, outputs in state.tool_outputs.items():
            for out in outputs:
                print(f"    [{tool}] {out}")
    if state.error_count:
        print(f"\n  ⚠  Tool errors encountered this session: {state.error_count}")
    if state.response:
        print(f"\n  Proposed answer:\n    {state.response.answer}")
        print(f"  Confidence: {state.response.confidence_score:.2f}")
        print(f"  Sources:    {state.response.sources}")
    print(f"\n{sep}")


# ---------------------------------------------------------------------------
# Core run logic
# ---------------------------------------------------------------------------


async def run(query: str, thread_id: str, max_iterations: int, hitl: bool) -> None:
    async with get_checkpointer() as checkpointer:
        graph = build_graph(checkpointer, hitl=hitl)
        initial_state = LoomState(
            messages=[HumanMessage(content=query)],
            query=query,
            max_iterations=max_iterations,
        )
        config = {"configurable": {"thread_id": thread_id}}
        logger.info("Invoking graph | thread_id=%s | query=%r | hitl=%s", thread_id, query, hitl)

        raw = await graph.ainvoke(initial_state, config=config)  # type: ignore[arg-type]

        if hitl:
            raw = await _handle_hitl(graph, checkpointer, config, raw, thread_id)
            if raw is None:
                return  # operator rejected; session already reset

        _print_result(thread_id, _state_from_raw(raw))


async def resume(thread_id: str) -> None:
    """Resume a previously interrupted (HITL) graph run.

    Use this when the process was restarted between the interrupt and the
    operator's approval — the checkpoint stores all state durably.
    """
    async with get_checkpointer() as checkpointer:
        graph = build_graph(checkpointer, hitl=True)
        config = {"configurable": {"thread_id": thread_id}}

        # Verify there actually is a pending interrupt for this thread.
        graph_state = await graph.aget_state(config)
        if not graph_state.next:
            print(
                f"Thread '{thread_id}' has no pending interrupt to resume.",
                file=sys.stderr,
            )
            sys.exit(1)

        current = LoomState.model_validate(graph_state.values)
        _print_hitl_summary(current)

        raw = await _approve_and_resume(graph, checkpointer, config, graph_state.values, thread_id)
        if raw is None:
            return
        _print_result(thread_id, _state_from_raw(raw))


# ---------------------------------------------------------------------------
# HITL helpers
# ---------------------------------------------------------------------------


async def _handle_hitl(graph, checkpointer, config, raw, thread_id):
    """Called immediately after the first ainvoke when --hitl is active.

    Checks whether the graph is sitting at the interrupt point (i.e. it
    produced a high-confidence answer and is waiting before respond_node).
    If not interrupted (max_iterations exhausted), just returns raw as-is.
    """
    graph_state = await graph.aget_state(config)
    if not graph_state.next:
        # Graph ran to completion without hitting the HITL interrupt
        # (max_iterations guard kicked in).
        return raw

    return await _approve_and_resume(graph, checkpointer, config, raw, thread_id)


async def _approve_and_resume(graph, checkpointer, config, raw, thread_id):
    """Show the paused state and ask the operator to approve, reject, or give feedback."""
    state = _state_from_raw(raw)
    _print_hitl_summary(state)

    while True:
        try:
            answer = input(
                "\n  Approve this response?\n"
                "  [y] Approve and finalise\n"
                "  [n] Reject and reset session\n"
                "  [<feedback>] Enter feedback to refine the answer\n"
                "  > "
            ).strip()
        except (EOFError, KeyboardInterrupt):
            print("\nInterrupted — session left open for later --resume.", file=sys.stderr)
            return None

        if answer.lower() == "y":
            print("\n  ✓ Approved. Resuming graph...\n", file=sys.stderr)
            # Resume: pass None so LangGraph continues from the checkpoint.
            final_raw = await graph.ainvoke(None, config=config)  # type: ignore[arg-type]
            return final_raw

        if answer.lower() == "n":
            print("\n  ✗ Rejected. Resetting session.\n", file=sys.stderr)
            await reset_thread(checkpointer, thread_id)
            return None

        # Operator provided feedback — inject it as a new HumanMessage and
        # update the checkpoint so the agent picks it up on the next run.
        feedback = answer
        print(f"\n  Injecting feedback: {feedback!r}\n  Re-running agent...\n", file=sys.stderr)

        # Update the graph's checkpoint with the new message, then re-run
        # from the interrupt point by providing the feedback as the new input.
        await graph.aupdate_state(
            config,
            {"messages": [HumanMessage(content=f"[Operator feedback] {feedback}")]},
            as_node="synthesizer",
        )
        # Re-invoke from the updated state; this resumes before respond_node
        # again if the new answer meets the confidence threshold.
        raw = await graph.ainvoke(None, config=config)  # type: ignore[arg-type]
        state = _state_from_raw(raw)

        # Check whether we hit the interrupt again (new answer ready for review)
        # or ran to completion (max_iterations).
        graph_state = await graph.aget_state(config)
        if not graph_state.next:
            return raw  # completed without another interrupt


# ---------------------------------------------------------------------------
# Session management
# ---------------------------------------------------------------------------


async def do_reset(thread_id: str) -> None:
    async with get_checkpointer() as checkpointer:
        await reset_thread(checkpointer, thread_id)
    print(f"✓ Session '{thread_id}' has been reset.")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="loom",
        description="Run a query through the Loom agentic workflow.",
    )
    parser.add_argument("--query", "-q", help="The question to answer.")
    parser.add_argument(
        "--thread", "-t",
        default="default",
        help="Session thread ID (default: 'default').  Required for persistence and resume.",
    )
    parser.add_argument(
        "--hitl",
        action="store_true",
        help=(
            "Enable Human-in-the-Loop mode: pause before the final response "
            "so you can inspect and approve the output."
        ),
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help=(
            "Resume a previously interrupted HITL session for --thread. "
            "Useful when the process was restarted between interrupt and approval."
        ),
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Wipe all history for --thread and exit (no query needed).",
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=settings.loom_max_iterations,
        help="Maximum agent→tool refinement loops.",
    )
    args = parser.parse_args()

    if args.reset:
        asyncio.run(do_reset(args.thread))
        return

    if args.resume:
        asyncio.run(resume(args.thread))
        return

    if not args.query:
        parser.error("--query is required unless --reset or --resume is specified.")

    asyncio.run(run(args.query, args.thread, args.max_iterations, args.hitl))


if __name__ == "__main__":
    main()
