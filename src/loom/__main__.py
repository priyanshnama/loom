"""CLI entry point: ``python -m loom``.

Usage
-----
    # Ask a question (creates or resumes the session)
    python -m loom --query "What is quantum entanglement?" --thread my-session

    # Follow-up in the same session
    python -m loom --query "Elaborate on the EPR paradox" --thread my-session

    # Wipe the session and start fresh (no query needed)
    python -m loom --thread my-session --reset
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


async def run(query: str, thread_id: str, max_iterations: int) -> None:
    async with get_checkpointer() as checkpointer:
        graph = build_graph(checkpointer)
        initial_state = LoomState(
            messages=[HumanMessage(content=query)],
            query=query,
            max_iterations=max_iterations,
        )
        config = {"configurable": {"thread_id": thread_id}}
        logger.info("Invoking graph | thread_id=%s | query=%r", thread_id, query)

        raw = await graph.ainvoke(initial_state, config=config)  # type: ignore[arg-type]
        final_state = LoomState.model_validate(raw)

        if final_state.response is None:
            print("No response generated.", file=sys.stderr)
            sys.exit(1)

        result = {
            "thread_id": thread_id,
            "answer": final_state.response.answer,
            "confidence_score": final_state.response.confidence_score,
            "sources": final_state.response.sources,
            "reasoning": final_state.response.reasoning,
            "iterations": final_state.iteration_count,
        }
        print(json.dumps(result, indent=2))


async def do_reset(thread_id: str) -> None:
    async with get_checkpointer() as checkpointer:
        await reset_thread(checkpointer, thread_id)
    print(f"✓ Session '{thread_id}' has been reset.")


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="loom",
        description="Run a query through the Loom agentic workflow.",
    )
    parser.add_argument("--query", "-q", help="The question to answer.")
    parser.add_argument(
        "--thread", "-t",
        default="default",
        help="Session thread ID (default: 'default').",
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

    if not args.query:
        parser.error("--query is required unless --reset is specified.")

    asyncio.run(run(args.query, args.thread, args.max_iterations))


if __name__ == "__main__":
    main()
