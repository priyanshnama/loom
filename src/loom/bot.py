"""Telegram bot entry point: ``python -m loom.bot``.

Each Telegram chat gets its own persistent session identified by
``tg_{chat_id}``.  The conversation history is stored in whichever backend
is configured (SQLite by default, Postgres in production).

Commands
--------
  /start  — welcome message
  /help   — show available commands
  /reset  — wipe THIS chat's conversation history and start fresh

Usage
-----
    # set TELEGRAM_BOT_TOKEN in .env, then:
    python -m loom.bot
"""

from __future__ import annotations

import asyncio
import html
import logging
import os
import sys
import threading
from contextlib import suppress
from http.server import BaseHTTPRequestHandler, HTTPServer
from textwrap import dedent

from dotenv import load_dotenv

load_dotenv()

from telegram import Update  # noqa: E402
from telegram.constants import ChatAction, ParseMode  # noqa: E402
from telegram.ext import (  # noqa: E402
    Application,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

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
logger = logging.getLogger("loom.bot")

# Telegram hard limit for a single message.
_MAX_MESSAGE_LEN = 4096


def _thread_id(update: Update) -> str:
    """Return a stable session identifier for this chat."""
    return f"tg_{update.effective_chat.id}"  # type: ignore[union-attr]


def _split_message(text: str, limit: int = _MAX_MESSAGE_LEN) -> list[str]:
    """Split ``text`` into chunks that fit within Telegram's message limit."""
    if len(text) <= limit:
        return [text]
    chunks: list[str] = []
    while text:
        chunks.append(text[:limit])
        text = text[limit:]
    return chunks


# ---------------------------------------------------------------------------
# Command handlers
# ---------------------------------------------------------------------------


async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(  # type: ignore[union-attr]
        dedent("""\
        👋 <b>Welcome to Loom!</b>

        I'm an AI research assistant backed by a persistent memory — \
I remember everything we discuss across sessions.

        Just send me any question or topic to get started.

        <b>Commands:</b>
        /help  — show this message
        /reset — wipe this chat's history and start fresh
        """),
        parse_mode=ParseMode.HTML,
    )


async def cmd_help(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(  # type: ignore[union-attr]
        dedent("""\
        <b>Loom commands</b>

        /start — welcome message
        /help  — show this message
        /reset — clear ALL conversation history for this chat

        Just send any text message to ask a question.
        I'll remember the full conversation context across sessions.
        """),
        parse_mode=ParseMode.HTML,
    )


async def cmd_reset(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    checkpointer = context.bot_data["checkpointer"]
    tid = _thread_id(update)

    await update.message.reply_text("🗑 Clearing your conversation history…")  # type: ignore[union-attr]
    try:
        await reset_thread(checkpointer, tid)
        await update.message.reply_text(  # type: ignore[union-attr]
            "✅ Done. Your history has been wiped — we're starting fresh!",
        )
        logger.info("bot | reset thread_id=%s", tid)
    except Exception as exc:
        logger.exception("bot | reset failed for thread_id=%s", tid)
        await update.message.reply_text(  # type: ignore[union-attr]
            f"❌ Reset failed: <code>{html.escape(str(exc))}</code>",
            parse_mode=ParseMode.HTML,
        )


# ---------------------------------------------------------------------------
# Message handler
# ---------------------------------------------------------------------------


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Run the user's message through the Loom graph and reply."""
    query = (update.message.text or "").strip()  # type: ignore[union-attr]
    if not query:
        return

    graph = context.bot_data["graph"]
    tid = _thread_id(update)

    logger.info("bot | query=%r thread_id=%s", query[:80], tid)

    stop_typing = asyncio.Event()
    typing_task = asyncio.create_task(
        _keep_typing(context.bot, update.effective_chat.id, stop_typing)  # type: ignore[union-attr]
    )

    try:
        initial_state = LoomState(
            messages=[HumanMessage(content=query)],
            query=query,
            max_iterations=settings.loom_max_iterations,
        )
        config = {"configurable": {"thread_id": tid}}
        raw = await graph.ainvoke(initial_state, config=config)
        final_state = LoomState.model_validate(raw)
    except Exception as exc:
        logger.exception("bot | graph error for thread_id=%s", tid)
        await update.message.reply_text(  # type: ignore[union-attr]
            f"⚠️ Something went wrong:\n<code>{html.escape(str(exc))}</code>",
            parse_mode=ParseMode.HTML,
        )
        return
    finally:
        stop_typing.set()
        typing_task.cancel()
        with suppress(asyncio.CancelledError):
            await typing_task

    if final_state.response is None:
        await update.message.reply_text("🤔 No response was generated.")  # type: ignore[union-attr]
        return

    resp = final_state.response
    confidence_bar = _confidence_bar(resp.confidence_score)

    footer_parts = [
        f"Confidence: {confidence_bar} {resp.confidence_score:.0%}"
        f" · {final_state.iteration_count} iteration(s)",
    ]
    if final_state.error_count:
        footer_parts.append(f"⚠️ Tool errors: {final_state.error_count}")
    if final_state.internal_monologue:
        monologue = html.escape(final_state.internal_monologue[:300])
        if len(final_state.internal_monologue) > 300:
            monologue += "…"
        footer_parts.append(f"💭 {monologue}")

    footer = "\n\n<i>" + "\n".join(footer_parts) + "</i>"

    body = html.escape(resp.answer) + footer
    for chunk in _split_message(body):
        await update.message.reply_text(chunk, parse_mode=ParseMode.HTML)  # type: ignore[union-attr]


async def _keep_typing(bot, chat_id: int, stop: asyncio.Event) -> None:
    """Re-send typing action every 4 s until stop is set (Telegram expires it after ~5 s)."""
    while not stop.is_set():
        with suppress(Exception):
            await bot.send_chat_action(chat_id=chat_id, action=ChatAction.TYPING)
        await asyncio.sleep(4)


def _confidence_bar(score: float) -> str:
    """Return a simple emoji bar representing confidence (0–1)."""
    filled = round(score * 5)
    return "🟩" * filled + "⬜" * (5 - filled)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


async def post_init(application: Application) -> None:  # type: ignore[type-arg]
    """Initialise the graph and checkpointer once at bot startup."""
    # We use an AsyncExitStack stored on the application so resources are
    # properly cleaned up when the bot shuts down.
    from contextlib import AsyncExitStack  # noqa: PLC0415

    stack = AsyncExitStack()
    application.bot_data["_stack"] = stack

    checkpointer = await stack.enter_async_context(get_checkpointer())
    graph = build_graph(checkpointer)

    application.bot_data["checkpointer"] = checkpointer
    application.bot_data["graph"] = graph
    logger.info("bot | graph initialised with backend=%s", settings.loom_persistence)


async def post_shutdown(application: Application) -> None:  # type: ignore[type-arg]
    """Release the checkpointer connection pool on shutdown."""
    stack = application.bot_data.get("_stack")
    if stack is not None:
        await stack.aclose()


class _HealthHandler(BaseHTTPRequestHandler):
    """Minimal HTTP handler so Cloud Run's health checks pass."""

    def do_GET(self) -> None:  # noqa: N802
        self.send_response(200)
        self.end_headers()
        self.wfile.write(b"ok")

    def log_message(self, *_: object) -> None:  # silence access logs
        pass


def _start_health_server() -> None:
    port = int(os.environ.get("PORT", 8080))
    HTTPServer(("0.0.0.0", port), _HealthHandler).serve_forever()


def main() -> None:
    token = settings.active_telegram_token
    if not token:
        env_var = "TELEGRAM_BOT_TOKEN" if settings.loom_env == "production" else "TELEGRAM_BOT_TOKEN_LOCAL"
        print(f"Error: {env_var} is not set in .env (LOOM_ENV={settings.loom_env})", file=sys.stderr)
        sys.exit(1)

    logger.info("bot | env=%s", settings.loom_env)
    app = (
        Application.builder()
        .token(token)
        .post_init(post_init)
        .post_shutdown(post_shutdown)
        .build()
    )

    app.add_handler(CommandHandler("start", cmd_start))
    app.add_handler(CommandHandler("help", cmd_help))
    app.add_handler(CommandHandler("reset", cmd_reset))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    # Cloud Run requires a port listener — start one in the background.
    threading.Thread(target=_start_health_server, daemon=True).start()
    logger.info("bot | starting polling…")
    app.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()
