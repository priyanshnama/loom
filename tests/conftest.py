"""Test configuration — sets minimal env vars so loom modules can be imported
without a real .env file (tests don't make any actual LLM or DB calls)."""

import os

# Must be set at module level (before any loom import) so that
# `settings = Settings()` in config.py succeeds during collection.
os.environ.setdefault("ANTHROPIC_API_KEY", "test-key-not-real")
os.environ.setdefault("LOOM_PERSISTENCE", "memory")
