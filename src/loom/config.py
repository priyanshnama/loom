"""Application-wide settings loaded from environment / .env file."""

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

# Default system prompt — edit src/loom/prompts/system.md or set LOOM_SYSTEM_PROMPT in .env.
_DEFAULT_SYSTEM_PROMPT: str = (
    Path(__file__).parent / "prompts" / "system.md"
).read_text(encoding="utf-8")


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # LLM
    anthropic_api_key: str = Field(..., alias="ANTHROPIC_API_KEY")
    loom_model: str = Field("anthropic:claude-sonnet-4-6", alias="LOOM_MODEL")

    # System prompt — loaded from prompts/system.md by default; override via env var.
    loom_system_prompt: str = Field(default=_DEFAULT_SYSTEM_PROMPT, alias="LOOM_SYSTEM_PROMPT")

    # Persistence backend: "memory" | "sqlite" | "neo4j"
    # memory  — in-process only; lost when the process exits (default, no setup needed)
    # sqlite  — file-backed SQLite; persists across CLI invocations, no server required
    # neo4j   — production-grade; requires a running Neo4j instance
    loom_persistence: str = Field("memory", alias="LOOM_PERSISTENCE")

    # SQLite persistence
    sqlite_path: str = Field("./loom_checkpoints.db", alias="SQLITE_PATH")

    # Neo4j persistence + knowledge graph (used when LOOM_PERSISTENCE=neo4j, and always for KG)
    neo4j_uri: str = Field("bolt://localhost:7687", alias="NEO4J_URI")
    neo4j_username: str = Field("neo4j", alias="NEO4J_USERNAME")
    neo4j_password: str = Field("", alias="NEO4J_PASSWORD")
    neo4j_database: str = Field("neo4j", alias="NEO4J_DATABASE")

    # Agent tuning
    loom_confidence_threshold: float = Field(0.75, alias="LOOM_CONFIDENCE_THRESHOLD")
    loom_max_iterations: int = Field(3, alias="LOOM_MAX_ITERATIONS")

    # Telegram bot
    telegram_token: str | None = Field(None, alias="TELEGRAM_BOT_TOKEN")

    # Logging
    log_level: str = Field("INFO", alias="LOG_LEVEL")


# Module-level singleton — import this everywhere.
settings = Settings()  # type: ignore[call-arg]
