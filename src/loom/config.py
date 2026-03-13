"""Application-wide settings loaded from environment / .env file."""

from pathlib import Path

from pydantic import Field, PostgresDsn
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

    # Persistence backend: "memory" | "sqlite" | "postgres"
    # memory  — in-process only; lost when the process exits (default, no setup needed)
    # sqlite  — file-backed SQLite; persists across CLI invocations, no server required
    # postgres — production-grade; requires a running Postgres instance
    loom_persistence: str = Field("memory", alias="LOOM_PERSISTENCE")

    # SQLite persistence
    sqlite_path: str = Field("./loom_checkpoints.db", alias="SQLITE_PATH")

    # Postgres persistence (used when LOOM_PERSISTENCE=postgres)
    postgres_dsn: PostgresDsn | None = Field(None, alias="POSTGRES_DSN")

    # Agent tuning
    loom_confidence_threshold: float = Field(0.75, alias="LOOM_CONFIDENCE_THRESHOLD")
    loom_max_iterations: int = Field(3, alias="LOOM_MAX_ITERATIONS")

    # Telegram bot
    telegram_token: str | None = Field(None, alias="TELEGRAM_BOT_TOKEN")

    # Logging
    log_level: str = Field("INFO", alias="LOG_LEVEL")

    @property
    def effective_postgres_dsn(self) -> str:
        """Return the Postgres DSN as a plain string."""
        if self.postgres_dsn:
            return str(self.postgres_dsn)
        raise ValueError("POSTGRES_DSN must be set when LOOM_PERSISTENCE=postgres")


# Module-level singleton — import this everywhere.
settings = Settings()  # type: ignore[call-arg]
