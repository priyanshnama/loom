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

    # Persistence backend: "memory" | "sqlite"
    # memory  — in-process only; lost when the process exits (default, no setup needed)
    # sqlite  — file-backed SQLite; persists across CLI invocations, no server required
    loom_persistence: str = Field("memory", alias="LOOM_PERSISTENCE")

    # SQLite persistence
    sqlite_path: str = Field("./loom_checkpoints.db", alias="SQLITE_PATH")

    # Agent tuning
    loom_confidence_threshold: float = Field(0.75, alias="LOOM_CONFIDENCE_THRESHOLD")
    loom_max_iterations: int = Field(3, alias="LOOM_MAX_ITERATIONS")

    # Environment selector — drives which Telegram token is active.
    # "local" (default) → TELEGRAM_BOT_TOKEN_LOCAL
    # "production"       → TELEGRAM_BOT_TOKEN
    loom_env: str = Field("local", alias="LOOM_ENV")

    # Telegram bot — two separate bots so local testing never touches prod.
    telegram_token: str | None = Field(None, alias="TELEGRAM_BOT_TOKEN")
    telegram_token_local: str | None = Field(None, alias="TELEGRAM_BOT_TOKEN_LOCAL")

    @property
    def active_telegram_token(self) -> str | None:
        """Return the token for the active environment."""
        if self.loom_env == "production":
            return self.telegram_token
        return self.telegram_token_local or self.telegram_token

    # Zomato MCP — set ZOMATO_MCP_ENABLED=true to activate
    zomato_mcp_enabled: bool = Field(False, alias="ZOMATO_MCP_ENABLED")

    # Logging
    log_level: str = Field("INFO", alias="LOG_LEVEL")


# Module-level singleton — import this everywhere.
settings = Settings()  # type: ignore[call-arg]
