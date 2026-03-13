"""Pydantic AI structured-output models.

All agent responses are validated against ``LoomResponse`` so downstream
consumers receive a guaranteed schema rather than raw text.
"""

from __future__ import annotations

from pydantic import BaseModel, Field, field_validator


class LoomResponse(BaseModel):
    """Structured output produced by the Loom agent."""

    answer: str = Field(
        description="The agent's synthesised answer to the user's query.",
    )
    confidence_score: float = Field(
        ge=0.0,
        le=1.0,
        description="Self-assessed confidence in the answer (0 = none, 1 = certain).",
    )
    sources: list[str] = Field(
        default_factory=list,
        description="References or source snippets used to produce the answer.",
    )
    reasoning: str = Field(
        default="",
        description="Optional chain-of-thought summary explaining the answer.",
    )

    @field_validator("confidence_score", mode="before")
    @classmethod
    def clamp_confidence(cls, v: float) -> float:
        """Clamp any out-of-range float coming from the model."""
        return max(0.0, min(1.0, float(v)))

    @property
    def needs_refinement(self) -> bool:
        """True when the score is too low to be considered a final answer."""
        # Import here to avoid circular dependency at module load time.
        from loom.config import settings

        return self.confidence_score < settings.loom_confidence_threshold
