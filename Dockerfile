# ---------- build stage ----------
FROM python:3.12-slim AS builder

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app
COPY pyproject.toml uv.lock README.md ./
COPY src/ ./src/

# Install all runtime deps + the loom package into .venv
RUN uv sync --frozen --no-dev

# ---------- runtime stage ----------
FROM python:3.12-slim

WORKDIR /app

COPY --from=builder /app/.venv /app/.venv
COPY --from=builder /app/src   /app/src
COPY .env .env

ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1

# Cloud Run injects PORT; default to 8080 for local runs.
EXPOSE 8080

CMD ["python", "-m", "loom.bot"]
