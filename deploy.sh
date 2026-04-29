#!/usr/bin/env bash
set -euo pipefail

ENV_FILE="${1:-.env}"

if [[ ! -f "$ENV_FILE" ]]; then
  echo "Error: $ENV_FILE not found" >&2
  exit 1
fi

# Build env-var string from .env, skipping comments/blanks and local-only keys.
# Production values for LOOM_ENV and LOOM_PERSISTENCE are appended at the end.
env_str=""

while IFS='=' read -r key value; do
  case "$key" in
    TELEGRAM_BOT_TOKEN_LOCAL|LOOM_ENV|LOOM_PERSISTENCE) continue ;;
  esac
  env_str="${env_str}${key}=${value}|"
done < <(grep -E '^[A-Za-z_][A-Za-z0-9_]*=' "$ENV_FILE" || true)

# Append production-specific overrides
env_str="${env_str}LOOM_ENV=production|LOOM_PERSISTENCE=postgres"

# Validate required keys are present
for key in ANTHROPIC_API_KEY POSTGRES_DSN TELEGRAM_BOT_TOKEN; do
  if ! echo "$env_str" | grep -q "${key}=."; then
    echo "Error: $key not found or empty in $ENV_FILE" >&2
    exit 1
  fi
done

IMAGE="asia-south1-docker.pkg.dev/gen-lang-client-0546866199/loom/bot:latest"
REGION="asia-south1"
SERVICE="loom-bot"

echo "→ Building and pushing Docker image (linux/amd64 for Cloud Run)..."
docker buildx build --platform linux/amd64 --push -t "$IMAGE" .

echo "→ Deploying $SERVICE to Cloud Run ($REGION)..."
gcloud run deploy "$SERVICE" \
  --image "$IMAGE" \
  --region "$REGION" \
  --platform managed \
  --no-allow-unauthenticated \
  --min-instances 1 \
  --no-cpu-throttling \
  --memory 512Mi \
  --set-env-vars "^|^${env_str}"

echo "✓ Deployment complete."
