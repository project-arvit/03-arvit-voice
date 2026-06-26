#!/usr/bin/env bash
# Apple `container` path for arvit-voice (single-container; no compose).
# Builds the image from the SAME Dockerfile as the Docker path, then runs it.
#
# By default it serves the WS voice loop on 127.0.0.1:${PORT:-8765}. Pass extra
# args to override the command, e.g. to run the suite inside the image:
#   scripts/container-up.sh pytest -q
#
# Secrets come ONLY via --env-file .env (.env is gitignored). The published
# port is bound to 127.0.0.1 (project convention).
set -euo pipefail

IMAGE="arvit-voice:latest"
NAME="arvit-voice"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

cd "$HERE"

PORT="${PORT:-8765}"

# Secrets ONLY via --env-file (.env is gitignored). Use .env if present.
ENV_ARGS=()
if [[ -f .env ]]; then
  ENV_ARGS=(--env-file .env)
fi

echo ">> container build -t ${IMAGE}"
container build -t "${IMAGE}" -f Dockerfile .

# Remove any prior instance with the same name so re-runs are clean.
container rm "${NAME}" >/dev/null 2>&1 || true

echo ">> container run --name ${NAME} (publishing 127.0.0.1:${PORT})"
# Apple `container` is single-container, no compose. Publish loopback only.
# Pass-through args ($@) override the image CMD.
container run --rm --name "${NAME}" \
  -p "127.0.0.1:${PORT}:8765" \
  ${ENV_ARGS[@]+"${ENV_ARGS[@]}"} \
  "${IMAGE}" "$@"
