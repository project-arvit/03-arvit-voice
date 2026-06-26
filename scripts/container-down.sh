#!/usr/bin/env bash
# Teardown for the Apple `container` path: stop + remove the container and the
# image. Safe to run repeatedly (ignores "not found").
set -euo pipefail

IMAGE="arvit-voice:latest"
NAME="arvit-voice"

echo ">> stopping/removing container ${NAME}"
container stop "${NAME}" >/dev/null 2>&1 || true
container rm "${NAME}" >/dev/null 2>&1 || true

echo ">> deleting image ${IMAGE}"
# Newer Apple `container` uses `image delete`; older builds use `rmi`.
container image delete "${IMAGE}" >/dev/null 2>&1 \
  || container rmi "${IMAGE}" >/dev/null 2>&1 \
  || true

echo ">> done"
