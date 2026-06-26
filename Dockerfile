# ARVIT voice teammate. The SAME Dockerfile drives BOTH the Docker path
# (docker compose / docker run) and the Apple `container` path
# (scripts/container-up.sh).
#
# The image runs the full real-time voice loop (faster-whisper STT -> Claude
# Opus 4.8 on Bedrock -> Piper TTS) over a FastAPI WebSocket, bound to
# 127.0.0.1. On the Orin, build for arm64 / CUDA and grant the WendyOS
# network(host)/audio/persist entitlements (see README).
#
# Note: this image pulls the heavy real-time stack (pipecat / torch / whisper /
# piper). The fast, dependency-light UNIT TESTS run on the host with uv and do
# NOT need this image (see README "Run the tests").
FROM python:3.11-slim

# uv for fast, reproducible installs (parity with the local uv workflow).
COPY --from=ghcr.io/astral-sh/uv:0.11 /uv /usr/local/bin/uv

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    UV_SYSTEM_PYTHON=1

# Piper/whisper/onnxruntime need a couple of native libs.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ffmpeg \
        libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install runtime deps first for layer caching. Mirror pyproject's runtime set.
RUN uv pip install --system \
        "pipecat-ai[silero,whisper,piper,webrtc]>=0.0.60" \
        "faster-whisper>=1.0" \
        "piper-tts>=1.2" \
        "fastapi>=0.110" \
        "uvicorn[standard]>=0.29" \
        "anthropic[bedrock]>=0.40"

# Copy the package and project metadata.
COPY arvit_voice/ ./arvit_voice/
COPY tests/ ./tests/
COPY pyproject.toml README.md ./

# Make the source tree importable (no editable install needed).
ENV PYTHONPATH=/app

# Bind loopback by default (project convention); WendyOS uses network(host).
ENV HOST=127.0.0.1 \
    PORT=8765

EXPOSE 8765

# Default: serve the WS voice loop. Override the command to run the tests, e.g.
#   docker run --rm arvit-voice pytest -q
CMD ["uvicorn", "arvit_voice.main:app", "--host", "127.0.0.1", "--port", "8765"]
