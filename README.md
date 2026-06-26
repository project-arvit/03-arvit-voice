# arvit-voice — the ARVIT voice teammate

Bidirectional, real-time voice for the ARVIT autonomous-inspection robodog
(Unitree Go2 EDU + Jetson Orin). The operator talks; ARVIT talks back:

```
WS audio in -> Silero VAD (barge-in) -> faster-whisper STT
            -> [ E-STOP gate | Claude Opus 4.8 on Bedrock (streamed) ]
            -> Piper TTS -> WS audio out
```

**Local STT + local TTS stay on the Orin** (faster-whisper + Piper); only the
**LLM stage hits the cloud** (Claude Opus 4.8 via AWS Bedrock), and it is
streamed token-by-token so TTS can start speaking before generation finishes.
The orchestration is [Pipecat](https://github.com/pipecat-ai/pipecat).

## Layout

| Module | What it is | Heavy deps? |
|--------|------------|-------------|
| `arvit_voice/estop.py` | Pure e-stop detection + `EstopRouter` (safety). | **none** |
| `arvit_voice/bedrock_llm.py` | `BedrockClaudeLLM` — streaming Bedrock wrapper. | only `anthropic[bedrock]` |
| `arvit_voice/pipeline.py` | The Pipecat pipeline (imports pipecat **lazily**). | pipecat/whisper/piper (lazy) |
| `arvit_voice/main.py` | FastAPI WS entrypoint, binds 127.0.0.1. | fastapi/uvicorn (+ pipecat lazy) |
| `tests/test_voice.py` | Fast unit tests — no audio, no network, no robot. | `anthropic[bedrock]` + pytest |

The split is deliberate: the safety-critical e-stop logic and the LLM request
builder are importable and testable **without** pipecat, audio, a robot, or a
live AWS account. `pipeline.py` and `main.py` import pipecat lazily.

## Safety: e-stop never moves the robot

`arvit_voice/estop.py` detects `stop` / `e-stop` / `estop` / `emergency stop` /
`halt` with word-boundary, case-insensitive matching (so `stopwatch`,
`nonstop`, and `halting` do **not** trigger). The `EstopRouter` only ever calls
a *stop* callback — it has **no motion surface by construction**, so a misheard
command cannot cause motion. The worst case is a spurious stop, which is the
safe direction. In the pipeline, an e-stop utterance short-circuits the LLM
entirely and fires the hard-stop hook (wire this to the Go2 `Damp` tool on the
Orin). Transcribed audio is treated as untrusted input (zero-trust).

## Run the tests (now, on macOS — no robot/audio/network/keys)

The tests install **only** `anthropic[bedrock]` + pytest. pipecat / whisper /
piper are NOT needed. `--no-project` keeps uv from trying to build the package
(and its heavy deps) just to run the suite:

```bash
cd arvit-voice
uv venv --python 3.11
uv pip install "anthropic[bedrock]" pytest
uv run --no-project pytest -q
```

Expected: **26 passed**. The suite monkeypatches the Bedrock client and asserts
the request shape (model id from env, `thinking={"type":"adaptive"}`,
`output_config.effort`, **no** `temperature`/`top_p`/`top_k`/`budget_tokens`,
streaming) and that it yields text deltas; plus full e-stop detection/routing
coverage.

## Run the full voice loop (Docker)

Same Dockerfile as the Apple `container` path. Published port bound to
127.0.0.1.

```bash
cp .env.example .env          # then fill in AWS creds (or use an instance role)
docker compose up --build     # serves ws://127.0.0.1:8765/ws
# run the suite inside the image:
docker compose run --rm voice pytest -q
```

## Run the full voice loop (Apple `container`)

Apple `container` is single-container (no compose): one `build` + one `run`.

```bash
cp .env.example .env
scripts/container-up.sh            # build + run, publishes 127.0.0.1:8765
scripts/container-up.sh pytest -q  # run the suite inside the image
scripts/container-down.sh          # stop + remove container and image
```

Endpoints: `GET /health` (liveness, no audio/cloud) and `WS /ws` (the voice
loop, one client at a time — the Pipecat WebSocket transport is single-client).

## What deploys to the Orin

- The full image (build for **arm64 / CUDA**): faster-whisper STT and Piper TTS
  run **locally** on the Orin GPU; set `WHISPER_DEVICE=cuda` /
  `WHISPER_COMPUTE_TYPE=float16` in `.env`.
- Only the **LLM stage** leaves the Orin (Bedrock, streamed). No audio, no STT,
  no TTS goes to the cloud.
- On WendyOS, declare these entitlements in `wendy.json`:
  - `{ "type": "network", "mode": "host" }` — cloud Bedrock access + serving the
    WS frontend.
  - `{ "type": "audio" }` — ALSA mic/speaker (USB headset or the Go2 mic/speaker).
  - `{ "type": "persist", "name": "app-voice-models", "path": "/models" }` —
    cache Whisper/Piper weights (and any TLS certs) across restarts.
  - Optionally `{ "type": "gpu" }` for CUDA STT/TTS.

## What needs hardware / keys (not exercised by the local tests)

- **AWS Bedrock access** (`AWS_REGION`, `BEDROCK_MODEL_ID`, credentials) — the
  only cloud dependency. The local unit tests stub this out entirely.
- **Mic + speaker** — a USB headset (operator) or the Go2's own mic/speaker.
  USB hot-plug needs a container restart.
- **pipecat / faster-whisper / piper** runtime — only needed to run the actual
  voice loop (the container image), not the unit tests.

## Noisy-site fallback (push-to-talk)

A live industrial site is loud and hands-free VAD/barge-in will misfire. Treat
VAD as the default but keep a literal **push-to-talk** control: gate audio into
the WS on the client so the pipeline only sees speech while the button is held.
Always confirm the parsed command back before acting ("Understood — checking
gauge three?"). Voice is a **P2 teammate flourish**; the operator console
dashboard is the fallback if voice slips.

## Configuration (env)

See `.env.example`. Key knobs: `BEDROCK_MODEL_ID`, `AWS_REGION`,
`WHISPER_MODEL_SIZE`/`WHISPER_DEVICE`/`WHISPER_COMPUTE_TYPE`, `PIPER_VOICE`,
`ALLOW_INTERRUPTIONS`, `HOST`/`PORT`.
