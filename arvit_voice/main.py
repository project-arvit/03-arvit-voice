"""FastAPI entrypoint for the ARVIT voice teammate.

Serves a single WebSocket endpoint that carries the operator's mic/speaker
audio in and out of the Pipecat pipeline. Binds to 127.0.0.1 by default
(project convention: published ports bind loopback; on WendyOS the app runs
with the ``network(host)`` entitlement).

pipecat is imported lazily (inside the WS handler / pipeline builder) so this
module stays importable for tooling that only needs the app object.

Run locally / on the Orin:
    uv run uvicorn arvit_voice.main:app --host 127.0.0.1 --port ${PORT:-8765}
"""

from __future__ import annotations

import os

from fastapi import FastAPI, WebSocket

from .pipeline import VoiceConfig, build_pipeline, make_vad_analyzer

app = FastAPI(title="ARVIT Voice Teammate", version="0.1.0")


@app.get("/health")
async def health() -> dict:
    """Liveness probe; no audio, no cloud."""
    cfg = VoiceConfig.from_env()
    return {
        "status": "ok",
        "whisper_model": cfg.whisper_model,
        "piper_voice": cfg.piper_voice,
        "allow_interruptions": cfg.allow_interruptions,
    }


@app.websocket("/ws")
async def ws_voice(websocket: WebSocket) -> None:
    """Bidirectional voice loop over a single WebSocket.

    One client at a time (the Pipecat WebSocket transport is single-client by
    design). VAD/barge-in is configured on the transport; the LLM stage routes
    e-stop utterances to the hard-stop hook and streams normal replies.
    """
    await websocket.accept()

    # Lazy imports keep module import light and avoid requiring pipecat for
    # health checks / unit tests.
    from pipecat.pipeline.runner import PipelineRunner
    from pipecat.serializers.protobuf import ProtobufFrameSerializer
    from pipecat.transports.network.fastapi_websocket import (
        FastAPIWebsocketParams,
        FastAPIWebsocketTransport,
    )

    cfg = VoiceConfig.from_env()

    transport = FastAPIWebsocketTransport(
        websocket=websocket,
        params=FastAPIWebsocketParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
            add_wav_header=False,
            vad_analyzer=make_vad_analyzer(),
            serializer=ProtobufFrameSerializer(),
        ),
    )

    _pipeline, task = build_pipeline(transport, cfg)
    runner = PipelineRunner(handle_sigint=False)
    await runner.run(task)


def main() -> None:
    """Console entrypoint: serve the WS app bound to loopback."""
    import uvicorn

    host = os.environ.get("HOST", "127.0.0.1")
    port = int(os.environ.get("PORT", "8765"))
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()
