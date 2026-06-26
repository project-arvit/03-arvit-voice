"""Pipecat real-time voice pipeline for the ARVIT voice teammate.

Dataflow (see vault: voice-pipecat-pipeline):

    WS audio in -> Silero VAD (barge-in) -> faster-whisper STT
                -> [ EstopRouter | Bedrock Claude LLM ]
                -> Piper TTS -> WS audio out

Local STT + local TTS stay on the Orin; only the LLM stage hits the cloud
(Bedrock), streamed. An "estop"/"stop" utterance is routed to a hard-stop hook
via :mod:`arvit_voice.estop` and MUST NEVER cause motion.

IMPORTANT: pipecat (and faster-whisper / piper) are imported LAZILY inside
``build_pipeline`` / ``ArvitLLMProcessor`` so that :mod:`arvit_voice.estop` and
:mod:`arvit_voice.bedrock_llm` remain importable — and unit-testable — without
the heavy real-time stack installed.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Awaitable, Callable, Optional

from .bedrock_llm import BedrockClaudeLLM
from .estop import EstopRouter


@dataclass
class VoiceConfig:
    """Environment-driven configuration for the voice pipeline.

    Every knob the vault calls out as configurable lives here: Whisper size,
    Piper voice, VAD/barge-in, and the LLM model/effort (via the LLM wrapper).
    """

    whisper_model: str = field(
        default_factory=lambda: os.environ.get("WHISPER_MODEL_SIZE", "tiny")
    )
    whisper_device: str = field(
        default_factory=lambda: os.environ.get("WHISPER_DEVICE", "cpu")
    )
    whisper_compute_type: str = field(
        default_factory=lambda: os.environ.get("WHISPER_COMPUTE_TYPE", "int8")
    )
    piper_voice: str = field(
        default_factory=lambda: os.environ.get("PIPER_VOICE", "en_US-lessac-medium")
    )
    piper_model_path: Optional[str] = field(
        default_factory=lambda: os.environ.get("PIPER_MODEL_PATH") or None
    )
    # Barge-in: VAD-detected speech-start cancels in-flight TTS.
    allow_interruptions: bool = field(
        default_factory=lambda: os.environ.get("ALLOW_INTERRUPTIONS", "1") not in ("0", "false", "False")
    )
    sample_rate: int = field(
        default_factory=lambda: int(os.environ.get("AUDIO_SAMPLE_RATE", "16000"))
    )

    @classmethod
    def from_env(cls) -> "VoiceConfig":
        return cls()


def default_stop_hook() -> None:
    """Fallback hard-stop hook used if the caller supplies none.

    On the Orin this is replaced by the real e-stop / Damp command (see
    mcp-robot-control). The default only logs, so the pipeline is safe to run
    on a laptop without any motion surface wired up. It NEVER commands motion.
    """
    import logging

    logging.getLogger("arvit_voice.estop").critical(
        "E-STOP utterance detected -> hard-stop hook fired (no motion path)."
    )


def make_arvit_llm_processor(
    llm: Optional[BedrockClaudeLLM] = None,
    stop_hook: Optional[Callable[[], None]] = None,
):
    """Build the ARVIT LLM FrameProcessor (imports pipecat lazily).

    The processor consumes transcription frames; an e-stop utterance is routed
    to ``stop_hook`` (and NOT forwarded to the LLM or any motion path), while
    normal speech is streamed through :class:`BedrockClaudeLLM` and emitted as
    TTS text frames for the downstream Piper stage.
    """
    from pipecat.frames.frames import (
        LLMFullResponseEndFrame,
        LLMFullResponseStartFrame,
        TextFrame,
        TranscriptionFrame,
    )
    from pipecat.processors.frame_processor import FrameDirection, FrameProcessor

    _llm = llm or BedrockClaudeLLM()
    _router = EstopRouter(stop_hook or default_stop_hook)

    class ArvitLLMProcessor(FrameProcessor):
        """STT-text -> e-stop-gate -> Bedrock LLM -> TTS-text frames."""

        def __init__(self) -> None:
            super().__init__()
            self._llm = _llm
            self._router = _router

        async def process_frame(self, frame, direction: "FrameDirection") -> None:
            await super().process_frame(frame, direction)

            if not isinstance(frame, TranscriptionFrame):
                # Pass everything else (audio, control, system frames) through.
                await self.push_frame(frame, direction)
                return

            transcript = (frame.text or "").strip()

            # SAFETY: e-stop is handled here and short-circuits the LLM. It can
            # never reach a motion primitive — the router only fires stop_hook.
            if self._router.route(transcript):
                # Speak a short confirmation; do not call the LLM.
                await self.push_frame(LLMFullResponseStartFrame())
                await self.push_frame(TextFrame("Emergency stop. Halting now."))
                await self.push_frame(LLMFullResponseEndFrame())
                return

            if not transcript:
                return

            # Stream the LLM reply as TTS text frames for low time-to-audio.
            await self.push_frame(LLMFullResponseStartFrame())
            async for delta in self._llm.generate(transcript):
                await self.push_frame(TextFrame(delta))
            await self.push_frame(LLMFullResponseEndFrame())

    return ArvitLLMProcessor()


def build_pipeline(
    transport,
    config: Optional[VoiceConfig] = None,
    *,
    llm: Optional[BedrockClaudeLLM] = None,
    stop_hook: Optional[Callable[[], None]] = None,
):
    """Assemble the full Pipecat pipeline around a WebSocket *transport*.

    pipecat + faster-whisper + piper are imported here (lazily). Returns a
    ``(pipeline, task)`` pair ready to run via ``PipelineRunner``.
    """
    config = config or VoiceConfig.from_env()

    from pipecat.audio.vad.silero import SileroVADAnalyzer
    from pipecat.pipeline.pipeline import Pipeline
    from pipecat.pipeline.task import PipelineParams, PipelineTask
    from pipecat.services.whisper.stt import WhisperSTTService
    from pipecat.services.piper.tts import PiperTTSService

    stt = WhisperSTTService(
        model=config.whisper_model,
        device=config.whisper_device,
        compute_type=config.whisper_compute_type,
    )

    tts = PiperTTSService(
        voice_name=config.piper_voice,
        model_path=config.piper_model_path,
        sample_rate=config.sample_rate,
    )

    llm_stage = make_arvit_llm_processor(llm=llm, stop_hook=stop_hook)

    pipeline = Pipeline(
        [
            transport.input(),   # WS audio in (VAD configured on the transport)
            stt,                 # faster-whisper (local)
            llm_stage,           # e-stop gate + Bedrock Claude (cloud, streamed)
            tts,                 # Piper (local)
            transport.output(),  # WS audio out
        ]
    )

    task = PipelineTask(
        pipeline,
        params=PipelineParams(
            allow_interruptions=config.allow_interruptions,  # barge-in
            audio_in_sample_rate=config.sample_rate,
            audio_out_sample_rate=config.sample_rate,
        ),
    )
    return pipeline, task


def make_vad_analyzer():
    """Silero VAD analyzer for the transport (barge-in). Imports pipecat lazily."""
    from pipecat.audio.vad.silero import SileroVADAnalyzer

    return SileroVADAnalyzer()
