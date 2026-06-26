"""ARVIT voice teammate: bidirectional real-time voice (STT -> LLM -> TTS).

This package is layered so the safety-critical and cloud pieces stay importable
WITHOUT pipecat / whisper / piper installed:

  - ``estop``        : pure e-stop utterance detection + router (no deps).
  - ``bedrock_llm``  : Bedrock Claude streaming wrapper (only needs anthropic).
  - ``pipeline``     : the Pipecat pipeline (imports pipecat lazily).
  - ``main``         : the FastAPI entrypoint (imports the pipeline lazily).

Only ``pipeline`` / ``main`` require the heavy real-time stack; the e-stop and
LLM modules can be imported and unit-tested on a laptop with no robot, no audio,
and no network.
"""

from .estop import EstopRouter, is_estop_utterance

__all__ = ["EstopRouter", "is_estop_utterance"]
