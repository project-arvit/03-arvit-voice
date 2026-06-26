"""Pipecat-free Bedrock Claude wrapper for the ARVIT voice teammate.

Only the LLM stage goes to the cloud; STT (faster-whisper) and TTS (Piper) stay
local on the Orin. This module owns the LLM stage and nothing else, so it has a
single third-party dependency (``anthropic[bedrock]``) and is importable and
unit-testable without pipecat / audio / a live AWS account.

Authoritative request shape (from the Anthropic Bedrock SDK + the ARVIT spec):

  * client  : ``AnthropicBedrockMantle`` (async variant for the pipeline).
  * model   : ``BEDROCK_MODEL_ID`` env, default ``anthropic.claude-opus-4-8``
              (Bedrock model IDs carry the ``anthropic.`` prefix).
  * thinking: ``{"type": "adaptive"}`` (adaptive thinking; supported on Bedrock).
  * depth   : controlled via ``output_config={"effort": ...}`` — NOT via
              ``budget_tokens``.
  * DO NOT pass ``temperature`` / ``top_p`` / ``top_k`` / ``budget_tokens``:
              they 400 on Opus 4.8.
  * streaming via ``client.messages.stream(...)``; ``max_tokens`` ~16000.
"""

from __future__ import annotations

import os
from typing import AsyncIterator, Optional

__all__ = ["BedrockClaudeLLM", "DEFAULT_MODEL_ID", "DEFAULT_MAX_TOKENS", "DEFAULT_EFFORT"]

DEFAULT_MODEL_ID = "anthropic.claude-opus-4-8"
DEFAULT_REGION = "eu-central-1"
DEFAULT_MAX_TOKENS = 16000
DEFAULT_EFFORT = "medium"

_DEFAULT_SYSTEM = (
    "You are ARVIT, the autonomous-inspection voice teammate riding on a "
    "Unitree Go2 quadruped. You are spoken to and you speak back, so keep "
    "replies short, concrete, and easy to hear over a noisy industrial site. "
    "Confirm any action you are about to take before describing results. You "
    "never command robot motion yourself; you only report and advise. Safety "
    "and stop commands are handled by a separate hard-stop layer."
)


class BedrockClaudeLLM:
    """Streaming Claude-on-Bedrock client for the voice loop.

    The heavy import (``anthropic``) is deferred to construction time so that
    merely importing this module is cheap and dependency-light at module scope.
    For tests, inject a fake client class via ``client_factory`` to assert the
    exact request without touching AWS.

    Parameters
    ----------
    model_id:
        Override the Bedrock model id (else ``BEDROCK_MODEL_ID`` env / default).
    region:
        AWS region (else ``AWS_REGION`` env / ``eu-central-1``).
    max_tokens:
        Output cap (default 16000, per spec).
    effort:
        ``output_config.effort`` — depth control. Default ``"medium"``.
    system:
        System prompt. Defaults to the ARVIT voice persona.
    client_factory:
        Callable returning a client exposing ``messages.stream(...)``. Defaults
        to ``anthropic.AsyncAnthropicBedrockMantle`` (imported lazily). Tests
        pass a fake here to capture the request kwargs.
    """

    def __init__(
        self,
        *,
        model_id: Optional[str] = None,
        region: Optional[str] = None,
        api_key: Optional[str] = None,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        effort: str = DEFAULT_EFFORT,
        system: str = _DEFAULT_SYSTEM,
        client_factory=None,
    ) -> None:
        # Project deploy vars are BEDROCK_MODEL / BEDROCK_REGION / BEDROCK_API_KEY
        # (see .env.example); the AWS_*/BEDROCK_MODEL_ID names are accepted as a
        # fallback. The bearer key, when set, becomes the Mantle client's
        # ``api_key`` (sent as ``Authorization: Bearer ...``); if unset, the SDK
        # resolves SigV4 creds from the standard chain.
        self.model_id = (
            model_id
            or os.environ.get("BEDROCK_MODEL")
            or os.environ.get("BEDROCK_MODEL_ID", DEFAULT_MODEL_ID)
        )
        self.region = (
            region
            or os.environ.get("BEDROCK_REGION")
            or os.environ.get("AWS_REGION", DEFAULT_REGION)
        )
        self.api_key = (
            api_key
            or os.environ.get("BEDROCK_API_KEY")
            or os.environ.get("AWS_BEARER_TOKEN_BEDROCK")
        )
        self.max_tokens = max_tokens
        self.effort = effort
        self.system = system

        if client_factory is None:
            # Lazy import: keep module import cheap and allow tests to inject.
            # Use the Bedrock Runtime client (bedrock-runtime.<region>.amazonaws.com,
            # the InvokeModel path) rather than the Mantle endpoint: the project's
            # `global.anthropic.claude-opus-4-8` cross-region inference profile is
            # served there, and the bearer key authenticates via Authorization:
            # Bearer. (The Mantle endpoint 404s the global. profile.)
            from anthropic import AsyncAnthropicBedrock

            client_factory = AsyncAnthropicBedrock

        # Pin the region; pass the bearer key only when present so the SigV4
        # path (and the injected-fake tests) still work with region alone.
        init_kwargs = {"aws_region": self.region}
        if self.api_key:
            init_kwargs["api_key"] = self.api_key
        self._client = client_factory(**init_kwargs)

    def build_request(self, text: str) -> dict:
        """Construct the exact kwargs passed to ``messages.stream(...)``.

        Split out so tests can assert the request shape directly, and so the
        forbidden parameters are visibly *absent* in one place.

        Notably ABSENT (these 400 on Opus 4.8): ``temperature``, ``top_p``,
        ``top_k``, and ``thinking.budget_tokens``. Depth is controlled only via
        ``output_config.effort``.
        """
        return {
            "model": self.model_id,
            "max_tokens": self.max_tokens,
            "system": self.system,
            "messages": [{"role": "user", "content": text}],
            # Adaptive thinking — no budget_tokens.
            "thinking": {"type": "adaptive"},
            # Depth control lives here, not in temperature/top_p/budget_tokens.
            "output_config": {"effort": self.effort},
        }

    async def generate(self, text: str) -> AsyncIterator[str]:
        """Stream the model's reply as token-delta strings.

        Yields each incremental text delta as it arrives so the TTS stage can
        begin speaking before generation finishes (latency budget). Thinking
        deltas are not surfaced — only assistant text is spoken.
        """
        request = self.build_request(text)
        async with self._client.messages.stream(**request) as stream:
            async for event in stream:
                # Only assistant *text* deltas are spoken. Thinking deltas,
                # message_start/stop, content_block_start/stop, etc. are skipped.
                if getattr(event, "type", None) == "content_block_delta":
                    delta = getattr(event, "delta", None)
                    if getattr(delta, "type", None) == "text_delta":
                        piece = getattr(delta, "text", "")
                        if piece:
                            yield piece
