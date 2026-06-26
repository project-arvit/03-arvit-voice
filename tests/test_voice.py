"""Unit tests for the ARVIT voice teammate — RUN NOW, no audio/network/robot.

These tests deliberately import ONLY the pipecat-free modules
(``arvit_voice.estop`` and ``arvit_voice.bedrock_llm``) and inject a fake
Bedrock client, so they pass with just ``anthropic[bedrock]`` + ``pytest``
installed. pipecat / faster-whisper / piper are NOT required here.

Coverage:
  * BedrockClaudeLLM builds the correct request (model id from env, adaptive
    thinking present, NO temperature/top_p/top_k/budget_tokens, streaming) and
    yields text deltas.
  * is_estop_utterance + EstopRouter route "stop"/"estop"/etc. to the stop
    callback and do nothing (and never trigger motion) for normal speech.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from arvit_voice.bedrock_llm import (
    DEFAULT_MAX_TOKENS,
    DEFAULT_MODEL_ID,
    BedrockClaudeLLM,
)
from arvit_voice.estop import EstopRouter, is_estop_utterance


# --------------------------------------------------------------------------- #
# Fake Bedrock streaming client (no AWS, no network).
# --------------------------------------------------------------------------- #
def _text_delta_event(text: str) -> SimpleNamespace:
    """Mimic a RawContentBlockDeltaEvent carrying a text_delta."""
    return SimpleNamespace(
        type="content_block_delta",
        delta=SimpleNamespace(type="text_delta", text=text),
    )


def _noise_events() -> list[SimpleNamespace]:
    """Events the wrapper must ignore: message lifecycle + thinking deltas."""
    return [
        SimpleNamespace(type="message_start"),
        SimpleNamespace(type="content_block_start"),
        # A thinking delta must NOT be surfaced as spoken text.
        SimpleNamespace(
            type="content_block_delta",
            delta=SimpleNamespace(type="thinking_delta", thinking="(reasoning)"),
        ),
    ]


class _FakeStream:
    """Async context manager + async iterator over a fixed event list."""

    def __init__(self, events):
        self._events = events

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    def __aiter__(self):
        return self._aiter()

    async def _aiter(self):
        for ev in self._events:
            yield ev


class _FakeMessages:
    def __init__(self, recorder, events):
        self._recorder = recorder
        self._events = events

    def stream(self, **kwargs):
        # Record the EXACT request kwargs for assertions.
        self._recorder["request"] = kwargs
        return _FakeStream(self._events)


class _FakeClient:
    """Stands in for AsyncAnthropicBedrockMantle."""

    def __init__(self, recorder, events, **init_kwargs):
        self._recorder = recorder
        recorder["init_kwargs"] = init_kwargs
        self.messages = _FakeMessages(recorder, events)


def _make_llm(recorder, events, **llm_kwargs) -> BedrockClaudeLLM:
    def factory(**init_kwargs):
        return _FakeClient(recorder, events, **init_kwargs)

    return BedrockClaudeLLM(client_factory=factory, **llm_kwargs)


async def _collect(llm: BedrockClaudeLLM, text: str) -> list[str]:
    return [d async for d in llm.generate(text)]


# --------------------------------------------------------------------------- #
# BedrockClaudeLLM request shape.
# --------------------------------------------------------------------------- #
def test_request_uses_env_model_id(monkeypatch):
    # Pin every Bedrock env var so an ambient .env in the dev shell can't leak
    # a real key/region/model into this assertion. BEDROCK_MODEL/REGION are the
    # project deploy names; clear the bearer key so init_kwargs is region-only.
    monkeypatch.setenv("BEDROCK_MODEL", "global.anthropic.claude-opus-4-8-custom")
    monkeypatch.setenv("BEDROCK_REGION", "eu-west-1")
    monkeypatch.delenv("BEDROCK_API_KEY", raising=False)
    monkeypatch.delenv("AWS_BEARER_TOKEN_BEDROCK", raising=False)
    recorder: dict = {}
    llm = _make_llm(recorder, [_text_delta_event("hi")])

    req = llm.build_request("hello robodog")

    assert req["model"] == "global.anthropic.claude-opus-4-8-custom"
    # Region is pinned on the client (not on the request). With no bearer key
    # set, only the region is passed; a key (when present) is added as api_key.
    assert recorder["init_kwargs"] == {"aws_region": "eu-west-1"}


def test_bearer_key_is_passed_as_api_key(monkeypatch):
    # A Bedrock bearer key (BEDROCK_API_KEY) must be forwarded to the client as
    # api_key (sent as Authorization: Bearer ...); region stays pinned too.
    monkeypatch.setenv("BEDROCK_REGION", "eu-central-1")
    monkeypatch.setenv("BEDROCK_API_KEY", "ABSK-test-token")
    recorder: dict = {}
    _make_llm(recorder, [_text_delta_event("hi")])
    assert recorder["init_kwargs"] == {
        "aws_region": "eu-central-1",
        "api_key": "ABSK-test-token",
    }


def test_request_defaults_to_bedrock_prefixed_model():
    recorder: dict = {}
    llm = _make_llm(recorder, [_text_delta_event("hi")], model_id=None, region="eu-central-1")
    # The default model id carries the Bedrock 'anthropic.' prefix.
    assert llm.model_id == DEFAULT_MODEL_ID
    assert DEFAULT_MODEL_ID.startswith("anthropic.")


def test_request_has_adaptive_thinking_and_effort():
    recorder: dict = {}
    llm = _make_llm(recorder, [_text_delta_event("hi")], effort="medium")
    req = llm.build_request("status?")

    assert req["thinking"] == {"type": "adaptive"}
    assert req["output_config"] == {"effort": "medium"}
    assert req["max_tokens"] == DEFAULT_MAX_TOKENS


def test_request_omits_forbidden_sampling_params():
    """temperature/top_p/top_k/budget_tokens 400 on Opus 4.8 — must be absent."""
    recorder: dict = {}
    llm = _make_llm(recorder, [_text_delta_event("hi")])
    req = llm.build_request("walk forward")

    for forbidden in ("temperature", "top_p", "top_k"):
        assert forbidden not in req, f"{forbidden} must not be in the request"
    # budget_tokens must not appear anywhere in the thinking config either.
    assert "budget_tokens" not in req.get("thinking", {})


def test_generate_streams_and_yields_text_deltas_only():
    recorder: dict = {}
    events = (
        _noise_events()
        + [_text_delta_event("Inspecting "), _text_delta_event("gauge three.")]
    )
    llm = _make_llm(recorder, events)

    out = asyncio.run(_collect(llm, "check the gauge"))

    # Only the two text deltas are surfaced; thinking/lifecycle events dropped.
    assert out == ["Inspecting ", "gauge three."]

    # The streaming path was actually used, with the right request kwargs.
    req = recorder["request"]
    assert req["model"] == llm.model_id
    assert req["thinking"] == {"type": "adaptive"}
    assert req["output_config"]["effort"] == llm.effort
    assert req["messages"] == [{"role": "user", "content": "check the gauge"}]
    assert "temperature" not in req and "top_p" not in req and "top_k" not in req


# --------------------------------------------------------------------------- #
# E-stop detection + routing (safety-critical).
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "utterance",
    [
        "stop",
        "STOP",
        "Stop!",
        "estop",
        "e-stop",
        "e stop",
        "emergency stop",
        "Robot, emergency stop now",
        "uh, stop please",
        "halt",
        "HALT the robot",
    ],
)
def test_is_estop_utterance_true(utterance):
    assert is_estop_utterance(utterance) is True


@pytest.mark.parametrize(
    "utterance",
    [
        "start the stopwatch",
        "nonstop walking",
        "halting problem",  # 'halting' must not match 'halt'
        "go to the next waypoint",
        "inspect gauge three",
        "",
        None,
    ],
)
def test_is_estop_utterance_false(utterance):
    assert is_estop_utterance(utterance) is False


def test_router_fires_stop_callback_on_estop():
    fired = {"count": 0}

    def stop_cb():
        fired["count"] += 1

    router = EstopRouter(stop_cb)

    assert router.route("emergency stop") is True
    assert router.route("STOP") is True
    assert fired["count"] == 2
    assert router.stop_count == 2


def test_router_ignores_normal_speech_and_never_moves():
    """The router must do NOTHING for normal speech and has no motion path.

    We assert the stop callback is not called AND that the router object exposes
    no motion-capable surface — its only effect on any input is (maybe) a stop.
    """
    events: list[str] = []

    router = EstopRouter(lambda: events.append("STOP"))

    for normal in ("inspect gauge three", "walk to waypoint two", "what's the status"):
        assert router.route(normal) is False

    assert events == []  # no stop fired
    assert router.stop_count == 0
    # No motion-ish method exists on the router by construction.
    motion_like = [a for a in dir(router) if any(k in a.lower() for k in ("move", "walk", "go", "motion", "drive"))]
    assert motion_like == []


def test_router_rejects_non_callable():
    with pytest.raises(TypeError):
        EstopRouter("not callable")  # type: ignore[arg-type]
