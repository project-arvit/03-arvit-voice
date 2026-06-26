"""Pure, pipecat-free e-stop detection for the ARVIT voice teammate.

Safety contract (see the vault's voice-interface / mcp-robot-control notes):

  * An "estop" / "stop" / "halt" / "emergency stop" utterance MUST map directly
    to the hard-stop hook (e.g. the Go2 ``Damp`` tool).
  * A misheard command MUST NEVER trigger motion. This module therefore only
    ever invokes a *stop* callback. It has no notion of, and no path to, motion.
  * Transcribed audio is untrusted input: detection is conservative on the
    "is this a stop?" question (word-boundary matching, case-insensitive,
    punctuation-tolerant) so it fires on genuine stop words but does not fire on
    incidental substrings like "stopwatch" or "nonstop".

This module has ZERO third-party dependencies and is importable without
pipecat, whisper, piper, or any cloud SDK.
"""

from __future__ import annotations

import re
from typing import Callable, Optional

__all__ = ["is_estop_utterance", "EstopRouter", "ESTOP_PHRASES"]

# Canonical stop phrases. Multi-word phrases are matched as adjacent words
# (whitespace/punctuation tolerant); single words are matched on word
# boundaries so "stopwatch"/"nonstop" do NOT trigger.
ESTOP_PHRASES: tuple[str, ...] = (
    "emergency stop",
    "e-stop",
    "estop",
    "stop",
    "halt",
)

# Each phrase becomes a regex alternative. We sort longest-first so the engine
# prefers the most specific phrase, then join with word boundaries.
#
#   - "emergency stop" -> r"emergency\s+stop"
#   - "e-stop"         -> r"e[-\s]?stop"  (hyphen optional, also matches "e stop")
#   - "estop"          -> r"estop"
#   - "stop" / "halt"  -> plain words
#
# The whole thing is wrapped in \b ... \b so it only matches whole tokens.
def _build_pattern() -> "re.Pattern[str]":
    parts: list[str] = [
        r"emergency\s+stop",
        r"e[-\s]?stop",   # covers "e-stop", "e stop", "estop"
        r"stop",
        r"halt",
    ]
    # \b on both sides anchors to word boundaries so substrings inside larger
    # words ("stopwatch", "nonstop", "halting") are rejected.
    alternation = "|".join(parts)
    return re.compile(rf"\b(?:{alternation})\b", re.IGNORECASE)


_ESTOP_RE = _build_pattern()


def is_estop_utterance(text: Optional[str]) -> bool:
    """Return True iff *text* contains a recognized emergency-stop command.

    Robust to case, surrounding punctuation, and filler ("uh, stop!" -> True).
    Conservative against substrings ("stopwatch" / "nonstop" -> False).

    >>> is_estop_utterance("STOP")
    True
    >>> is_estop_utterance("Robot, emergency stop now")
    True
    >>> is_estop_utterance("e-stop")
    True
    >>> is_estop_utterance("estop!")
    True
    >>> is_estop_utterance("start the stopwatch")
    False
    >>> is_estop_utterance("walk to the next waypoint")
    False
    >>> is_estop_utterance(None)
    False
    """
    if not text:
        return False
    return _ESTOP_RE.search(text) is not None


class EstopRouter:
    """Routes transcripts to a hard-stop callback. NEVER triggers motion.

    Wire it once with the platform's hard-stop hook (the function that issues
    the e-stop / ``Damp`` command). Feed it every final STT transcript via
    :meth:`route`; on a recognized stop utterance it invokes the stop callback
    exactly once per utterance and reports that it handled the transcript.

    The router has no reference to any motion primitive by construction, so a
    misheard command cannot cause the robot to move. The worst-case failure mode
    is a spurious *stop*, which is the safe direction.

    Parameters
    ----------
    stop_callback:
        Zero-arg callable invoked on an e-stop utterance. Should be fast and
        non-blocking (e.g. enqueue the Damp command); exceptions are propagated
        to the caller so the pipeline can log/alarm, but detection state is not
        corrupted.
    """

    def __init__(self, stop_callback: Callable[[], None]) -> None:
        if not callable(stop_callback):
            raise TypeError("stop_callback must be callable")
        self._stop_callback = stop_callback
        self._stop_count = 0

    @property
    def stop_count(self) -> int:
        """Number of times the stop callback has been invoked."""
        return self._stop_count

    def route(self, transcript: Optional[str]) -> bool:
        """Inspect *transcript*; fire the stop callback if it is an e-stop.

        Returns True iff the transcript was an e-stop utterance and was routed
        to the stop callback. Returns False for normal speech (the caller then
        forwards the transcript to the LLM stage as usual). It NEVER initiates
        motion under any input.
        """
        if is_estop_utterance(transcript):
            self._stop_count += 1
            self._stop_callback()
            return True
        return False
