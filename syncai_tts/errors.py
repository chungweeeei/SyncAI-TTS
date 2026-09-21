"""Domain errors, and the machine-readable codes that ride beside their prose.

The shape is deliberately the one ``syncai_backend`` already publishes: a JSON
body of ``{"detail": "<prose>", "code": "<stable string>"}``. That is how
``ConflictError.code`` is rendered there, and how ``gateways/failure.py`` tags a
gateway message, so the backend's replacement ``TtsGateway`` can read ``code``
off this service's response and re-tag it with its own ``Failure`` enum without
either side matching on a sentence.

``unknown_voice`` in particular is load-bearing across the two repos: it is the
one TTS failure that is the caller's to fix, so the backend's REST layer answers
400 instead of its uniform 502 **and** the Temporal SPEAK activity marks the
attempt non-retryable. Both read the code. Keep the string.
"""

from enum import Enum
from typing import Optional


class Failure(str, Enum):
    """Stable discriminators for the failures a caller answers differently."""

    # The caller's mistake: a voice this model does not carry. 400 here, 400 in
    # the backend's router, non-retryable for a SPEAK step. Same string as
    # syncai_backend's Failure.UNKNOWN_VOICE — do not rename.
    UNKNOWN_VOICE = "unknown_voice"

    # The weights are not on disk, or the session would not build. The service
    # cannot do its job at all; the caller should not retry in a tight loop.
    MODEL_UNAVAILABLE = "model_unavailable"

    # Kokoro raised while rendering this text. Possibly transient.
    SYNTHESIS_FAILED = "synthesis_failed"

    # `aplay` is not installed (alsa-utils missing from the image).
    PLAYER_UNAVAILABLE = "player_unavailable"

    # aplay ran and exited non-zero, or never exited. Device trouble.
    PLAYBACK_FAILED = "playback_failed"
    PLAYBACK_TIMEOUT = "playback_timeout"

    # More utterances are already waiting than TTS_MAX_QUEUE allows.
    QUEUE_FULL = "queue_full"

    # No job with that id: never existed, or aged out of the history window.
    JOB_NOT_FOUND = "job_not_found"


class TtsError(Exception):
    """Base for everything the API answers with a structured body.

    Carries its own status code so the raise site — not a mapping table in the
    router — decides what the failure means. There are few enough of them that
    a table would only add a place for the two to disagree.
    """

    status_code = 500

    def __init__(
        self,
        detail: str,
        code: Optional[Failure] = None,
        status_code: Optional[int] = None,
    ):
        super().__init__(detail)
        self.detail = detail
        self.code = code
        if status_code is not None:
            self.status_code = status_code


class BadRequestError(TtsError):
    """The request is wrong in a way the caller can fix (400)."""

    status_code = 400


class NotFoundError(TtsError):
    """No such job (404)."""

    status_code = 404


class ConflictError(TtsError):
    """The request contradicts current state, e.g. a full queue (409)."""

    status_code = 409


class EngineError(TtsError):
    """Synthesis ran and failed (500)."""

    status_code = 500


class UnavailableError(TtsError):
    """A dependency this service needs is absent: weights, aplay (503)."""

    status_code = 503
