"""The kokoro-onnx session: load once, render text to WAV bytes.

Ported from ``syncai_backend``'s ``TtsGateway``, with two changes.

Failures raise instead of returning ``(success, message, payload)``. That tuple
convention exists in the backend because one gateway instance is called by both
a REST router and a Temporal activity in the same process, and neither wanted
the other's exception types; here there is one caller, the API, and an exception
carrying its own status code and :class:`Failure` is less to thread through.

The session and the speaker are now separate objects. The gateway held two locks
for that reason already (``_lock`` for the session, ``_playback_lock`` for the
device); this module keeps the first and :mod:`syncai_tts.player` owns the
second. A caller that only wants bytes is still never made to wait out somebody
else's utterance.
"""

import os
import threading
import wave
from dataclasses import dataclass
from io import BytesIO
from typing import List, Optional

import numpy as np
import structlog

from syncai_tts.config import Settings
from syncai_tts.errors import (
    BadRequestError,
    EngineError,
    Failure,
    TtsError,
    UnavailableError,
)


# What /health reports about the session, so an operator can tell "still
# loading" from "the weights are not there" without reading the log.
STATE_UNLOADED = "unloaded"
STATE_LOADING = "loading"
STATE_READY = "ready"
STATE_MISSING = "missing"
STATE_ERROR = "error"


@dataclass(frozen=True)
class Utterance:
    """Rendered audio, ready to hand to the speaker or to a HTTP response."""

    wav: bytes
    duration_s: float
    sample_rate: int


class KokoroEngine:
    def __init__(self, logger: structlog.stdlib.BoundLogger, settings: Settings):
        self._logger = logger
        self._settings = settings

        self._kokoro = None
        self._state = STATE_UNLOADED
        self._error: Optional[str] = None

        # Covers load-then-infer, so two first callers cannot build two sessions
        # and synthesis stays single-threaded on a machine whose cores are
        # spoken for. Held across `create()`, which is a few hundred ms.
        self._lock = threading.Lock()

        # Read without the lock by /health, so state and error are only ever
        # assigned as whole strings under it — never mutated in place.
        self._state_lock = threading.Lock()

        self._logger.info(
            "[KokoroEngine] Using kokoro model",
            model=self._settings.model_path,
            voices=self._settings.voices_path,
        )

    # --- State ---------------------------------------------------------------

    @property
    def state(self) -> str:
        with self._state_lock:
            return self._state

    @property
    def error(self) -> Optional[str]:
        with self._state_lock:
            return self._error

    def _set_state(self, state: str, error: Optional[str] = None) -> None:
        with self._state_lock:
            self._state = state
            self._error = error

    # --- Loading -------------------------------------------------------------

    def preload(self) -> None:
        """Build the session now, swallowing failure into the reported state.

        Used by the startup thread: a missing model must not stop the service
        from serving /health, which is the endpoint that would explain it.
        """
        try:
            self.ensure_loaded()
        except TtsError:
            # Already recorded in `state`/`error` and logged by ensure_loaded.
            # Raising out of a daemon thread would only print a second traceback
            # nobody can act on differently.
            pass

    def preload_in_background(self) -> threading.Thread:
        """Load in a daemon thread so uvicorn starts serving immediately.

        /health answers `loading` in the meantime, and a request that arrives
        first simply blocks on the same lock rather than building a second
        session.
        """
        thread = threading.Thread(
            target=self.preload, name="kokoro-preload", daemon=True
        )
        thread.start()
        return thread

    def ensure_loaded(self) -> None:
        """Build the kokoro session if it is not built yet.

        Raises :class:`UnavailableError` if the weights are missing or the
        session will not build — the caller cannot fix either, so it is a 503
        rather than a 500, and the message names the path we looked at.
        """
        with self._lock:
            if self._kokoro is not None:
                return

            for path in (self._settings.model_path, self._settings.voices_path):
                if not os.path.isfile(path):
                    detail = (
                        f"kokoro model file missing: {path} — download it from the "
                        "kokoro-onnx 'model-files' GitHub release (URLs in "
                        ".env.example)"
                    )
                    self._set_state(STATE_MISSING, detail)
                    raise UnavailableError(detail, code=Failure.MODEL_UNAVAILABLE)

            self._set_state(STATE_LOADING)
            try:
                # Imported here, not at module scope: the import alone pulls in
                # onnxruntime, and the test suite fakes this session so it can
                # run on a laptop with neither onnxruntime nor the weights.
                import onnxruntime as ort
                from kokoro_onnx import Kokoro

                opts = ort.SessionOptions()
                opts.intra_op_num_threads = self._settings.intra_op_threads
                session = ort.InferenceSession(
                    self._settings.model_path, opts, providers=["CPUExecutionProvider"]
                )
                self._kokoro = Kokoro.from_session(session, self._settings.voices_path)
            except Exception as exc:
                detail = f"failed to load kokoro model: {exc}"
                self._set_state(STATE_ERROR, detail)
                self._logger.error("[KokoroEngine] Model load failed", error=str(exc))
                raise UnavailableError(detail, code=Failure.MODEL_UNAVAILABLE)

            self._set_state(STATE_READY)
            self._logger.info("[KokoroEngine] Kokoro model loaded")

    # --- API -----------------------------------------------------------------

    def voices(self) -> List[str]:
        self.ensure_loaded()
        with self._lock:
            return sorted(self._kokoro.get_voices())

    def synthesize(self, text: str, voice: str, speed: float) -> Utterance:
        """Render text to a mono 16-bit WAV.

        An unknown voice is the one failure here that is the caller's to fix, so
        it is a 400 tagged :attr:`Failure.UNKNOWN_VOICE` rather than the 500 the
        rest get. The backend's REST layer and its SPEAK activity both key off
        that code — see :mod:`syncai_tts.errors`.
        """
        self.ensure_loaded()

        with self._lock:
            if voice not in self._kokoro.get_voices():
                raise BadRequestError(
                    f"unknown voice: {voice!r}", code=Failure.UNKNOWN_VOICE
                )

            try:
                samples, sample_rate = self._kokoro.create(text, voice=voice, speed=speed)
            except Exception as exc:
                self._logger.error("[KokoroEngine] Synthesis failed", error=str(exc))
                raise EngineError(
                    f"synthesis failed: {exc}", code=Failure.SYNTHESIS_FAILED
                )

        # Encoding is pure numpy over a buffer we own, so it happens outside the
        # lock: there is nothing shared left to protect and it is the longest
        # part of a long utterance.
        samples = np.asarray(samples, dtype=np.float32)
        pcm = (np.clip(samples, -1.0, 1.0) * 32767.0).astype(np.int16)

        buffer = BytesIO()
        with wave.open(buffer, "wb") as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(sample_rate)
            wav.writeframes(pcm.tobytes())

        # Exact, from the sample count. The gateway this replaces derived it
        # from the encoded length minus a 44-byte canonical header, which was
        # only ever "close enough for a timeout margin".
        duration_s = len(pcm) / float(sample_rate) if sample_rate else 0.0

        return Utterance(
            wav=buffer.getvalue(), duration_s=duration_s, sample_rate=int(sample_rate)
        )
