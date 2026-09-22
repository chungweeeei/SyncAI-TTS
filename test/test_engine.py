"""Synthesis, and the one failure that is the caller's to fix."""

import threading
import time
import wave
from io import BytesIO

import pytest

from syncai_tts.engine import STATE_MISSING, STATE_READY, KokoroEngine
from syncai_tts.errors import BadRequestError, EngineError, Failure, UnavailableError

from .conftest import make_settings


def test_synthesis_returns_a_playable_mono_wav(engine):
    utterance = engine.synthesize(text="hello", voice="af_heart", speed=1.0)

    assert utterance.wav.startswith(b"RIFF")
    assert utterance.sample_rate == 24000

    with wave.open(BytesIO(utterance.wav), "rb") as wav:
        assert wav.getnchannels() == 1
        assert wav.getsampwidth() == 2
        assert wav.getframerate() == 24000
        assert wav.getnframes() == 2400


def test_duration_comes_from_the_sample_count(engine):
    """Exact, not the encoded-length-minus-a-44-byte-header approximation the
    in-process gateway used for its aplay timeout."""
    utterance = engine.synthesize(text="hello", voice="af_heart", speed=1.0)
    assert utterance.duration_s == pytest.approx(0.1)


def test_an_unknown_voice_is_the_callers_mistake(engine):
    """Tagged, not just worded. The backend's REST layer answers 400 off this
    code and its SPEAK activity marks the attempt non-retryable off the same
    one; matching prose in two places is how those drift apart on a reword."""
    with pytest.raises(BadRequestError) as excinfo:
        engine.synthesize(text="hello", voice="nosuchvoice", speed=1.0)

    assert excinfo.value.code is Failure.UNKNOWN_VOICE
    assert excinfo.value.status_code == 400
    assert "nosuchvoice" in excinfo.value.detail


def test_an_unknown_voice_never_renders_anything(engine, kokoro):
    with pytest.raises(BadRequestError):
        engine.synthesize(text="hello", voice="nosuchvoice", speed=1.0)
    assert kokoro.calls == []


def test_a_synthesis_error_is_ours_not_the_callers(engine, kokoro):
    def _explode(text, voice, speed):
        raise RuntimeError("phonemizer fell over")

    kokoro.create = _explode

    with pytest.raises(EngineError) as excinfo:
        engine.synthesize(text="hello", voice="af_heart", speed=1.0)

    assert excinfo.value.code is Failure.SYNTHESIS_FAILED
    assert excinfo.value.status_code == 500


def test_voices_are_sorted(engine):
    assert engine.voices() == ["af_heart", "am_adam"]


def test_missing_weights_are_a_503_naming_the_path_we_looked_at(logger):
    """The path is neither a parameter nor an env var the operator sees in the
    error otherwise, so the message is the only pointer to where we looked."""
    settings = make_settings(model_path="/nope/kokoro.onnx", voices_path="/nope/voices.bin")
    engine = KokoroEngine(logger=logger, settings=settings)

    with pytest.raises(UnavailableError) as excinfo:
        engine.voices()

    assert excinfo.value.code is Failure.MODEL_UNAVAILABLE
    assert excinfo.value.status_code == 503
    assert "/nope/kokoro.onnx" in excinfo.value.detail
    assert engine.state == STATE_MISSING
    assert engine.error is not None


def test_preload_records_failure_instead_of_raising(logger):
    """It runs in a daemon thread at startup; a missing model must leave the
    service able to serve /health, which is what explains it."""
    engine = KokoroEngine(logger=logger, settings=make_settings())

    engine.preload()  # must not raise

    assert engine.state == STATE_MISSING


def test_synthesis_is_serialised_by_the_session_lock(engine, kokoro):
    """The ONNX session is shared state and the Orin's cores are spoken for.

    The in-process gateway held this same lock across load-then-infer; keeping
    it here is what stops four REST callers from running four inferences at
    once on a machine that also has to run the robot.
    """
    inside = {"now": 0, "max": 0}
    counter_lock = threading.Lock()
    real_create = kokoro.create

    def _counting_create(text, voice, speed):
        with counter_lock:
            inside["now"] += 1
            inside["max"] = max(inside["max"], inside["now"])
        try:
            time.sleep(0.02)
            return real_create(text, voice, speed)
        finally:
            with counter_lock:
                inside["now"] -= 1

    kokoro.create = _counting_create

    threads = [
        threading.Thread(
            target=engine.synthesize,
            kwargs={"text": "hello", "voice": "af_heart", "speed": 1.0},
        )
        for _ in range(4)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=10.0)

    assert all(not thread.is_alive() for thread in threads)
    assert inside["max"] == 1
    assert len(kokoro.calls) == 4


def test_an_already_loaded_session_is_not_rebuilt(engine, kokoro):
    """ensure_loaded short-circuits, so the 310 MB load happens once."""
    engine.ensure_loaded()
    engine.ensure_loaded()
    assert engine._kokoro is kokoro
    assert engine.state == STATE_READY
