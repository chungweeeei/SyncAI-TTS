"""Shared fixtures.

Nothing in this suite needs onnxruntime, kokoro-onnx, a 310 MB model file, ALSA
or a sound card. The kokoro session is a fake object assigned onto the engine
(so ``ensure_loaded`` short-circuits) and ``aplay`` is a fake process class
patched over ``subprocess.Popen``. That is deliberate and worth keeping: the
service this repo replaced could only be tested inside the robot container,
because its module graph reached ROS, and the result was that the one thing it
owned — who is allowed to touch the speaker — had almost no coverage.
"""

import subprocess
import threading
import time
from typing import List, Optional

import numpy as np
import pytest
import structlog

from syncai_tts.config import Settings
from syncai_tts.engine import STATE_READY, KokoroEngine
from syncai_tts.player import SpeechPlayer


# --- Settings ---------------------------------------------------------------


def make_settings(**overrides) -> Settings:
    """A Settings with test-shaped defaults; override what a test cares about."""
    base = dict(
        host="127.0.0.1",
        port=8080,
        cors_origins=[],
        model_path="/nonexistent/kokoro-v1.0.onnx",
        voices_path="/nonexistent/voices-v1.0.bin",
        intra_op_threads=1,
        preload=False,
        default_voice="af_heart",
        speaker_pcm_link="/nonexistent/speaker_pcm",
        fallback_device="plughw:CARD=TEST,DEV=0",
        max_queue=4,
        job_history=8,
        # Short, so the timeout test does not take the real 10 s margin.
        playback_timeout_margin=0.3,
        log_json=False,
    )
    base.update(overrides)
    return Settings(**base)


@pytest.fixture
def settings() -> Settings:
    return make_settings()


@pytest.fixture
def logger():
    return structlog.get_logger()


# --- Fake kokoro session ----------------------------------------------------


class FakeKokoro:
    """Stands in for the loaded ONNX session."""

    def __init__(self, synth_delay: float = 0.0, samples: int = 2400):
        self.synth_delay = synth_delay
        self.samples = samples
        self.calls: List[tuple] = []

    def get_voices(self):
        return ["af_heart", "am_adam"]

    def create(self, text, voice, speed):
        self.calls.append((text, voice, speed))
        if self.synth_delay:
            time.sleep(self.synth_delay)
        # A tenth of a second of silence at kokoro's native rate.
        return np.zeros(self.samples, dtype=np.float32), 24000


@pytest.fixture
def kokoro() -> FakeKokoro:
    return FakeKokoro()


@pytest.fixture
def engine(logger, settings, kokoro) -> KokoroEngine:
    eng = KokoroEngine(logger=logger, settings=settings)
    eng._kokoro = kokoro
    eng._set_state(STATE_READY)
    return eng


# --- Fake aplay -------------------------------------------------------------


class FakeAplayProcess:
    """A stand-in for the aplay Popen: plays for `play_time`, or hangs."""

    def __init__(self, factory: "FakeAplay", argv: List[str]):
        self._factory = factory
        self.argv = argv
        self.returncode: Optional[int] = None
        self._stop = threading.Event()
        self._finished = False

    # -- the Popen surface the player uses --
    def communicate(self, input=None, timeout=None):
        factory = self._factory
        if self._finished:
            return None, factory.stderr

        with factory.lock:
            factory.inside += 1
            factory.max_inside = max(factory.max_inside, factory.inside)
        try:
            if factory.hang:
                # Never finishes on its own: only a terminate/kill, or the
                # player's own timeout, ends this.
                stopped = self._stop.wait(timeout=timeout)
                if not stopped:
                    raise subprocess.TimeoutExpired(self.argv, timeout)
            else:
                budget = (
                    factory.play_time
                    if timeout is None
                    else min(factory.play_time, timeout)
                )
                stopped = self._stop.wait(timeout=budget)
                if not stopped and timeout is not None and timeout < factory.play_time:
                    raise subprocess.TimeoutExpired(self.argv, timeout)

            self.returncode = -15 if stopped else factory.returncode
            self._finished = True
            return None, factory.stderr
        finally:
            with factory.lock:
                factory.inside -= 1

    def poll(self):
        return self.returncode

    def terminate(self):
        self._factory.terminated += 1
        self._stop.set()

    def kill(self):
        self._factory.killed += 1
        self._stop.set()
        self._finished = False  # the player calls communicate() again after kill

    def wait(self, timeout=None):
        if not self._stop.wait(timeout=timeout):
            raise subprocess.TimeoutExpired(self.argv, timeout)
        return self.returncode if self.returncode is not None else -15


class FakeAplay:
    """Factory patched over subprocess.Popen; records what the player did."""

    def __init__(
        self,
        play_time: float = 0.05,
        hang: bool = False,
        returncode: int = 0,
        stderr: bytes = b"",
        missing: bool = False,
    ):
        self.play_time = play_time
        self.hang = hang
        self.returncode = returncode
        self.stderr = stderr
        self.missing = missing

        self.lock = threading.Lock()
        self.inside = 0
        self.max_inside = 0
        self.calls: List[List[str]] = []
        self.devices: List[str] = []
        self.procs: List[FakeAplayProcess] = []
        self.terminated = 0
        self.killed = 0
        # When set to an Event, __call__ blocks in the middle of "spawning"
        # until a test releases it. That is how a test can hold the player in
        # the window between marking a job PLAYING and having a process to
        # signal, which is a window a cancel really does land in.
        self.gate: Optional[threading.Event] = None

    def __call__(self, argv, stdin=None, stdout=None, stderr=None):
        if self.missing:
            raise FileNotFoundError(argv[0])
        with self.lock:
            self.calls.append(argv)
            self.devices.append(argv[argv.index("-D") + 1])
        if self.gate is not None:
            self.gate.wait(timeout=5.0)
        proc = FakeAplayProcess(self, argv)
        self.procs.append(proc)
        return proc


@pytest.fixture
def aplay(monkeypatch) -> FakeAplay:
    from syncai_tts import player as player_module

    fake = FakeAplay()
    monkeypatch.setattr(player_module.subprocess, "Popen", fake)
    return fake


@pytest.fixture
def player(logger, settings, aplay):
    spk = SpeechPlayer(logger=logger, settings=settings)
    spk.start()
    try:
        yield spk
    finally:
        spk.stop(timeout=5.0)


# --- Helpers ----------------------------------------------------------------


def wait_until(predicate, timeout: float = 5.0, interval: float = 0.005) -> bool:
    """Poll `predicate` until it is true. Returns whether it became true."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(interval)
    return predicate()
