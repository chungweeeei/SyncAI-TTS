"""Settings, read from the environment exactly once.

Every value is read inside :meth:`Settings.from_env`, never at module import.
That is not a style preference: ``syncai_backend/temporal/shared.py`` reads
``TEMPORAL_ADDRESS`` at import time and ``main.py`` imports it before calling
``load_dotenv()``, so a value present only in ``.env`` is silently ignored and
the default is used instead. Reading inside a function called after
``load_dotenv()`` makes that class of bug structurally impossible here.
"""

import os
from dataclasses import dataclass
from typing import List, Optional

# The wire contract for a request's text length, shared by /synthesize and
# /speak. A constant rather than a setting: it is part of the API pydantic
# validates against (and the same 1000 the backend's SynthesizeRequest uses), so
# it cannot vary per deployment without the two contracts drifting.
MAX_TEXT_LENGTH = 1000
MIN_SPEED = 0.5
MAX_SPEED = 2.0


def _env_str(name: str, default: str) -> str:
    value = os.getenv(name)
    return default if value is None or value.strip() == "" else value.strip()


def _env_path(name: str, default: str) -> str:
    return os.path.expanduser(_env_str(name, default))


def _env_int(name: str, default: int) -> int:
    raw = _env_str(name, str(default))
    try:
        return int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer, got {raw!r}") from exc


def _env_float(name: str, default: float) -> float:
    raw = _env_str(name, str(default))
    try:
        return float(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be a number, got {raw!r}") from exc


def _env_bool(name: str, default: bool) -> bool:
    raw = _env_str(name, "true" if default else "false").lower()
    if raw in ("1", "true", "yes", "on"):
        return True
    if raw in ("0", "false", "no", "off"):
        return False
    raise ValueError(f"{name} must be a boolean, got {raw!r}")


def _env_list(name: str, default: Optional[List[str]] = None) -> List[str]:
    raw = _env_str(name, "")
    if not raw:
        return list(default or [])
    return [item.strip() for item in raw.split(",") if item.strip()]


@dataclass(frozen=True)
class Settings:
    host: str
    port: int
    cors_origins: List[str]

    model_path: str
    voices_path: str
    intra_op_threads: int
    preload: bool
    default_voice: str

    speaker_pcm_link: str
    fallback_device: str

    max_queue: int
    job_history: int
    playback_timeout_margin: float

    log_json: bool

    @classmethod
    def from_env(cls) -> "Settings":
        """Build settings from the current environment. Call after load_dotenv()."""
        settings = cls(
            host=_env_str("TTS_HOST", "0.0.0.0"),
            port=_env_int("TTS_PORT", 8080),
            cors_origins=_env_list("TTS_CORS_ORIGINS"),
            model_path=_env_path(
                "TTS_MODEL_PATH", "~/robot_ws/models/kokoro/kokoro-v1.0.onnx"
            ),
            voices_path=_env_path(
                "TTS_VOICES_PATH", "~/robot_ws/models/kokoro/voices-v1.0.bin"
            ),
            intra_op_threads=_env_int("TTS_INTRA_OP_THREADS", 4),
            preload=_env_bool("TTS_PRELOAD", True),
            default_voice=_env_str("TTS_DEFAULT_VOICE", "af_heart"),
            speaker_pcm_link=_env_str("TTS_SPEAKER_PCM_LINK", "/dev/syncai/speaker_pcm"),
            fallback_device=_env_str("TTS_FALLBACK_DEVICE", "plughw:CARD=CD002AUDIO,DEV=0"),
            max_queue=_env_int("TTS_MAX_QUEUE", 8),
            job_history=_env_int("TTS_JOB_HISTORY", 64),
            playback_timeout_margin=_env_float("TTS_PLAYBACK_TIMEOUT_MARGIN", 10.0),
            log_json=_env_bool("TTS_LOG_JSON", False),
        )
        settings.validate()
        return settings

    def validate(self) -> None:
        """Fail at startup rather than on the request that first needs the value."""
        if not 1 <= self.port <= 65535:
            raise ValueError(f"TTS_PORT out of range: {self.port}")
        if self.intra_op_threads < 1:
            raise ValueError(f"TTS_INTRA_OP_THREADS must be >= 1: {self.intra_op_threads}")
        if self.max_queue < 1:
            raise ValueError(f"TTS_MAX_QUEUE must be >= 1: {self.max_queue}")
        if self.job_history < 1:
            raise ValueError(f"TTS_JOB_HISTORY must be >= 1: {self.job_history}")
        if self.playback_timeout_margin < 0:
            raise ValueError(
                f"TTS_PLAYBACK_TIMEOUT_MARGIN must be >= 0: {self.playback_timeout_margin}"
            )
