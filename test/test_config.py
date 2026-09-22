"""Settings, and the load-order property that makes .env actually work."""

import os

import pytest

from syncai_tts.config import Settings


def test_defaults_apply_when_nothing_is_set(monkeypatch):
    for name in [n for n in os.environ if n.startswith("TTS_")]:
        monkeypatch.delenv(name, raising=False)

    settings = Settings.from_env()

    assert settings.host == "0.0.0.0"
    assert settings.port == 8080
    assert settings.cors_origins == []
    assert settings.default_voice == "af_heart"
    assert settings.preload is True


def test_the_environment_is_read_when_from_env_is_called_not_at_import(monkeypatch):
    """The property that syncai_backend's temporal/shared.py does not have.

    There, TEMPORAL_ADDRESS is read at module import and main.py imports the
    module before load_dotenv(), so a value present only in .env is silently
    ignored. Reading inside this call is what makes a later-loaded .env win.
    """
    import syncai_tts.config as config_module  # already imported by now

    monkeypatch.setenv("TTS_PORT", "9999")
    settings = config_module.Settings.from_env()

    assert settings.port == 9999


def test_paths_are_expanded(monkeypatch):
    monkeypatch.setenv("TTS_MODEL_PATH", "~/models/kokoro.onnx")
    settings = Settings.from_env()
    assert settings.model_path.startswith("/")
    assert "~" not in settings.model_path


def test_cors_origins_are_a_comma_separated_list(monkeypatch):
    monkeypatch.setenv("TTS_CORS_ORIGINS", "http://a.local, http://b.local ,")
    settings = Settings.from_env()
    assert settings.cors_origins == ["http://a.local", "http://b.local"]


@pytest.mark.parametrize(
    "value,expected",
    [("true", True), ("1", True), ("on", True), ("false", False), ("0", False)],
)
def test_booleans_accept_the_usual_spellings(monkeypatch, value, expected):
    monkeypatch.setenv("TTS_PRELOAD", value)
    assert Settings.from_env().preload is expected


def test_a_bad_value_fails_at_startup_not_on_the_first_request(monkeypatch):
    monkeypatch.setenv("TTS_PORT", "not-a-port")
    with pytest.raises(ValueError, match="TTS_PORT"):
        Settings.from_env()


def test_an_out_of_range_value_is_rejected(monkeypatch):
    monkeypatch.setenv("TTS_MAX_QUEUE", "0")
    with pytest.raises(ValueError, match="TTS_MAX_QUEUE"):
        Settings.from_env()
