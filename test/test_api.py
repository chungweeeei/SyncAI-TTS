"""The REST surface, end to end over TestClient.

The app is built with the same create_app the entrypoint uses, so the exception
handlers, the lifespan (playback thread up, then down) and the routes are the
real ones. Only the kokoro session and aplay are fakes.
"""

import pytest

pytest.importorskip("httpx", reason="fastapi.testclient needs httpx")

from fastapi.testclient import TestClient  # noqa: E402

from syncai_tts.api.app import create_app  # noqa: E402
from syncai_tts.engine import STATE_MISSING, KokoroEngine  # noqa: E402
from syncai_tts.player import SpeechPlayer  # noqa: E402

from .conftest import make_settings, wait_until  # noqa: E402


@pytest.fixture
def client(logger, settings, engine, player):
    app = create_app(logger=logger, settings=settings, engine=engine, player=player)
    # The context manager runs the lifespan; the player fixture already started
    # the thread, and start()/stop() are both idempotent enough to overlap.
    with TestClient(app) as test_client:
        yield test_client


def job_done(client, job_id: str) -> bool:
    body = client.get(f"/api/v1/speak/{job_id}").json()
    return body["status"] in ("done", "failed", "cancelled")


# --- Voices and synthesis ---------------------------------------------------


def test_voices_lists_what_the_model_carries(client):
    response = client.get("/api/v1/voices")
    assert response.status_code == 200
    assert response.json() == {"voices": ["af_heart", "am_adam"]}


def test_synthesize_returns_the_wav_itself(client, aplay):
    response = client.post("/api/v1/synthesize", json={"text": "hello"})

    assert response.status_code == 200
    assert response.headers["content-type"] == "audio/wav"
    assert response.content.startswith(b"RIFF")
    assert float(response.headers["x-audio-duration"]) == pytest.approx(0.1)
    # Nothing was played.
    assert aplay.calls == []


def test_synthesize_rejects_an_unknown_voice_with_a_code(client):
    """400 with `unknown_voice`, which is what the backend's router turns back
    into its own 400 and its SPEAK activity reads to not retry."""
    response = client.post("/api/v1/synthesize", json={"text": "hi", "voice": "nope"})

    assert response.status_code == 400
    body = response.json()
    assert body["code"] == "unknown_voice"
    assert "nope" in body["detail"]


@pytest.mark.parametrize(
    "body",
    [
        {"text": "x" * 1001},
        {"text": ""},
        {"text": "hi", "speed": 3.0},
        {"text": "hi", "speed": 0.1},
    ],
)
@pytest.mark.parametrize("path", ["/api/v1/synthesize", "/api/v1/speak"])
def test_text_is_capped_and_speed_is_bounded(client, path, body):
    response = client.post(path, json=body)
    assert response.status_code == 422


@pytest.mark.parametrize("path", ["/api/v1/synthesize", "/api/v1/speak"])
def test_a_rejected_body_keeps_the_error_contract(client, path):
    """pydantic's rejections answer `{detail, code}` like every other error.

    FastAPI's default is a list of dicts under `detail` and no `code` at all, so
    a caller that reads `body["code"]` — which is what this service's contract
    tells it to do, and what syncai_backend's gateway does — breaks on the most
    reachable input error there is: text past the 1000-char cap.
    """
    response = client.post(path, json={"text": "x" * 1001})

    assert response.status_code == 422
    body = response.json()
    assert body["code"] == "invalid_request"
    assert isinstance(body["detail"], str)
    # The field is named, and the framework's "body" prefix is not.
    assert body["detail"].startswith("text: ")


def test_a_rejected_body_reports_every_bad_field(client):
    body = client.post(
        "/api/v1/speak", json={"text": "", "speed": 99.0}
    ).json()

    assert body["code"] == "invalid_request"
    assert "and 1 more" in body["detail"]


def test_the_default_voice_is_used_when_none_is_given(client, kokoro):
    client.post("/api/v1/synthesize", json={"text": "hello"})
    assert kokoro.calls[-1][1] == "af_heart"


# --- Speaking ---------------------------------------------------------------


def test_speak_accepts_the_utterance_and_answers_with_a_job(client, aplay):
    response = client.post("/api/v1/speak", json={"text": "hello"})

    assert response.status_code == 202
    body = response.json()
    assert body["status"] in ("queued", "playing", "done")
    assert body["duration"] == pytest.approx(0.1)
    assert body["voice"] == "af_heart"
    assert body["error"] is None

    assert wait_until(lambda: job_done(client, body["id"]))
    assert client.get(f"/api/v1/speak/{body['id']}").json()["status"] == "done"


def test_speak_with_wait_holds_the_response_until_it_is_spoken(client, aplay):
    """The backend's existing POST /api/v1/tts/speak contract is blocking, so
    this keeps that answer available while the job API is what a Temporal
    activity polls."""
    response = client.post("/api/v1/speak", json={"text": "hello", "wait": True})

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "done"
    assert body["finished_at"] is not None


def test_a_wait_that_times_out_answers_202_with_the_live_state(client, aplay):
    aplay.play_time = 5.0
    response = client.post(
        "/api/v1/speak", json={"text": "hello", "wait": True, "wait_timeout": 0.1}
    )

    assert response.status_code == 202
    assert response.json()["status"] in ("queued", "playing")


def test_speak_rejects_an_unknown_voice_before_queueing_anything(client, aplay):
    response = client.post("/api/v1/speak", json={"text": "hi", "voice": "nope"})

    assert response.status_code == 400
    assert response.json()["code"] == "unknown_voice"
    assert aplay.calls == []
    assert client.get("/api/v1/speak").json()["jobs"] == []


# --- Job lifecycle ----------------------------------------------------------


def test_an_unknown_job_is_a_404_with_a_code(client):
    response = client.get("/api/v1/speak/nosuchjob")
    assert response.status_code == 404
    assert response.json()["code"] == "job_not_found"


def test_a_job_can_be_cancelled_over_http(client, aplay):
    aplay.play_time = 5.0
    job_id = client.post("/api/v1/speak", json={"text": "hello"}).json()["id"]
    assert wait_until(
        lambda: client.get(f"/api/v1/speak/{job_id}").json()["status"] == "playing"
    )

    response = client.delete(f"/api/v1/speak/{job_id}")

    assert response.status_code == 200
    assert wait_until(lambda: job_done(client, job_id))
    assert client.get(f"/api/v1/speak/{job_id}").json()["status"] == "cancelled"


def test_listing_returns_recent_jobs_newest_first(client, aplay):
    first = client.post("/api/v1/speak", json={"text": "one"}).json()["id"]
    second = client.post("/api/v1/speak", json={"text": "two"}).json()["id"]
    assert wait_until(lambda: job_done(client, first) and job_done(client, second))

    jobs = client.get("/api/v1/speak").json()["jobs"]
    assert [job["id"] for job in jobs] == [second, first]
    assert jobs[0]["text"] == "two"


def test_a_full_queue_answers_409_rather_than_promising_to_speak(logger, engine, aplay):
    settings = make_settings(max_queue=1)
    player = SpeechPlayer(logger=logger, settings=settings)
    aplay.play_time = 5.0
    app = create_app(logger=logger, settings=settings, engine=engine, player=player)

    with TestClient(app) as client:
        client.post("/api/v1/speak", json={"text": "playing"})
        assert wait_until(lambda: player.snapshot()["state"] == "playing")
        client.post("/api/v1/speak", json={"text": "queued"})

        response = client.post("/api/v1/speak", json={"text": "too much"})

    assert response.status_code == 409
    assert response.json()["code"] == "queue_full"


# --- Health -----------------------------------------------------------------


def test_health_is_ok_when_the_model_is_loaded_and_the_thread_is_alive(client):
    body = client.get("/health").json()

    assert body["status"] == "ok"
    assert body["model"] == "ready"
    assert body["model_error"] is None
    assert body["speaker"]["state"] == "idle"
    assert body["speaker"]["thread_alive"] is True
    assert body["version"]


def test_health_is_degraded_but_still_200_when_the_weights_are_missing(logger, aplay):
    """A non-200 wired into a container healthcheck would restart-loop the
    service exactly when an operator wants to read this body."""
    settings = make_settings()
    engine = KokoroEngine(logger=logger, settings=settings)
    engine.preload()
    player = SpeechPlayer(logger=logger, settings=settings)
    app = create_app(logger=logger, settings=settings, engine=engine, player=player)

    with TestClient(app) as client:
        response = client.get("/health")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "degraded"
    assert body["model"] == STATE_MISSING
    assert "kokoro-v1.0.onnx" in body["model_error"]


def test_health_reports_the_speaker_while_it_is_busy(client, aplay):
    aplay.play_time = 5.0
    job_id = client.post("/api/v1/speak", json={"text": "hello"}).json()["id"]
    assert wait_until(
        lambda: client.get("/health").json()["speaker"]["state"] == "playing"
    )

    speaker = client.get("/health").json()["speaker"]
    assert speaker["job_id"] == job_id
    assert speaker["max_queue"] == 4


# --- Docs -------------------------------------------------------------------


def test_the_openapi_schema_builds(client):
    """Cheap, and it catches a response_model that cannot be generated."""
    response = client.get("/openapi.json")
    assert response.status_code == 200
    paths = response.json()["paths"]
    for route in (
        "/api/v1/voices",
        "/api/v1/synthesize",
        "/api/v1/speak",
        "/api/v1/speak/{job_id}",
        "/health",
    ):
        assert route in paths
