# syncai_tts

Speech synthesis and speaker playback for the SyncAI robot, as a standalone
REST service.

One process owns the speaker. Every utterance — an operator pressing *Speak* in
the console, or a scheduled task reaching a `SPEAK` step — is queued here and
played one at a time.

## Running

Fetch the weights into the repo first — they are gitignored, and
`models/README.md` has the two `curl` commands:

```
models/kokoro/kokoro-v1.0.onnx   # ~310 MB
models/kokoro/voices-v1.0.bin    # ~27 MB
```

Then:

```bash
docker compose up --build
```

**The service is not published to the host.** There is no `ports:` in
`docker-compose.yml` on purpose: the callers are other containers, and they
reach it by name on the shared `syncai` network. Publishing 8080 would put an
unauthenticated speaker on the robot's LAN.

A calling stack joins the network and talks to the container name:

```yaml
services:
  backend:
    environment:
      TTS_BASE_URL: http://syncai_tts:8080
    networks: [syncai]

networks:
  syncai:
    external: true        # whichever stack starts first creates it
```

To poke it by hand, go in through the container rather than reopening the port:

```bash
docker compose exec tts python -c \
  "import urllib.request; print(urllib.request.urlopen('http://127.0.0.1:8080/health').read().decode())"
```

### Starting at boot

`restart: unless-stopped` is already in `docker-compose.yml`, and on its own it
is usually enough: the Docker daemon restores containers with a restart policy
when it starts, so a reboot brings the service back. Two things it does not
cover:

- `docker compose down` removes the container, and a removed container has no
  restart policy left to honour.
- a manual `docker stop` stays stopped across reboots — that is what
  `unless-stopped` means.

Check the daemon itself is enabled first, since everything above depends on it:

```bash
systemctl is-enabled docker     # want: enabled
```

For a robot, `deploy/syncai-tts.service` closes the two gaps: it runs
`docker compose up -d` at boot, so the service returns even after a `down`, and
re-reads the compose file each time. Edit `User=` and `WorkingDirectory=` for
the checkout, then:

```bash
sudo cp deploy/syncai-tts.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now syncai-tts
```

Keep `restart: unless-stopped` either way — systemd starts the service at boot,
the restart policy is what recovers it from a crash in between.

### Device wiring

The device wiring is the other fiddly part, and it is all in
`docker-compose.yml`: `/dev/snd` as a device, `/dev/syncai` read-only for the
udev symlink, `./models/kokoro` mounted read-only at `/models/kokoro`, and
`group_add` set to the **host's** audio gid
(`getent group audio | cut -d: -f3`). The weights are a mount, not an image
layer — 310 MB would be rebuilt and re-pushed on every source change.

Locally, without Docker — the project is managed with
[uv](https://docs.astral.sh/uv/):

```bash
uv sync                 # creates .venv from uv.lock; fetches CPython 3.10 if needed
cp .env.example .env
uv run syncai-tts
```

`uv sync` is the whole setup. `.python-version` pins 3.10, the same interpreter
the container runs — which is also what caps `onnxruntime` at 1.23.x, the last
line with cp310 wheels.

Dependencies live in `pyproject.toml` and resolve into `uv.lock`, which is
committed and is what the image installs:

```bash
uv add <pkg>                    # or: uv add --dev <pkg>
uv lock --upgrade-package <pkg> # re-resolve one package
uv sync --no-dev                # what the runtime image installs
```

`onnxruntime` is pinned exactly (`==1.18.1`) and the reason is the Orin, not
taste: ≥1.19 aborts there when `nvpmodel` keeps cores offline. 1.23.2 was
retested on a JetPack 6.2 Orin in September 2026 and still fails. Read the
comment in `pyproject.toml`, and `CLAUDE.md`'s "Verifying the onnxruntime pin on
the Orin", before changing it.

## API

Everything is under `/api/v1`, except `/health`. No authentication: this sits on
the robot's LAN behind the same trust boundary as the backend.

| endpoint | what it does |
|---|---|
| `GET /api/v1/voices` | the loaded model's voice ids |
| `POST /api/v1/synthesize` | text → `audio/wav` (mono, 16-bit, 24 kHz) with an `X-Audio-Duration` header. Nothing is played |
| `POST /api/v1/speak` | render and queue for the speaker; `202` with a job |
| `GET /api/v1/speak?limit=20` | recent jobs, newest first |
| `GET /api/v1/speak/{id}` | one job's state |
| `DELETE /api/v1/speak/{id}` | cancel it, queued or playing |
| `GET /health` | always 200; degradation is in the body |

Request body for both `synthesize` and `speak`: `text` (≤1000 chars, English
only), `voice` (defaults to `TTS_DEFAULT_VOICE`), `speed` (0.5–2.0). `speak`
also takes `wait` and `wait_timeout`.

A job:

```json
{
  "id": "0b284b83c2f74b65b7ef8bd3fd011a9e",
  "status": "queued",
  "text": "Charging complete.",
  "voice": "af_heart",
  "speed": 1.0,
  "duration": 1.85,
  "queue_position": 0,
  "queued_at": "2026-09-21T03:07:25.114Z",
  "started_at": null, "finished_at": null,
  "error": null, "code": null
}
```

`status` walks `queued → playing → done | failed | cancelled`.

Notes worth knowing:

- Synthesis happens inline, so the two failures a caller can act on — unknown
  voice (400) and missing weights (503) — come back synchronously rather than
  being discovered by polling. Only playback becomes a job.
- `"wait": true` holds the response until playback finishes and answers `200`.
  That keeps the backend's existing blocking contract for the console while a
  Temporal activity uses the polling form. A wait that expires is still `202`
  with the current state — "still going, ask again", not an error.
- Cancelling an already-finished job answers `200` with its terminal state, so a
  caller racing its own cancel against the end of an utterance does not have to
  tell the two apart.
- `/health` reports `model` (`unloaded`/`loading`/`ready`/`missing`/`error`) and
  `speaker` (state, queue depth, `thread_alive`). It stays 200 because a
  healthcheck that failed on missing weights would restart-loop the container
  exactly when an operator wants to read the reason.

### Errors

Every error is `{"detail": "<prose>", "code": "<stable string>"}` — the shape
`syncai_backend` already publishes. **Read the code, never the sentence.**

| code | status | meaning |
|---|---|---|
| `unknown_voice` | 400 | the caller's mistake; do not retry |
| `invalid_request` | 422 | body failed validation: text empty or over 1000 chars, speed outside 0.5–2.0 |
| `model_unavailable` | 503 | weights missing, or the session would not build |
| `synthesis_failed` | 500 | kokoro raised on this text |
| `queue_full` | 409 | more than `TTS_MAX_QUEUE` already waiting |
| `job_not_found` | 404 | never existed, or aged out |
| `player_unavailable` | — | on a job: `aplay` is not installed |
| `playback_failed` | — | on a job: aplay exited non-zero |
| `playback_timeout` | — | on a job: the device is wedged |

The last three are recorded on a failed job; by then the request is answered.

`invalid_request` is the one FastAPI would otherwise answer in its own shape (a
list under `detail`, no `code`). A handler in `api/app.py` normalises it, so
`code` is readable on every error this service returns without exception.

`unknown_voice` is load-bearing across both repos — the backend answers 400 on
it and its SPEAK activity marks the attempt non-retryable. The string is shared
with `syncai_backend`'s `Failure.UNKNOWN_VOICE`. Do not rename it.

## Configuration

Environment only, read once at startup — a change needs a restart. `.env.example`
is the annotated list; the ones worth knowing:

| variable | default | notes |
|---|---|---|
| `TTS_PORT` | `8080` | |
| `TTS_MODEL_PATH` | `models/kokoro/kokoro-v1.0.onnx` | repo-relative; `/models/...` in the image |
| `TTS_VOICES_PATH` | `models/kokoro/voices-v1.0.bin` | |
| `TTS_PRELOAD` | `true` | load the session at startup, in a background thread |
| `TTS_MAX_QUEUE` | `8` | utterances allowed to wait behind the one playing |
| `TTS_JOB_HISTORY` | `64` | finished jobs kept readable |
| `TTS_SPEAKER_PCM_LINK` | `/dev/syncai/speaker_pcm` | the udev symlink |
| `TTS_FALLBACK_DEVICE` | `plughw:CARD=CD002AUDIO,DEV=0` | hosts without those rules |
| `TTS_CORS_ORIGINS` | *(empty)* | no CORS middleware at all when empty |

## Tests

CI runs the two commands below on every push to `main`/`dev` and on every pull
request (`.github/workflows/ci.yml`), plus `uv lock --check`. It needs no
hardware — but it cannot speak for the Orin either; a dependency bump that goes
green here still has to pass the on-device check in `CLAUDE.md`.

```bash
uv run pytest test/ -q
uv run pytest test/test_player.py::test_name -q   # one test
uv run ruff check .
```

**No test needs ROS, onnxruntime, a model file or a sound card**: the kokoro
session is a fake object and `aplay` is a fake process class patched over
`subprocess.Popen`. This is worth protecting — the code this replaced could only
run inside the robot container, so the one thing it owned, who may touch the
speaker, had almost no coverage.

The tests that matter most are in `test/test_player.py`: one utterance on the
device at a time, FIFO order, cancellation queued and playing, the narrow
windows where a cancel or a shutdown lands between marking a job playing and
having a process to signal, a wedged device, and a playback thread that survives
an unexpected error instead of silently taking the speaker with it.

Or in the image, with nothing mounted but the source:

```bash
docker build --target dev -t syncai-tts:dev .
docker run --rm -v "$PWD:/app" syncai-tts:dev pytest test/ -q
```

The venv lives at `/opt/venv` in the image, outside `/app`, precisely so that
bind mount does not hide it.

## Layout

```
pyproject.toml     dependencies, uv config, ruff/pytest settings
uv.lock            the committed resolution; the image installs from it
deploy/            systemd unit for starting the stack at boot
models/kokoro/     the weights: gitignored, mounted into the container
syncai_tts/
  main.py          entrypoint: load .env, build settings/engine/player, serve
  config.py        Settings.from_env() — every value read here, not at import
  errors.py        Failure codes + the exceptions the API renders
  engine.py        the kokoro session: lazy load, text -> WAV bytes
  player.py        the speaker: one thread, a FIFO deque, cancellable jobs
  logger.py        structlog, with stdlib/uvicorn bridged into one format
  api/
    app.py         create_app: lifespan, exception handlers, /health
    routes.py      the REST surface
```

Wiring is explicit constructor injection: `create_app` is handed the engine and
the player rather than building them or reaching for singletons. Each module's
docstring carries the reasoning behind its design; `CLAUDE.md` is the short
version.

## Migrating syncai_backend onto this

Not done yet; this repo is the service only. The backend side:

1. Replace `gateways/tts/tts.py` with an HTTP client of the same shape — keep
   the `(success, message, ...)` returns and re-tag the response's `code` with
   the backend's own `Failure`. `routers/tts.py` needs no change.
2. Point `POST /api/v1/tts/speak` at `POST /api/v1/speak` with `wait=true`, so
   the console's contract is unchanged.
3. Rewrite `execute_speak` to enqueue, poll `GET /api/v1/speak/{id}` once a
   second with `activity.heartbeat(status)`, and `DELETE` from
   `except CancelledError`. Then drop the SPEAK special-case in `workflows.py`
   that removes `heartbeat_timeout`.
4. Remove `onnxruntime`, `colorlog`, `espeakng-loader`, `phonemizer-fork` and the
   `--no-deps kokoro-onnx` line from the backend's requirements and Dockerfile.

Step 3 is what the job API was shaped for; steps 1 and 2 can ship on their own.

## Gotchas

- **`aplay` is the whole audio stack.** No `aplay` in the image means every job
  fails with `player_unavailable`.
- The container's `audio` group is meaningless if the host's gid differs and
  `group_add` is not set — the symptom is a permission error on a device that
  looks present.
- A job's audio is dropped the moment it is terminal; only the small record is
  kept, and records are evicted as *new* utterances arrive, never when a job
  ends — so the id you were handed always resolves at least until more work
  comes in.
- The ALSA device is resolved per utterance, because a replug moves the card
  number.
