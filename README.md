# syncai_tts

Speech synthesis and speaker playback for the SyncAI robot, as a standalone
REST service.

One process owns the speaker. Every utterance — an operator pressing *Speak* in
the console, or a scheduled task reaching a `SPEAK` step — is queued here and
played one at a time.

## Why this is its own service

It used to be `TtsGateway`, a class inside `syncai_backend`'s rclpy process.
Two things pushed it out.

**The speaker needs exactly one owner.** "One `aplay` stream on the pcm node at
a time" was enforced by a `threading.Lock`, which works only while every caller
is in the same process. The plan to move the Temporal worker into a process of
its own breaks that: two `TtsGateway` instances hold two locks and nothing stops
a scheduled step and a manual request from opening two streams on one device.
Moving the speaker behind a service puts the lock back in front of the single
piece of hardware, whatever the callers do.

**The dependency pins had nothing to do with ROS.** `onnxruntime==1.18.1` is
pinned because anything newer corrupts the heap on the Orin when `nvpmodel`
offlines cores. Inside the backend that pin also had to coexist with the ROS
ecosystem's numpy ceiling, and `kokoro-onnx` had to be installed `--no-deps` to
stop pip resolving past it. Here the pin answers to the hardware alone, the
image is `python:3.10-slim` with no ROS in it, and the backend process sheds
onnxruntime, phonemizer and a ~310 MB model from its address space.

A third thing came free. Because playback is now a job rather than a blocking
call, a `SPEAK` activity can poll it once a second — which is a heartbeat — and
cancel it. In the backend today that activity holds one blocking `speak()`,
cannot heartbeat at all, and so runs on `start_to_close` alone and is
uncancellable mid-utterance.

## The API

Everything is under `/api/v1`, except `/health`. There is no authentication:
this sits on the robot's LAN behind the same trust boundary as the backend.

### `GET /api/v1/voices`

```json
{ "voices": ["af_heart", "am_adam", "..."] }
```

503 `model_unavailable` if the weights are not on disk.

### `POST /api/v1/synthesize`

Render text and hand back the audio. Nothing reaches the speaker.

```json
{ "text": "Charging complete.", "voice": "af_heart", "speed": 1.0 }
```

Answers `audio/wav` (mono, 16-bit, 24 kHz) with an `X-Audio-Duration` header in
seconds, so a caller can size a timeout without decoding the file.

### `POST /api/v1/speak`

Render text and queue it for the speaker. Same body, plus `wait` and
`wait_timeout`.

Synthesis happens inline, so the two failures a caller can act on — an unknown
voice (400) and missing weights (503) — come back synchronously rather than
being discovered by polling. Only playback, the part that takes as long as the
utterance, becomes a job.

`202 Accepted`:

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
  "started_at": null,
  "finished_at": null,
  "error": null,
  "code": null
}
```

With `"wait": true` the response is held until playback finishes and comes back
`200` with `status: "done"`. That exists so the backend's existing blocking
`POST /api/v1/tts/speak` contract can be kept for the console while a Temporal
activity uses the polling form. If the wait expires the answer is still `202`
with whatever state the job is in — a timeout there means "still going, ask
again", not an error.

### `GET /api/v1/speak/{id}`

The same body. `status` walks `queued → playing → done | failed | cancelled`.
On a failure, `error` is a sentence for an operator and `code` is the stable
discriminator to branch on. 404 `job_not_found` once it ages out of the history
window.

### `DELETE /api/v1/speak/{id}`

Stop it, queued or playing. A queued job leaves the queue; a playing one gets
`SIGTERM` to `aplay`, then `SIGKILL` if it does not let go of the device.

Cancelling an already-finished job answers 200 with its terminal state rather
than an error: a caller racing its own cancel against the end of an utterance
should not have to tell those two outcomes apart.

### `GET /api/v1/speak?limit=20`

Recent jobs, newest first. Queued and playing ones are in there too.

### `GET /health`

Always 200 while the process is alive; degradation lives in the body. A
healthcheck that failed on missing weights would restart-loop the container
exactly when an operator wants to read the reason.

```json
{
  "status": "degraded",
  "version": "0.1.0",
  "model": "missing",
  "model_error": "kokoro model file missing: /models/kokoro/kokoro-v1.0.onnx — ...",
  "model_path": "/models/kokoro/kokoro-v1.0.onnx",
  "speaker": {
    "state": "idle", "job_id": null,
    "queue_depth": 0, "max_queue": 8, "thread_alive": true
  }
}
```

`model` is `unloaded` / `loading` / `ready` / `missing` / `error`.
`speaker.thread_alive` is there because a dead playback thread is the one
failure this service could otherwise have silently, answering 202 to everything
and speaking none of it.

### Errors

Every error is `{"detail": "<prose>", "code": "<stable string>"}` — the shape
`syncai_backend` already publishes for its own `ConflictError`, so the backend
can lift `code` straight off the body. **Read the code, never the sentence.**

| code | status | meaning |
|---|---|---|
| `unknown_voice` | 400 | the caller's mistake; do not retry |
| `model_unavailable` | 503 | weights missing, or the session would not build |
| `synthesis_failed` | 500 | kokoro raised on this text |
| `queue_full` | 409 | more than `TTS_MAX_QUEUE` already waiting |
| `job_not_found` | 404 | never existed, or aged out |
| `player_unavailable` | — | on a job: `aplay` is not installed |
| `playback_failed` | — | on a job: aplay exited non-zero |
| `playback_timeout` | — | on a job: the device is wedged |

The last three are recorded on a failed job rather than returned from a request,
because by then the request has already been answered.

`unknown_voice` is load-bearing across both repos: it is the one TTS failure
that is the caller's to fix, so the backend's REST layer answers 400 instead of
its uniform 502 **and** its SPEAK activity marks the attempt non-retryable. The
string is shared with `syncai_backend`'s `Failure.UNKNOWN_VOICE`. Do not rename
it.

## Design notes

**The queue is FIFO and bounded.** A lock wakes waiters in whatever order the OS
chooses, so utterances could come out in any order; a single playback thread
draining a deque plays them in the order they were accepted, and refuses work
past `TTS_MAX_QUEUE` rather than answering 202 to something it will speak
minutes later.

**Two locks, not one.** The kokoro session and the speaker are separate
resources: synthesis takes a few hundred milliseconds, playback takes as long as
the utterance. Sharing one lock is the bug `syncai_backend` fixed in 79fa196 —
a caller that only wants bytes must not be made to wait out someone else's
utterance, and synthesis finishing must not release the device.

**Handlers are plain `def`, not `async def`.** Synthesis is CPU-bound and
`?wait=true` blocks; declared `def`, FastAPI runs them in its threadpool. Same
rule as the backend, same reason.

**The ALSA device is resolved per utterance.** `/dev/syncai/speaker_pcm` is a
udev symlink to `pcmC<card>D<dev>p` that `syncai_sys_manager`'s
`99-syncai-devices.rules` maintains. A replug moves the card number, so a name
resolved once at startup would go stale. A missing or malformed link falls back
to the by-name dongle rather than raising — `aplay` is the error reporter with
an actionable message, and a broken link should not become a traceback.

**Shutdown is wired.** uvicorn runs the app's lifespan shutdown on `SIGTERM`,
which cuts the current utterance, reaps `aplay` and drops the queue. Nothing is
left holding the pcm node open.

**Settings are read inside `Settings.from_env()`, never at module import.** That
is deliberate: `syncai_backend/temporal/shared.py` reads `TEMPORAL_ADDRESS` at
import time and `main.py` imports it before `load_dotenv()`, so a value living
only in `.env` is silently ignored. Reading after the load makes that class of
bug impossible here.

## Configuration

Environment only, read once at startup. See `.env.example` for the annotated
list. The ones worth knowing:

| variable | default | notes |
|---|---|---|
| `TTS_PORT` | `8080` | |
| `TTS_MODEL_PATH` | `~/robot_ws/models/kokoro/kokoro-v1.0.onnx` | `/models/...` in the image |
| `TTS_VOICES_PATH` | `~/robot_ws/models/kokoro/voices-v1.0.bin` | |
| `TTS_PRELOAD` | `true` | load the session at startup in a background thread |
| `TTS_MAX_QUEUE` | `8` | utterances allowed to wait behind the one playing |
| `TTS_SPEAKER_PCM_LINK` | `/dev/syncai/speaker_pcm` | the udev symlink |
| `TTS_FALLBACK_DEVICE` | `plughw:CARD=CD002AUDIO,DEV=0` | hosts without those rules |
| `TTS_CORS_ORIGINS` | *(empty)* | no CORS middleware at all when empty |

`TTS_PRELOAD` defaults on, unlike the lazy load it replaces. The in-process
gateway loaded lazily so a boot that never spoke never paid ~3 s and 310 MB;
this container exists only to speak, and a cold first utterance would otherwise
burn that against a Temporal `SPEAK` step's budget.

CORS is off by default. The callers are `syncai_backend` and the Temporal
worker, both server-side and not subject to CORS; a permissive policy for them
would only widen what a browser on the robot LAN can reach.

## Running

```bash
docker compose up --build
```

The device wiring is the fiddly part and it is all in `docker-compose.yml`:
`/dev/snd` as a device, `/dev/syncai` read-only for the udev symlink, the
weights bind-mounted at `/models/kokoro`, and `group_add` set to the **host's**
audio gid (`getent group audio | cut -d: -f3`) — the container user is in
`audio`, but the host's gid is what governs `/dev/snd`.

The weights are a volume rather than an image layer: 310 MB baked in would be
rebuilt and re-pushed on every source change, and the robot already has them.

```bash
# https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.1/kokoro-v1.0.onnx
# https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.1/voices-v1.0.bin
```

Locally, without Docker:

```bash
python3 -m venv .venv && . .venv/bin/activate
pip install -r requirements-dev.txt
cp .env.example .env
syncai-tts          # or: python -m syncai_tts.main
```

Note that `requirements.txt` pins `onnxruntime==1.18.1`, which has no wheels for
recent Pythons. On a dev machine where it will not install, install the web
stack alone (`fastapi uvicorn pydantic structlog python-dotenv pytest httpx
numpy`) — the test suite needs nothing more, and the service still starts and
serves `/health` and the error paths.

## Tests

```bash
pytest test/ -q
ruff check .
```

64 tests, and **none of them need ROS, onnxruntime, a model file or a sound
card**: the kokoro session is a fake object assigned onto the engine and `aplay`
is a fake process class patched over `subprocess.Popen`. This is worth
protecting. The code this replaced could only be exercised inside the robot
container, because its module graph reached `rclpy` and the generated ROS
interfaces, and the result was that the one thing it owned — who may touch the
speaker — had almost no coverage.

Or in the image, with nothing mounted but the source:

```bash
docker build --target dev -t syncai-tts:dev .
docker run --rm -v "$PWD:/app" syncai-tts:dev pytest test/ -q
```

The tests that matter most are in `test/test_player.py`: one utterance on the
device at a time, FIFO order, cancellation both queued and playing, the two
narrow windows where a cancel or a shutdown lands between marking a job playing
and having a process to signal, a wedged device, and a playback thread that
survives an unexpected error instead of silently taking the speaker with it.

## Layout

```
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

Wiring is explicit constructor injection — `create_app` is handed the engine and
the player rather than building them or reaching for module-level singletons —
which is the convention `syncai_backend` follows and the reason its routers are
testable.

## Migrating syncai_backend onto this

Not done yet; this repo is the service only. The backend side is small:

1. Replace `gateways/tts/tts.py` with an HTTP client of the same shape. Keep
   `(success, message, ...)` returns and re-tag the response's `code` with the
   backend's own `Failure`, and `routers/tts.py` needs no change at all.
2. Point `POST /api/v1/tts/speak` at `POST /api/v1/speak` with `wait=true`, so
   the console's contract is unchanged.
3. Rewrite `execute_speak` to enqueue, then poll `GET /api/v1/speak/{id}` once a
   second with `activity.heartbeat(status)`, and call `DELETE` from
   `except CancelledError`. Then drop the SPEAK special-case in
   `workflows.py` that removes `heartbeat_timeout`.
4. Remove `onnxruntime`, `colorlog`, `espeakng-loader`, `phonemizer-fork` and
   the `--no-deps kokoro-onnx` line from the backend's requirements and
   Dockerfile.

Step 3 is what the job API was shaped for; steps 1 and 2 can ship on their own
first.

## Gotchas

- **`aplay` is the whole audio stack.** The service writes a WAV to its stdin.
  No `aplay` in the image means every job fails with `player_unavailable`.
- **Configuration is read once, at startup.** Changing `.env` needs a restart.
- The `audio` group inside the container is meaningless if the host's gid
  differs and `group_add` is not set — the symptom is `aplay` failing with a
  permission error on a device that looks present.
- A job's audio is dropped the moment it reaches a terminal state; only the
  small record is kept. Records are evicted as *new* utterances arrive, oldest
  first, past `TTS_JOB_HISTORY` — never when a job ends, so the id you were
  handed always resolves at least until more work comes in.
- The text cap (1000 chars), the speed range (0.5–2.0) and English-only G2P are
  the same constraints the backend's route already advertises.
