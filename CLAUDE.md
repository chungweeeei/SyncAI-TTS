# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

The project is managed with **uv**. `pyproject.toml` declares the dependencies,
`uv.lock` is committed and is what the image installs, and `.python-version`
pins CPython 3.10 — the container's interpreter, and the reason
`onnxruntime==1.18.1` (no wheels for 3.13+) installs on a laptop at all. There
is no `requirements.txt`; do not reintroduce one.

```bash
uv sync                                         # create/update .venv from the lock
uv run pytest test/ -q                          # full suite (~3 s, 77 tests)
uv run pytest test/test_player.py::test_name -q # one test
uv run ruff check .                             # lint
uv run syncai-tts                               # run the service
uv add <pkg>            # or: uv add --dev <pkg>; then commit the lock change
uv lock --upgrade-package <pkg>
```

In Docker (the only way to actually reach the speaker):

```bash
docker compose up --build
docker build --target dev -t syncai-tts:dev . && docker run --rm -v "$PWD:/app" syncai-tts:dev pytest test/ -q
docker compose exec tts python -c \
  "import urllib.request; print(urllib.request.urlopen('http://127.0.0.1:8080/health').read().decode())"
```

The weights live in the repo at `models/kokoro/` (gitignored; `models/README.md`
has the download commands) and are bind-mounted read-only at `/models/kokoro` —
never baked into the image. In the image the venv is at `/opt/venv`, outside
`/app`, so the dev stage's source bind mount does not hide it.

**The service is not published to the host.** `docker-compose.yml` has no
`ports:`, only `expose`. Callers are sibling containers on the external-by-name
`syncai` network and reach it at `http://syncai_tts:8080`; there is no
authentication, so publishing the port would put the speaker on the robot's LAN.
Do not add `ports:` to debug — use the `exec` line above.

Commits follow Conventional Commits — see `.github/prompt/copilot-commit-message-instructions.md`.

## What this service is

Speech synthesis and speaker playback for the SyncAI robot, split out of
`syncai_backend`'s in-process `TtsGateway`. The reason for the split is the
invariant: **exactly one process owns the speaker.** A `threading.Lock` only
enforced "one `aplay` stream at a time" while every caller lived in one process,
which stops being true once the Temporal worker moves out. Changes that
reintroduce a second path to the device defeat the point of the repo.

## Architecture

Request path: `main.build_app()` loads `.env` → `Settings.from_env()` →
constructs `KokoroEngine` and `SpeechPlayer` → `create_app()` is *handed* both.
Wiring is explicit constructor injection everywhere; there are no module-level
singletons, and tests depend on that.

**Two resources, two locks, two objects.** `engine.py` owns the kokoro session
(synthesis, a few hundred ms); `player.py` owns the speaker (playback, as long
as the utterance). They must never share a lock — a caller that only wants WAV
bytes must not wait out someone else's utterance.

**`player.py` is the heart of the repo.** One worker thread drains a bounded
FIFO deque; every field of `_Job` is written under the player's lock, and the
API only ever sees immutable `JobView` snapshots. `POST /speak` synthesises
inline (so `unknown_voice` 400 and `model_unavailable` 503 are answered
synchronously) and makes only *playback* a job: `queued → playing → done |
failed | cancelled`, pollable and cancellable. Two windows are real and are
tested: a cancel or a shutdown landing between marking a job `playing` and
having a `Popen` to signal (`job.cancel_requested` covers it), and the worker
surviving an unexpected exception instead of silently taking the speaker down.
History is evicted when new work is *accepted*, never when a job ends, so a
poller never races its own job out of existence.

**The error contract is shared with `syncai_backend`.** Every error body is
`{"detail": "<prose>", "code": "<stable string>"}` and callers branch on `code`,
never the sentence. That holds for pydantic's rejections too, but only because
`api/app.py` installs a `RequestValidationError` handler — FastAPI's default
body has no `code` and a list under `detail`. Any new handler that can answer
an error must go through `TtsError` or that handler; nothing else may invent a
body shape. `Failure.UNKNOWN_VOICE` (`"unknown_voice"`) is load-bearing
across both repos — the backend's REST layer answers 400 on it and its SPEAK
activity marks the attempt non-retryable. Do not rename these strings.

**`aplay` is the entire audio stack.** The service writes a WAV to its stdin. The
device name is resolved *per utterance* from the udev symlink
`/dev/syncai/speaker_pcm`, because a replug moves the card number; an unresolvable
link falls back to the by-name dongle rather than raising.

## Conventions that are load-bearing

- **Every setting is read inside `Settings.from_env()`, never at module import.**
  `main.py` calls `load_dotenv()` first; reading at import time is the bug in
  `syncai_backend/temporal/shared.py` this layout exists to avoid.
- **Route handlers are plain `def`, not `async def`.** Synthesis is CPU-bound and
  `?wait=true` blocks; `def` puts them in FastAPI's threadpool instead of on the
  event loop.
- **`/health` always returns 200**; degradation lives in the body (`model`,
  `speaker.thread_alive`). A failing healthcheck would restart-loop the container
  exactly when an operator wants to read the reason.
- **`onnxruntime==1.18.1` is a hardware decision, and it has been retested.** On
  the Orin, ≥1.19's ARM CPU-detection path indexes past the end of its core list
  whenever `nvpmodel` keeps cores offline. 1.23.2 was tried on a JetPack 6.2 Orin
  on 2026-09-21 and **still aborts** — see the comment in `pyproject.toml` for
  the exact assertion and the upstream issue. It is an `abort()`, not an
  exception: the process dies and the container restart-loops. Do not attempt
  the upgrade again until upstream closes it or every core is online.
- **No ROS, no database, no Temporal client in this process.** Keeping the
  dependency set small is half the reason the split happened.
- `ruff.toml` pins `select = ["E4", "E7", "E9", "F"]` explicitly and deliberately
  omits isort — imports are grouped by layer and "organize imports" collapses them.

## Tests

`test/conftest.py` fakes the kokoro session (assigned onto `engine._kokoro`) and
patches a `FakeAplay` over `subprocess.Popen`, so **no test needs ROS,
onnxruntime, a model file or a sound card.** Keep it that way: the code this
replaced could only run inside the robot container, and so the one thing it
owned — who may touch the speaker — was barely covered. `FakeAplay.max_inside`
is how concurrency on the device is asserted; `FakeAplay.gate` holds the player
inside the spawn window a cancel can land in.

## Verifying the onnxruntime pin on the Orin

The crash that freezes the pin at 1.18.1 only reproduces on the device, with
cores offline — no amount of laptop or CI testing settles it, and 1.23.2 passed
every off-device check (macOS arm64, a linux/arm64 container, real synthesis)
before failing on the robot within seconds. On a JetPack 6.2 Orin:

```bash
nvpmodel -q                       # confirm the mode that offlines cores (MODE_30W: 8-11)
nproc                             # fewer than the full core count = the failing condition
docker compose up -d --build
for i in $(seq 20); do            # the crash is a malloc assertion at session construction,
  docker compose restart tts      # and it is intermittent — one clean start proves nothing
  sleep 15
  docker compose exec -T tts python -c \
    "import urllib.request,json; print(json.load(urllib.request.urlopen('http://127.0.0.1:8080/health'))['model'])"
done
```

`model: ready` on every iteration, with no assertion and no exit 134 in
`docker compose logs tts`, is the evidence. Anything else: revert the pin to
`1.18.1`, `numpy<2` and the `[tool.uv] override-dependencies` block that goes
with it.

The failure looks like this — it is what 1.23.2 produced on 2026-09-21:

```
onnxruntime cpuid_info warning: Unknown CPU vendor. cpuinfo_vendor value: 0
/opt/rh/gcc-toolset-14/root/usr/include/c++/14/bits/stl_vector.h:1130:
  Assertion '__n < this->size()' failed.
```

The `gcc-toolset` version in that path identifies the wheel: 1.18.1 is built
with gcc-toolset-12, 1.23.x with 14. It is a quick way to tell which pin a
container is actually running.
