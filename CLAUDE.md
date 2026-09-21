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
uv run pytest test/ -q                          # full suite (~3 s, 67 tests)
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
never the sentence. `Failure.UNKNOWN_VOICE` (`"unknown_voice"`) is load-bearing
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
- **`onnxruntime==1.18.1` and `numpy<2` are not negotiable.** ≥1.19's CPU-topology
  probe corrupts the heap on the Jetson Orin when `nvpmodel` offlines cores.
  `kokoro-onnx`'s metadata disagrees (it wants `onnxruntime>=1.20.1`, `numpy>=2`);
  `[tool.uv] override-dependencies` overrules it, so its real deps are resolved
  and locked rather than hand-listed beside a `--no-deps` install. Changing
  either pin means re-reading that comment in `pyproject.toml` first.
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
