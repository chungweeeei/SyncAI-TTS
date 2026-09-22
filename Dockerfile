# syncai_tts — speech synthesis and speaker playback, as its own container.
#
# No ROS. That is the point of the image: the service this replaced lived inside
# syncai_backend's rclpy process, which meant its onnxruntime pin had to coexist
# with the ROS ecosystem's numpy, and a ~310 MB model sat in the same address
# space as the nav2 action client and the teleop watchdog. Here the only
# constraints are onnxruntime's own, so the base is plain python:3.10-slim.
#
# 3.10 matches the robot container's interpreter and .python-version, so a wheel
# that resolves on a laptop resolves here.
#
# Dependencies come from uv.lock — the same resolution developers run, byte for
# byte, rather than a requirements.txt that drifts from it.
#
# Build:
#   docker build -t syncai-tts .
#   docker build -t syncai-tts:dev --target dev .
# Run: see docker-compose.yml — the device mounts are the fiddly part.

ARG PYTHON_VERSION=3.10

# ── base ─────────────────────────────────────────────────────────────────────
FROM python:${PYTHON_VERSION}-slim AS base

# uv as a static binary rather than `pip install uv`: it keeps the tool out of
# the environment it manages, and the version is pinned like any other input.
# Pinned: bump this line and uv.lock together.
COPY --from=ghcr.io/astral-sh/uv:0.10.8 /uv /bin/uv

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    # The venv lives outside /app because the dev stage bind-mounts the source
    # over /app, which would otherwise hide it.
    UV_PROJECT_ENVIRONMENT=/opt/venv \
    # Use the image's interpreter instead of downloading a managed CPython: it
    # is already 3.10, and a second copy would only add ~50 MB.
    UV_PYTHON=/usr/local/bin/python3 \
    UV_PYTHON_DOWNLOADS=never \
    # Hardlinking across the cache mount's filesystem boundary is not possible.
    UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1 \
    PATH="/opt/venv/bin:$PATH"

# alsa-utils is `aplay`, which is how audio reaches the speaker — the service
# writes a WAV to its stdin rather than linking an audio library, so this is the
# whole of the audio stack. libgomp1 is onnxruntime's OpenMP runtime; the wheel
# links it and python:slim does not ship it.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        alsa-utils \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Dependencies before source, so an edit to the service does not reinstall
# onnxruntime. --no-install-project is what makes that split work: the lock's
# third-party packages land here, the project itself in the runtime stage.
COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-install-project --no-dev

# ── dev ──────────────────────────────────────────────────────────────────────
# Test and lint image. Source is bind-mounted rather than copied, so an edit
# needs no rebuild:
#   docker run --rm -v "$PWD:/app" syncai-tts:dev pytest test/ -q
FROM base AS dev

RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-install-project

# The suite fakes the kokoro session and the aplay subprocess, so it needs
# neither the model nor a sound card and passes in this image with nothing
# mounted but the source.
CMD ["pytest", "test/", "-q"]

# ── runtime ──────────────────────────────────────────────────────────────────
FROM base AS runtime

COPY syncai_tts/ ./syncai_tts/
COPY README.md ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev

# Non-root, and in `audio` so /dev/snd is reachable. The host's audio gid may
# differ from the container's; docker-compose.yml passes `group_add` for that
# rather than relying on this number matching.
RUN useradd --create-home --shell /bin/bash --groups audio syncai
USER syncai

# The model is a volume, not a layer: ~310 MB of weights in the image would be
# rebuilt and re-pushed on every source change. docker-compose.yml mounts the
# repo's own models/kokoro/ here, read-only; see models/README.md.
ENV TTS_MODEL_PATH=/models/kokoro/kokoro-v1.0.onnx \
    TTS_VOICES_PATH=/models/kokoro/voices-v1.0.bin \
    TTS_HOST=0.0.0.0 \
    TTS_PORT=8080

# Documentation only; docker-compose.yml deliberately publishes nothing to the
# host. Callers are sibling containers on the `syncai` network.
EXPOSE 8080

# Always 200 while the process is alive; the body carries `degraded`. A
# healthcheck that failed on missing weights would restart-loop the container
# exactly when an operator wants to read the reason out of /health.
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8080/health', timeout=4).status == 200 else 1)"

CMD ["syncai-tts"]
