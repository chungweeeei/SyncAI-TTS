# syncai_tts — speech synthesis and speaker playback, as its own container.
#
# No ROS. That is the point of the image: the service this replaced lived inside
# syncai_backend's rclpy process, which meant its onnxruntime pin had to coexist
# with the ROS ecosystem's numpy, and a ~310 MB model sat in the same address
# space as the nav2 action client and the teleop watchdog. Here the only
# constraints are onnxruntime's own, so the base is plain python:3.10-slim.
#
# 3.10 matches the robot container's interpreter, so a wheel that resolves here
# resolves there.
#
# Build:
#   docker build -t syncai-tts .
#   docker build -t syncai-tts:dev --target dev .
# Run: see docker-compose.yml — the device mounts are the fiddly part.

ARG PYTHON_VERSION=3.10

# ── base ─────────────────────────────────────────────────────────────────────
FROM python:${PYTHON_VERSION}-slim AS base

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

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
# onnxruntime.
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# kokoro-onnx last and --no-deps, because its metadata demands
# onnxruntime>=1.20.1 and numpy>=2 and both are wrong for the Orin (the pin's
# reasoning is in requirements.txt). Its real dependencies are already installed
# above, spelled out.
RUN pip install --no-cache-dir --no-deps kokoro-onnx

# ── dev ──────────────────────────────────────────────────────────────────────
# Test and lint image. Source is bind-mounted rather than copied, so an edit
# needs no rebuild:
#   docker run --rm -v "$PWD:/app" syncai-tts:dev pytest test/ -q
FROM base AS dev

COPY requirements-dev.txt ./
RUN pip install --no-cache-dir -r requirements-dev.txt

# The suite fakes the kokoro session and the aplay subprocess, so it needs
# neither the model nor a sound card and passes in this image with nothing
# mounted but the source.
CMD ["pytest", "test/", "-q"]

# ── runtime ──────────────────────────────────────────────────────────────────
FROM base AS runtime

COPY syncai_tts/ ./syncai_tts/
COPY pyproject.toml README.md ./
RUN pip install --no-cache-dir --no-deps -e .

# Non-root, and in `audio` so /dev/snd is reachable. The host's audio gid may
# differ from the container's; docker-compose.yml passes `group_add` for that
# rather than relying on this number matching.
RUN useradd --create-home --shell /bin/bash --groups audio syncai
USER syncai

# The model is a volume, not a layer: ~310 MB of weights in the image would be
# rebuilt and re-pushed on every source change, and the robot already has them
# on disk at ~/robot_ws/models/kokoro.
ENV TTS_MODEL_PATH=/models/kokoro/kokoro-v1.0.onnx \
    TTS_VOICES_PATH=/models/kokoro/voices-v1.0.bin \
    TTS_HOST=0.0.0.0 \
    TTS_PORT=8080

EXPOSE 8080

# Always 200 while the process is alive; the body carries `degraded`. A
# healthcheck that failed on missing weights would restart-loop the container
# exactly when an operator wants to read the reason out of /health.
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8080/health', timeout=4).status == 200 else 1)"

CMD ["syncai-tts"]
