# models/

The kokoro weights live here. They are **gitignored** — ~310 MB has no business
in a git history — but the directory is part of the repo so that both the local
run and `docker-compose.yml`'s bind mount have one fixed place to look:

```
models/kokoro/kokoro-v1.0.onnx   # ~310 MB
models/kokoro/voices-v1.0.bin    # ~27 MB
```

Fetch them once, from the kokoro-onnx `model-files` release:

```bash
mkdir -p models/kokoro
curl -L -o models/kokoro/kokoro-v1.0.onnx \
  https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.1/kokoro-v1.0.onnx
curl -L -o models/kokoro/voices-v1.0.bin \
  https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.1/voices-v1.0.bin
```

`TTS_MODEL_PATH` and `TTS_VOICES_PATH` default to these paths, resolved against
the working directory the service is started from. In the container they are set
to `/models/kokoro/...`, where `docker-compose.yml` mounts this directory
read-only.

Without them the service still starts: `/health` answers `200` with
`"model": "missing"` and the path it looked at, and every synthesis request
answers `503 model_unavailable`. The test suite does not need them at all.
