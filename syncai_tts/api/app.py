"""FastAPI application: exception handling, lifespan, /health.

Wiring is explicit constructor injection, the convention ``syncai_backend``
follows: :func:`create_app` is handed the engine and the player rather than
building them or reaching for module-level singletons, so a test gets a fresh
pair per test and nothing leaks between them.
"""

from contextlib import asynccontextmanager

import structlog
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from syncai_tts import __version__
from syncai_tts.api.routes import init_routes
from syncai_tts.config import Settings
from syncai_tts.engine import STATE_READY, KokoroEngine
from syncai_tts.errors import TtsError
from syncai_tts.player import SpeechPlayer


def register_exception_handlers(app: FastAPI) -> None:
    """Render domain errors as ``{"detail": ..., "code": ...}``.

    The shape matches what syncai_backend already publishes for its own
    ``ConflictError``, so the backend's TtsGateway can lift ``code`` straight off
    the body and re-tag it with its own ``Failure`` enum. Neither side ever has
    to match on the prose — see syncai_tts/errors.py.
    """

    @app.exception_handler(TtsError)
    async def _tts_error(_: Request, exc: TtsError) -> JSONResponse:
        content = {"detail": exc.detail}
        if exc.code is not None:
            content["code"] = exc.code.value
        return JSONResponse(status_code=exc.status_code, content=content)


def create_app(
    logger: structlog.stdlib.BoundLogger,
    settings: Settings,
    engine: KokoroEngine,
    player: SpeechPlayer,
) -> FastAPI:

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        # Startup. The playback thread must exist before the first request can
        # queue anything, and preloading happens in the background so uvicorn
        # starts answering /health immediately — which is the endpoint that
        # explains a model that is still loading or missing.
        player.start()
        if settings.preload:
            engine.preload_in_background()
        logger.info(
            "syncai_tts ready",
            version=__version__,
            host=settings.host,
            port=settings.port,
            preload=settings.preload,
        )

        yield

        # Shutdown. uvicorn runs this on SIGTERM, so a container stop cuts the
        # utterance and reaps aplay instead of orphaning it holding the pcm
        # node open.
        player.stop()

    app = FastAPI(
        title="SyncAI TTS Service",
        description=(
            "Speech synthesis and speaker playback for the SyncAI robot. One "
            "process owns the speaker: every utterance, whether it came from an "
            "operator or from a scheduled task, is queued here and played one at "
            "a time."
        ),
        version=__version__,
        lifespan=lifespan,
    )

    # No CORS by default. The callers are syncai_backend and the Temporal
    # worker, both server-side, and neither is subject to CORS; installing a
    # permissive policy for them would only widen what a browser on the robot
    # LAN can reach. Set TTS_CORS_ORIGINS to a real list if that ever changes.
    if settings.cors_origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=settings.cors_origins,
            allow_credentials=False,
            allow_methods=["GET", "POST", "DELETE", "OPTIONS"],
            allow_headers=["Content-Type"],
        )

    register_exception_handlers(app)

    @app.get("/health", tags=["Health"])
    async def health() -> dict:
        """Liveness, plus the two things that actually stop speech working.

        Always HTTP 200, the same choice syncai_backend's /health makes: a
        non-200 wired into a container healthcheck would restart-loop the
        service whenever the weights are missing, which is exactly when an
        operator wants to read this body. Degradation lives in the body.

        `model` is `unloaded`/`loading`/`ready`/`missing`/`error`; `speaker`
        carries the queue depth and whether the playback thread is alive, which
        is the failure this service could otherwise have silently.
        """
        speaker = player.snapshot()
        model_state = engine.state
        healthy = model_state == STATE_READY and speaker["thread_alive"]
        return {
            "status": "ok" if healthy else "degraded",
            "version": __version__,
            "model": model_state,
            "model_error": engine.error,
            "model_path": settings.model_path,
            "speaker": speaker,
        }

    app.include_router(
        init_routes(logger=logger, settings=settings, engine=engine, player=player)
    )

    return app
