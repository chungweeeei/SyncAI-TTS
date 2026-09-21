"""Entrypoint: build the settings, the engine and the player, then serve.

Order matters in one place. ``load_dotenv()`` runs before
``Settings.from_env()``, and every value is read inside that call rather than at
module import, so a variable that lives only in ``.env`` is actually seen.
"""

import sys

import dotenv
import structlog
import uvicorn

from syncai_tts.api.app import create_app
from syncai_tts.config import Settings
from syncai_tts.engine import KokoroEngine
from syncai_tts.logger import setup_logging
from syncai_tts.player import SpeechPlayer


def build_app():
    """Construct everything and return the wired app. Also the ASGI factory."""
    dotenv.load_dotenv()
    settings = Settings.from_env()

    setup_logging(json_logs=settings.log_json)
    logger = structlog.get_logger()

    engine = KokoroEngine(logger=logger, settings=settings)
    player = SpeechPlayer(logger=logger, settings=settings)

    return create_app(logger=logger, settings=settings, engine=engine, player=player), settings


def main() -> int:
    try:
        app, settings = build_app()
    except ValueError as exc:
        # A bad setting is a startup failure, not something to discover on the
        # first request that needs the value.
        print(f"syncai_tts: invalid configuration: {exc}", file=sys.stderr)
        return 2

    # uvicorn installs its own SIGINT/SIGTERM handling and runs the lifespan
    # shutdown, which is what stops the playback thread and reaps aplay. No
    # signal handler of our own is needed, and adding one would race it.
    uvicorn.run(
        app,
        host=settings.host,
        port=settings.port,
        # Handlers are plain `def` and run in this pool. One utterance occupies
        # a thread only while it synthesises (or waits, with ?wait=true), never
        # while it plays.
        log_config=None,  # logging is already configured; see logger.py
        access_log=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
