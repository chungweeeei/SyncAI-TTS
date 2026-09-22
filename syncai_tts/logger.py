import logging
import sys

import structlog


def setup_logging(json_logs: bool = False) -> None:
    """Configure structlog, and route stdlib logging through the same renderer.

    Unlike syncai_backend's logger.py, this also bridges the standard library:
    uvicorn and its access log are stdlib loggers, and left alone they print in
    their own format alongside structlog's, giving one process two log shapes.
    ``ProcessorFormatter`` puts both through the renderer chosen here, so a log
    shipper sees one format and a human reads one column layout.
    """
    timestamper = structlog.processors.TimeStamper(fmt="%Y-%m-%d %H:%M:%S", utc=False)

    shared_processors = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        timestamper,
    ]

    renderer = (
        structlog.processors.JSONRenderer()
        if json_logs
        else structlog.dev.ConsoleRenderer()
    )

    structlog.configure(
        processors=[
            *shared_processors,
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        wrapper_class=structlog.make_filtering_bound_logger(logging.NOTSET),
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )

    formatter = structlog.stdlib.ProcessorFormatter(
        foreign_pre_chain=shared_processors,
        processors=[
            structlog.stdlib.ProcessorFormatter.remove_processors_meta,
            renderer,
        ],
    )

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)

    root = logging.getLogger()
    # Replace rather than append: re-running setup (tests, a reload) would
    # otherwise print every line once per previous call.
    root.handlers = [handler]
    root.setLevel(logging.INFO)

    # uvicorn installs its own handlers at import; clearing them and letting the
    # records propagate to root is what actually puts them through the formatter
    # above. Without this, access lines keep uvicorn's default format.
    for name in ("uvicorn", "uvicorn.error", "uvicorn.access"):
        stdlib_logger = logging.getLogger(name)
        stdlib_logger.handlers = []
        stdlib_logger.propagate = True
