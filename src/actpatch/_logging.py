"""Lightweight, opt-in debug tracing for actpatch.

By default the library emits nothing (a `NullHandler` swallows records), so it
never pollutes a host application's logs. Call `enable_debug_logging()` to turn
on verbose tracing of caching, hook registration, patch application, and the
offline cache surgery — useful when a patch isn't taking effect as expected.

    import actpatch
    actpatch.enable_debug_logging()        # DEBUG to stderr
    actpatch.enable_debug_logging(level=logging.INFO)

Every module fetches its logger via `get_logger(__name__)`, all sharing the
`actpatch` parent, so a single call configures the whole package.
"""
from __future__ import annotations

import logging

# Root logger for the whole package. A NullHandler keeps us silent until the
# host app (or `enable_debug_logging`) attaches a real handler.
_ROOT_NAME = "actpatch"
_root_logger = logging.getLogger(_ROOT_NAME)
_root_logger.addHandler(logging.NullHandler())


def get_logger(name: str) -> logging.Logger:
    """Return a child logger under the shared `actpatch` root."""
    # `name` is typically a module's __name__ (e.g. "actpatch.hooks"); if it is
    # already under the root we use it as-is, otherwise we nest it.
    if name == _ROOT_NAME or name.startswith(_ROOT_NAME + "."):
        return logging.getLogger(name)
    return logging.getLogger(f"{_ROOT_NAME}.{name}")


def enable_debug_logging(
    level: int = logging.DEBUG,
    stream=None,
    fmt: str | None = None,
) -> logging.Logger:
    """Attach a StreamHandler to the `actpatch` logger and set its level.

    Idempotent: repeated calls update the level instead of stacking handlers.

    Args:
        level: logging level (default DEBUG).
        stream: target stream (default stderr).
        fmt: optional custom format string.

    Returns:
        The configured `actpatch` root logger.
    """
    if fmt is None:
        fmt = "%(asctime)s %(name)s %(levelname)s %(message)s"

    # Reuse a handler we previously installed rather than adding another.
    handler = next(
        (h for h in _root_logger.handlers if getattr(h, "_actpatch_handler", False)),
        None,
    )
    if handler is None:
        handler = logging.StreamHandler(stream)
        handler._actpatch_handler = True  # type: ignore[attr-defined]
        _root_logger.addHandler(handler)
    elif stream is not None:
        handler.setStream(stream)

    handler.setFormatter(logging.Formatter(fmt))
    handler.setLevel(level)
    _root_logger.setLevel(level)
    _root_logger.debug("actpatch debug logging enabled at level %s", logging.getLevelName(level))
    return _root_logger


def disable_debug_logging() -> None:
    """Remove the handler installed by `enable_debug_logging` and go quiet."""
    for h in list(_root_logger.handlers):
        if getattr(h, "_actpatch_handler", False):
            _root_logger.removeHandler(h)
    _root_logger.setLevel(logging.WARNING)
