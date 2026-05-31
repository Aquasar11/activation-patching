"""Unit tests for the opt-in debug logging helpers."""
from __future__ import annotations

import logging

from actpatch import disable_debug_logging, enable_debug_logging, get_logger
from actpatch._logging import _root_logger


def _actpatch_handlers():
    return [h for h in _root_logger.handlers if getattr(h, "_actpatch_handler", False)]


def teardown_function():
    # Keep tests isolated: always return to the quiet default.
    disable_debug_logging()


def test_get_logger_namespacing():
    assert get_logger("actpatch").name == "actpatch"
    assert get_logger("actpatch.hooks").name == "actpatch.hooks"
    # Bare module names are nested under the package root.
    assert get_logger("hooks").name == "actpatch.hooks"


def test_enable_sets_level_and_adds_one_handler():
    enable_debug_logging(level=logging.DEBUG)
    assert _root_logger.level == logging.DEBUG
    assert len(_actpatch_handlers()) == 1


def test_enable_is_idempotent():
    enable_debug_logging()
    enable_debug_logging(level=logging.INFO)
    # No stacked handlers; level updated to the latest call.
    assert len(_actpatch_handlers()) == 1
    assert _root_logger.level == logging.INFO


def test_disable_removes_handler():
    enable_debug_logging()
    assert len(_actpatch_handlers()) == 1
    disable_debug_logging()
    assert len(_actpatch_handlers()) == 0
    assert _root_logger.level == logging.WARNING


def test_debug_records_emitted_when_enabled(caplog):
    enable_debug_logging(level=logging.DEBUG)
    with caplog.at_level(logging.DEBUG, logger="actpatch"):
        get_logger("actpatch.test").debug("hello-%d", 1)
    assert any("hello-1" in r.message for r in caplog.records)
