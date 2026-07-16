from __future__ import annotations

import gzip
import importlib
import logging
import sys
import types
from collections.abc import Callable

import pytest

from subscriber import logger


def test_init_applies_debug_log_level() -> None:
    logger.init("test-subscriber", level="debug")

    assert logger.is_debug_enabled() is True
    subscriber_logger = logging.getLogger("subscriber")
    if subscriber_logger.handlers:
        assert subscriber_logger.level == logging.DEBUG


def test_fallback_logger_preserves_exception_traceback(caplog) -> None:
    """exc_info=True renders the traceback into the formatted message."""

    with caplog.at_level(logging.WARNING, logger="subscriber"):
        try:
            raise RuntimeError("boom")
        except RuntimeError:
            logger.warning("operation failed", step="test", exc_info=True)

    message = caplog.records[-1].getMessage()
    assert "operation failed" in message
    assert "RuntimeError" in message
    assert "boom" in message
    assert "Traceback" in message


def test_fallback_logger_compresses_rotated_file(tmp_path) -> None:
    log_file = tmp_path / "subscriber.log"
    handler = logger._GzipRotatingFileHandler(log_file, maxBytes=1, backupCount=3)
    handler.emit(logging.makeLogRecord({"msg": "log record"}))
    handler.doRollover()

    rotated_log = tmp_path / "subscriber.log.1.gz"
    assert rotated_log.exists()
    with gzip.open(rotated_log, "rt") as file:
        assert "log record" in file.read()


# ---------------------------------------------------------------------------
# exc_info rendering — stdlib path
# ---------------------------------------------------------------------------


def test_stdlib_exc_info_none_omits_traceback(caplog) -> None:
    with caplog.at_level(logging.WARNING, logger="subscriber"):
        logger.warning("plain warning", step="s")

    message = caplog.records[-1].getMessage()
    assert "plain warning" in message
    assert "Traceback" not in message


def test_stdlib_exc_info_exception_object_renders_traceback_in_message(caplog) -> None:
    try:
        raise ValueError("specific failure")
    except ValueError as exc:
        captured = exc
    assert captured is not None

    with caplog.at_level(logging.WARNING, logger="subscriber"):
        logger.warning("op failed", step="s", exc_info=captured)

    message = caplog.records[-1].getMessage()
    assert "op failed" in message
    assert "ValueError" in message
    assert "specific failure" in message
    assert "Traceback" in message


def test_stdlib_exc_info_true_renders_current_exception(caplog) -> None:
    with caplog.at_level(logging.WARNING, logger="subscriber"):
        try:
            raise RuntimeError("live boom")
        except RuntimeError:
            logger.warning("during op", step="s", exc_info=True)

    message = caplog.records[-1].getMessage()
    assert "during op" in message
    assert "RuntimeError" in message
    assert "live boom" in message
    assert "Traceback" in message


def test_stdlib_exc_info_false_omits_traceback(caplog) -> None:
    with caplog.at_level(logging.WARNING, logger="subscriber"):
        logger.warning("quiet", exc_info=False)

    assert "Traceback" not in caplog.records[-1].getMessage()


# ---------------------------------------------------------------------------
# exc_info rendering — dashlog path (mocked)
# ---------------------------------------------------------------------------


@pytest.fixture()
def dashlog_mocked(monkeypatch):
    """Force logger.py onto the dashlog code path with a fake dashlog.

    Each level function on the fake is a spy that records (args, kwargs); the
    captured list is exposed as an attribute so tests can assert on it. Spies
    are installed BEFORE the reload so logger.py binds its wrappers to them.
    """

    def _make_spy() -> tuple[Callable[..., None], list[tuple[tuple, dict]]]:
        calls: list[tuple[tuple, dict]] = []

        def _impl(*args: object, **kwargs: object) -> None:
            calls.append((args, kwargs))

        return _impl, calls

    spies = {
        name: _make_spy() for name in ("debug", "info", "warning", "error", "critical")
    }
    fake = types.SimpleNamespace(
        init=lambda *args, **kwargs: None,
        **{name: impl for name, (impl, _calls) in spies.items()},
    )
    for name, (_impl, calls) in spies.items():
        setattr(fake, f"{name}_calls", calls)

    monkeypatch.setitem(sys.modules, "dashlog", fake)
    importlib.reload(logger)
    try:
        yield fake
    finally:
        monkeypatch.delitem(sys.modules, "dashlog", raising=False)
        importlib.reload(logger)


def test_dashlog_exc_info_exception_object_appends_traceback(dashlog_mocked) -> None:
    try:
        raise ValueError("dashlog failure")
    except ValueError as exc:
        captured = exc
    assert captured is not None

    logger.warning("op broke", step="s", exc_info=captured)

    assert len(dashlog_mocked.warning_calls) == 1
    args, kwargs = dashlog_mocked.warning_calls[0]
    message = args[0]
    assert "op broke" in message
    assert "ValueError" in message
    assert "dashlog failure" in message
    assert "Traceback" in message
    assert "exc_info" not in kwargs


def test_dashlog_exc_info_true_appends_current_traceback(dashlog_mocked) -> None:
    try:
        raise RuntimeError("live dashlog boom")
    except RuntimeError:
        logger.warning("during op", step="s", exc_info=True)

    args, kwargs = dashlog_mocked.warning_calls[0]
    message = args[0]
    assert "during op" in message
    assert "RuntimeError" in message
    assert "live dashlog boom" in message
    assert "Traceback" in message
    assert "exc_info" not in kwargs


def test_dashlog_exc_info_none_passes_through(dashlog_mocked) -> None:
    logger.warning("plain", step="s")

    args, kwargs = dashlog_mocked.warning_calls[0]
    assert args[0] == "plain"
    assert "Traceback" not in args[0]
    assert "exc_info" not in kwargs


def test_dashlog_exc_info_false_passes_through(dashlog_mocked) -> None:
    logger.warning("plain", exc_info=False)

    args, kwargs = dashlog_mocked.warning_calls[0]
    assert "Traceback" not in args[0]
    assert "exc_info" not in kwargs


def test_dashlog_does_not_double_render_step_and_tags(dashlog_mocked) -> None:
    """step/tags are native dashlog fields; they must live in kwargs and NOT
    be duplicated into the message string."""

    logger.warning(
        "op happened",
        step="kvcm_register",
        tags={"phase": "initial", "retry": 3},
    )

    args, kwargs = dashlog_mocked.warning_calls[0]
    message = args[0]
    assert "op happened" in message
    # step/tags must NOT leak into the message
    assert "step=kvcm_register" not in message
    assert "phase=initial" not in message
    assert "retry=3" not in message
    # step/tags must stay in kwargs as native dashlog fields
    assert kwargs.get("step") == "kvcm_register"
    assert kwargs.get("tags") == {"phase": "initial", "retry": 3}


def test_render_exc_preserves_exception_chain(dashlog_mocked) -> None:
    """`raise B from A` must surface both A and B in the rendered traceback."""

    try:
        try:
            raise ValueError("root cause")
        except ValueError as inner:
            raise RuntimeError("wrapper") from inner
    except RuntimeError as outer:
        logger.warning("chained failure", exc_info=outer)

    message = dashlog_mocked.warning_calls[0][0][0]
    assert "wrapper" in message
    assert "root cause" in message
    assert "Traceback" in message


def test_render_exc_accepts_tuple_form(dashlog_mocked) -> None:
    """sys.exc_info() returns (type, value, tb); wrapper must render it
    instead of silently swallowing the traceback."""

    try:
        raise KeyError("tuple-form failure")
    except KeyError:
        triple = sys.exc_info()

    logger.warning("tuple path", exc_info=triple)

    message = dashlog_mocked.warning_calls[0][0][0]
    assert "tuple path" in message
    assert "KeyError" in message
    assert "tuple-form failure" in message
    assert "Traceback" in message
