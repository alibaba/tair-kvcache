from __future__ import annotations

import logging

from subscriber import logger


def test_init_applies_debug_log_level() -> None:
    logger.init("test-subscriber", level="debug")

    assert logger.is_debug_enabled() is True
    subscriber_logger = logging.getLogger("subscriber")
    if subscriber_logger.handlers:
        assert subscriber_logger.level == logging.DEBUG


def test_fallback_logger_preserves_exception_traceback(caplog) -> None:
    with caplog.at_level(logging.WARNING, logger="subscriber"):
        try:
            raise RuntimeError("boom")
        except RuntimeError:
            logger.warning("operation failed", step="test", exc_info=True)

    record = caplog.records[-1]
    assert record.exc_info is not None
    assert record.exc_info[0] is RuntimeError
