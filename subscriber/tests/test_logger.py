from __future__ import annotations

import gzip
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


def test_fallback_logger_compresses_rotated_file(tmp_path) -> None:
    log_file = tmp_path / "subscriber.log"
    handler = logger._GzipRotatingFileHandler(log_file, maxBytes=1, backupCount=3)
    handler.emit(logging.makeLogRecord({"msg": "log record"}))
    handler.doRollover()

    rotated_log = tmp_path / "subscriber.log.1.gz"
    assert rotated_log.exists()
    with gzip.open(rotated_log, "rt") as file:
        assert "log record" in file.read()
