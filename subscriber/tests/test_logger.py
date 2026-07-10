from __future__ import annotations

import logging

from subscriber import logger


def test_init_applies_debug_log_level() -> None:
    logger.init("test-subscriber", level="debug")

    assert logger.is_debug_enabled() is True
    subscriber_logger = logging.getLogger("subscriber")
    if subscriber_logger.handlers:
        assert subscriber_logger.level == logging.DEBUG
