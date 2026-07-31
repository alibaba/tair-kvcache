import os
import logging

logger = logging.getLogger(__name__)
logger.propagate = False
handler = logging.StreamHandler()
handler.setLevel(logging.DEBUG)
formatter = logging.Formatter(
    "[KVCM] %(levelname)s %(asctime)s [%(filename)s:%(lineno)d] %(message)s",
    "%m-%d %H:%M:%S")
handler.setFormatter(formatter)
logger.addHandler(handler)

_DEFAULT_LEVEL = logging.WARNING
_ENV_VAR = "KVCM_LOG_LEVEL"


def set_log_level(level_str: str):
    """Set KVCM logger level from a string like 'DEBUG', 'INFO', 'WARNING'.

    Falls back to WARNING if the level string is invalid or empty.
    """
    if not level_str:
        logger.setLevel(_DEFAULT_LEVEL)
        return
    level = logging.getLevelName(level_str.upper())
    if not isinstance(level, int):
        logger.warning("Invalid log level '%s', falling back to WARNING", level_str)
        level = _DEFAULT_LEVEL
    logger.setLevel(level)


def configure_log_level(param_level: str = ""):
    """Apply log level with priority: env var > param > default(WARNING).

    Args:
        param_level: Log level from startup parameter (kv_connector_extra_config).
    """
    env_level = os.environ.get(_ENV_VAR)
    if env_level:
        set_log_level(env_level)
    elif param_level:
        set_log_level(param_level)
    else:
        logger.setLevel(_DEFAULT_LEVEL)


# Apply environment variable on module load
configure_log_level()
