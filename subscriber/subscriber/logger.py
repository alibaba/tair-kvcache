from __future__ import annotations

import gzip
import logging as _logging
import os
import shutil
from collections.abc import Callable
from logging.handlers import RotatingFileHandler

_LOG_FILE_MAX_BYTES = 100 * 1024 * 1024
_LOG_FILE_BACKUP_COUNT = 3
try:
    import dashlog as _dl
except ImportError:
    _dl = None


if _dl is not None:
    _debug_enabled = False

    _LEVEL_TO_INT = {
        "debug": 9,
        "info": 7,
        "warning": 5,
        "error": 4,
        "critical": 2,
    }

    def init(name: str, *, level: str = "info", **_ignored: object) -> None:
        global _debug_enabled
        _debug_enabled = level.lower() == "debug"
        log_level = _LEVEL_TO_INT.get(level.lower(), 7)
        _dl.init(name, log_level=log_level)

    def is_debug_enabled() -> bool:
        return _debug_enabled

    debug = _dl.debug
    info = _dl.info
    warning = _dl.warning
    error = _dl.error
    critical = _dl.critical

else:

    def _gzip_rotator(source: str, destination: str) -> None:
        with open(source, "rb") as source_file, gzip.open(destination, "wb") as output:
            shutil.copyfileobj(source_file, output)
        os.remove(source)

    class _GzipRotatingFileHandler(RotatingFileHandler):
        """Rotate logs into compressed backups to bound disk usage."""

        def __init__(
            self,
            filename: str | os.PathLike[str],
            mode: str = "a",
            maxBytes: int = 0,
            backupCount: int = 0,
            encoding: str | None = None,
            delay: bool = False,
            errors: str | None = None,
        ) -> None:
            super().__init__(
                filename,
                mode=mode,
                maxBytes=maxBytes,
                backupCount=backupCount,
                encoding=encoding,
                delay=delay,
                errors=errors,
            )
            self.namer = lambda default_name: f"{default_name}.gz"
            self.rotator = _gzip_rotator

    _log = _logging.getLogger("subscriber")
    _log.setLevel(_logging.INFO)
    _log.propagate = False

    _fmt = _logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s")

    _fh = _GzipRotatingFileHandler(
        "subscriber.log",
        maxBytes=_LOG_FILE_MAX_BYTES,
        backupCount=_LOG_FILE_BACKUP_COUNT,
        encoding="utf-8",
    )
    _fh.setFormatter(_fmt)
    _log.addHandler(_fh)

    _sh = _logging.StreamHandler()
    _sh.setFormatter(_fmt)
    _log.addHandler(_sh)

    def init(name: str, *, level: str = "info", **_: object) -> None:
        log_level = getattr(_logging, level.upper(), _logging.INFO)
        _log.setLevel(log_level)
        _log.info("initialized stdlib logging (name=%s)", name)

    def _make_log(level: int) -> Callable[..., None]:
        def _log_fn(
            msg: object,
            *args: object,
            step: str = "",
            tags: dict[str, object] | None = None,
            exc_info: bool = False,
            **_: object,
        ) -> None:
            parts = [str(msg) % args if args else str(msg)]
            if step:
                parts.append(f"step={step}")
            if tags:
                parts.append(" ".join(f"{key}={value}" for key, value in tags.items()))
            _log.log(level, " | ".join(parts), exc_info=exc_info)

        return _log_fn

    def is_debug_enabled() -> bool:
        return _log.isEnabledFor(_logging.DEBUG)

    debug = _make_log(_logging.DEBUG)
    info = _make_log(_logging.INFO)
    warning = _make_log(_logging.WARNING)
    error = _make_log(_logging.ERROR)
    critical = _make_log(_logging.CRITICAL)
