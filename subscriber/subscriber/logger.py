from __future__ import annotations

import gzip
import logging as _logging
import os
import shutil
import traceback as _tb
from collections.abc import Callable
from logging.handlers import RotatingFileHandler

_LOG_FILE_MAX_BYTES = 100 * 1024 * 1024
_LOG_FILE_BACKUP_COUNT = 3
try:
    import dashlog as _dl
except ImportError:
    _dl = None


def _render_exc(exc_info: object) -> str:
    """Render ``exc_info`` into a traceback string.

    Accepts ``None`` / ``False`` (no output), ``True`` (current exception via
    ``sys.exc_info``), a ``(type, value, tb)`` triple (the standard
    ``sys.exc_info`` shape), or an exception object (uses its
    ``__traceback__`` and follows ``__cause__`` / ``__context__`` chains).
    Returns the empty string when there is nothing to render.
    """

    if exc_info is None or exc_info is False:
        return ""
    if exc_info is True:
        return _tb.format_exc().rstrip()
    if isinstance(exc_info, BaseException):
        return "".join(_tb.format_exception(exc_info)).rstrip()
    if (
        isinstance(exc_info, tuple)
        and len(exc_info) == 3
        and isinstance(exc_info[1], BaseException)
    ):
        return "".join(_tb.format_exception(*exc_info)).rstrip()
    return ""


def _render_body(msg: object, args: tuple[object, ...], exc_info: object) -> str:
    """Percent-format ``msg`` with ``args`` and append a rendered traceback.

    This is the backend-agnostic core shared by the stdlib and dashlog code
    paths. Backend-specific structured fields (``step``, ``tags``, ...) are
    handled by each path's wrapper and deliberately NOT folded in here.
    """

    text = str(msg) % args if args else str(msg)
    tb = _render_exc(exc_info)
    return f"{text}\n{tb}" if tb else text


def _wrap_dashlog(dl_fn: Callable[..., object]) -> Callable[..., None]:
    """Forward to a dashlog level function with ``exc_info`` folded into the
    message string. dashlog itself ignores ``exc_info``; this wrapper makes
    tracebacks visible by embedding them in the message and strips the kwarg
    so it never leaks downstream. ``step`` / ``tags`` / ``request_id`` /
    ``flush`` / ``stacklevel`` pass through in kwargs as native dashlog
    fields and are NOT rendered into the message."""

    def _log(
        msg: object,
        *args: object,
        exc_info: object = None,
        **kwargs: object,
    ) -> None:
        try:
            dl_fn(_render_body(msg, args, exc_info), **kwargs)
        except Exception:
            pass

    return _log


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
        try:
            _dl.init(name, log_level=log_level)
        except Exception:
            pass

    def is_debug_enabled() -> bool:
        return _debug_enabled

    debug = _wrap_dashlog(_dl.debug)
    info = _wrap_dashlog(_dl.info)
    warning = _wrap_dashlog(_dl.warning)
    error = _wrap_dashlog(_dl.error)
    critical = _wrap_dashlog(_dl.critical)

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

    # Runtime log files must never land in the repo root (AGENTS.md): the
    # file handler is opt-in via SUBSCRIBER_LOG_FILE. Unset -> stderr only.
    _log_file = os.environ.get("SUBSCRIBER_LOG_FILE", "")
    if _log_file:
        _fh = _GzipRotatingFileHandler(
            _log_file,
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
        try:
            log_level = getattr(_logging, level.upper(), _logging.INFO)
            _log.setLevel(log_level)
            _log.info("initialized stdlib logging (name=%s)", name)
        except Exception:
            pass

    def _make_log(level: int) -> Callable[..., None]:
        def _log_fn(
            msg: object,
            *args: object,
            step: str = "",
            tags: dict[str, object] | None = None,
            exc_info: object = None,
            **_: object,
        ) -> None:
            try:
                parts = [_render_body(msg, args, exc_info)]
                if step:
                    parts.append(f"step={step}")
                if tags:
                    parts.append(
                        " ".join(f"{key}={value}" for key, value in tags.items())
                    )
                _log.log(level, " | ".join(parts))
            except Exception:
                pass

        return _log_fn

    def is_debug_enabled() -> bool:
        return _log.isEnabledFor(_logging.DEBUG)

    debug = _make_log(_logging.DEBUG)
    info = _make_log(_logging.INFO)
    warning = _make_log(_logging.WARNING)
    error = _make_log(_logging.ERROR)
    critical = _make_log(_logging.CRITICAL)
