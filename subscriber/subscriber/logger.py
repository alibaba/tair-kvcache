from __future__ import annotations

# Thin shim: use real dashlog when installed (prod), fall back to stdlib
# logging when not available (local dev / CI without the internal wheel).
try:
    import logging as _logging

    import dashlog as _dl  # type: ignore[import-not-found]

    _debug_enabled = False

    def init(name: str, *, level: str = "info", **kwargs: object) -> None:
        global _debug_enabled
        _debug_enabled = level.lower() == "debug"
        _dl.init(name, level=level, **kwargs)

    def is_debug_enabled() -> bool:
        return _debug_enabled

    debug = _dl.debug
    info = _dl.info
    warning = _dl.warning
    error = _dl.error
    critical = _dl.critical

except ImportError:
    import logging as _logging
    from logging.handlers import RotatingFileHandler

    _log = _logging.getLogger("subscriber")
    _log.setLevel(_logging.INFO)

    _fmt = _logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s")

    _fh = RotatingFileHandler(
        "subscriber.log", maxBytes=100 * 1024 * 1024, backupCount=3
    )
    _fh.setFormatter(_fmt)
    _log.addHandler(_fh)

    _sh = _logging.StreamHandler()
    _sh.setFormatter(_fmt)
    _log.addHandler(_sh)

    def init(name: str, *, level: str = "info", **_: object) -> None:
        log_level = getattr(_logging, level.upper(), _logging.INFO)
        _log.setLevel(log_level)
        _log.info("dashlog not installed, using stdlib logging (name=%s)", name)

    def _make_log(level: int):  # type: ignore[no-untyped-def]
        def _log_fn(
            msg: object,
            *args: object,
            step: str = "",
            tags: dict | None = None,  # type: ignore[type-arg]
            **_: object,
        ) -> None:
            parts = [str(msg) % args if args else str(msg)]
            if step:
                parts.append(f"step={step}")
            if tags:
                parts.append(" ".join(f"{k}={v}" for k, v in tags.items()))
            _log.log(level, " | ".join(parts))

        return _log_fn

    def is_debug_enabled() -> bool:
        return _log.isEnabledFor(_logging.DEBUG)

    debug = _make_log(_logging.DEBUG)
    info = _make_log(_logging.INFO)
    warning = _make_log(_logging.WARNING)
    error = _make_log(_logging.ERROR)
    critical = _make_log(_logging.CRITICAL)
