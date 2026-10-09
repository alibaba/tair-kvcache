from __future__ import annotations

import os
import time
from collections.abc import Mapping

try:
    import dashlog as _dashlog
except ImportError:
    _dashlog = None

_METRIC_PREFIX = "kvcache_subscriber_"

_RESET_INTERVAL_S = 10.0
_counter_accumulators: dict[tuple[str, tuple[tuple[str, str], ...]], float] = {}
_counter_last_reset_s: float = 0.0

# Mirrors dashlog EASOTelClient::Enabled() truthiness so the subscriber's
# counter routing never disagrees with the native backend.
_OTEL_DISABLED_VALUES = frozenset(("false", "False", "FALSE", "0"))


def _otel_enabled() -> bool:
    val = os.environ.get("DS_EAS_USE_OTEL")
    return bool(val) and val not in _OTEL_DISABLED_VALUES


def _dashlog_init(app_name: str) -> None:
    if _dashlog is not None:
        _dashlog.Init(app_name)


def _dashlog_gauge(
    name: str, value: float, tags: Mapping[str, str] | None = None
) -> None:
    if _dashlog is not None:
        _dashlog.Gauge(
            _METRIC_PREFIX + name, value, tags=dict(tags or {}), add_app_prefix=False
        )


def _dashlog_counter(
    name: str, value: int, tags: Mapping[str, str] | None = None
) -> None:
    if _otel_enabled():
        # ASI-EAS (DS_EAS_USE_OTEL): dashlog routes Counter to the native OTel
        # instrument with delta semantics, so no local accumulation is needed.
        # Truthiness mirrors EASOTelClient::Enabled() (dashlog
        # csrc/metrics/eas_otel_client.cc:1136); Counter accepts non-negative
        # deltas only, and all callers pass counts.
        if _dashlog is None:
            return
        try:
            _dashlog.Counter(
                _METRIC_PREFIX + name,
                value,
                tags=dict(tags or {}),
                add_app_prefix=False,
            )
        except Exception:
            pass
        return
    # PAI-EAS: EASClient::AddCounters drops data when OTel is disabled, so
    # route counters through Gauge to reach /realtime_metrics.
    # Limitation: the reset (and idle-key zeroing) is driven lazily by the
    # next counter that fires, so if every counter goes idle the last window
    # values remain until any counter fires again.
    global _counter_last_reset_s
    now = time.monotonic()
    key = (name, tuple(sorted((tags or {}).items())))
    if now - _counter_last_reset_s >= _RESET_INTERVAL_S:
        stale_keys = [k for k in _counter_accumulators if k != key]
        _counter_accumulators.clear()
        _counter_last_reset_s = now
        for stale_name, stale_tags in stale_keys:
            try:
                _dashlog_gauge(stale_name, 0.0, tags=dict(stale_tags) or None)
            except Exception:
                pass
    accumulated = _counter_accumulators.get(key, 0.0) + value
    _counter_accumulators[key] = accumulated
    try:
        _dashlog_gauge(name, accumulated, tags=tags)
    except Exception:
        pass


def init_dashlog(app_name: str) -> None:
    from subscriber import logger

    if _dashlog is None:
        logger.warning(
            "dashlog is unavailable; metrics reporting is disabled",
            step="metrics_init",
            tags={"app_name": app_name},
        )
        return
    try:
        _dashlog_init(app_name)
    except Exception as exc:
        logger.warning(
            "dashlog init failed; metrics reporting is disabled",
            step="metrics_init",
            tags={
                "app_name": app_name,
                "error": exc.__class__.__name__,
                "message": str(exc),
            },
        )
