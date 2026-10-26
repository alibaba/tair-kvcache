"""Caller node identity for affinity-aware KVCM placement.

The manager only runs its affinity strategy when a control-plane request
carries ``CallerNode`` (``meta_service.proto``: ``StartWriteCacheRequest.caller``
and ``GetCacheLocationRequest.caller``). The compiled C++ client attaches that
field deep inside ``grpc_stub.cc``, but every Python connector talks to the
manager over plain HTTP, so nothing on this side ever fills it in and the
manager silently degrades to affinity-neutral placement.

The identity is the mempool node id of the pace storage provider co-located
with this process. Only the pod-local sidecar knows it, and it is exposed
through the ``kvcm_py_client.local_node_id`` pybind export. Resolution is
best-effort and cached: an empty string means "unknown" (no compiled client, no
sidecar, or an ambiguous provider set), and the request then stays
byte-identical to what an old client sends.
"""

import threading
import time
from typing import Any, Mapping, Optional

from kv_cache_manager.py_connector.common.logger import logger

# Endpoints whose request message declares a ``caller`` field. Attaching it
# anywhere else would only be dropped by the manager's JSON decoder.
CALLER_ENDPOINTS = frozenset(
    {
        "/api/startWriteCache",
        "/api/getCacheLocation",
    }
)

# A resolved identity is re-checked periodically so a sidecar restart, which
# re-registers the local provider under a new mempool node id, cannot leave a
# stale caller behind for the whole lifetime of the process.
_SUCCESS_REFRESH_SECONDS = 300.0
# A failed lookup backs off instead, so a deployment without a pace sidecar does
# not probe the local socket on every request of a hot read path.
_FAILURE_RETRY_SECONDS = 30.0

_state_lock = threading.Lock()
_node_id = ""
_last_attempt = 0.0


def _query_local_node_id() -> str:
    """Ask the compiled client for the co-located provider's node id."""
    try:
        # kvcm_py_client is the compiled pybind11 client. It ships no type stubs
        # and is absent from wheels that only carry the pure-Python HTTP client.
        from kv_cache_manager.client.pybind import (  # ty: ignore[unresolved-import]
            kvcm_py_client,
        )
    except ImportError:
        return ""

    resolve = getattr(kvcm_py_client, "local_node_id", None)
    if resolve is None:
        # Wheel built before the affinity export.
        return ""
    try:
        return str(resolve() or "")
    except Exception:
        # Reporting the caller must never break the cache path.
        logger.debug("kvcm local_node_id lookup failed", exc_info=True)
        return ""


def _refresh_interval_seconds(node_id: str) -> float:
    """Return how long a resolved (or failed) identity may be reused."""
    return _SUCCESS_REFRESH_SECONDS if node_id else _FAILURE_RETRY_SECONDS


def local_caller_node_id() -> str:
    """Return the caller node id, resolving it at most once per interval."""
    global _node_id, _last_attempt

    if _last_attempt:
        elapsed = time.monotonic() - _last_attempt
        if elapsed < _refresh_interval_seconds(_node_id):
            return _node_id

    with _state_lock:
        now = time.monotonic()
        if _last_attempt:
            elapsed = now - _last_attempt
            if elapsed < _refresh_interval_seconds(_node_id):
                return _node_id

        resolved = _query_local_node_id()
        _last_attempt = now
        if not resolved:
            # Keep the previous identity when a refresh comes back empty: a
            # transiently unavailable sidecar must not downgrade requests that
            # are already affinity-aware.
            return _node_id
        if resolved != _node_id:
            logger.info(
                "reporting KVCM caller node_id=%s for affinity placement", resolved
            )
            _node_id = resolved
    return _node_id


def with_caller_node(
    endpoint: str, data: Optional[Mapping[str, Any]]
) -> Optional[Mapping[str, Any]]:
    """Return ``data`` with the caller node attached for affinity endpoints.

    The input mapping is never mutated, because call sites reuse their payload
    across retries, and an explicit ``caller`` already in the payload wins.
    """
    if endpoint not in CALLER_ENDPOINTS or data is None or "caller" in data:
        return data
    node_id = local_caller_node_id()
    if not node_id:
        return data
    payload: dict[str, Any] = dict(data)
    payload["caller"] = {"node_id": node_id}
    return payload
