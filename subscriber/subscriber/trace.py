"""EagleEye-style trace ID generator.

Produces a 30-character identifier with the layout::

    {ip_hex8}{timestamp_ms13}{counter4}d{pid_hex4}

Matches EagleEye's time-ordered, machine-identifiable correlation format. Its
counter intentionally cycles from 1000 through 9001; IP resolution failures
fall back to ``ffffffff``.
"""

from __future__ import annotations

import os
import socket
import threading
import time

_FALLBACK_IP_HEX = "ffffffff"


def _resolve_ip_hex() -> str:
    try:
        addr = socket.gethostbyname(socket.gethostname())
        return "".join(f"{int(octet):02x}" for octet in addr.split("."))
    except Exception:
        return _FALLBACK_IP_HEX


_IP_HEX: str = _resolve_ip_hex()

_pid = os.getpid()
_PID_HEX: str = f"{_pid % 60000 if _pid > 65535 else _pid:04x}"

_counter_lock = threading.Lock()
_counter = 1000


def generate_trace_id() -> str:
    """Generate an EagleEye-compatible trace ID (30 chars, never raises)."""
    global _counter
    ts = int(time.time() * 1000)
    with _counter_lock:
        _counter = 1000 if _counter > 9000 else _counter + 1
        seq = _counter
    return f"{_IP_HEX}{ts}{seq}d{_PID_HEX}"
