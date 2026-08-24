"""Immutable request payload shared by DashTrace observation sinks."""

from __future__ import annotations

from array import array
from dataclasses import dataclass


@dataclass(frozen=True)
class ObservedRequest:
    """One parsed request, ordered at the DashTrace observation boundary."""

    sequence: int
    timestamp_ns: int
    trace_id: str
    instance_id: str
    token_ids: array
