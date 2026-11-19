from __future__ import annotations

from enum import Enum


class LivenessEvent(Enum):
    """Engine-independent liveness signal consumed by the health coordinator."""

    HEALTHY = "healthy"
    UNHEALTHY = "unhealthy"
