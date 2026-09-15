from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class AbstractKvCacheManagerClient(ABC):
    """Async contract implemented by KVCM manager transports."""

    @abstractmethod
    async def start(self) -> None:
        """Start the client and any required connection discovery."""

    @abstractmethod
    async def is_ready(self) -> bool:
        """Return whether the client currently has a usable endpoint."""

    @abstractmethod
    async def register_instance(
        self, data: dict[str, Any], check_response: bool = True
    ) -> dict[str, Any]:
        """Register this inference instance with KVCM."""

    @abstractmethod
    async def report_event(
        self, data: dict[str, Any], check_response: bool = True
    ) -> dict[str, Any]:
        """Report KVCM lifecycle or block events."""

    @abstractmethod
    async def close(self) -> None:
        """Release client resources."""
