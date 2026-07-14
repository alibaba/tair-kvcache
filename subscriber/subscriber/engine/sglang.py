from __future__ import annotations

from collections.abc import AsyncGenerator

from subscriber.config import SubscriberConfig
from subscriber.engine.base import AbstractEngineAdapter, EngineEventBatch
from subscriber.health.events import LivenessEvent
from subscriber.metrics import StageTimer


@AbstractEngineAdapter.register("sglang")
class SGLangAdapter(AbstractEngineAdapter):
    def __init__(self, config: SubscriberConfig) -> None:
        self._config = config

    async def subscribe_kv_events(self) -> AsyncGenerator[EngineEventBatch, None]:
        pass
        if False:  # pragma: no cover - placeholder to satisfy async generator protocol
            yield EngineEventBatch([], StageTimer())

    async def watch_liveness(self) -> AsyncGenerator[LivenessEvent, None]:
        pass
        if False:  # pragma: no cover - placeholder to satisfy async generator protocol
            yield

    def map_medium(self, medium: str | None) -> str:
        pass
        return ""

    def supported_mediums(self) -> list[str]:
        pass
        return []

    def storage_type(self) -> str:
        pass
        return ""

    async def reset_generation_state(self) -> None:
        pass
