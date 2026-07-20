from __future__ import annotations

import asyncio
import time
from collections.abc import Callable
from enum import Enum

from subscriber import logger
from subscriber.config import SubscriberConfig
from subscriber.engine.base import AbstractEngineAdapter
from subscriber.health.events import LivenessEvent
from subscriber.kvcm.client import KvcmClient
from subscriber.types import AllBlocksCleared, KVEventBatch


class EngineHealthState(Enum):
    STARTING = "starting"
    HEALTHY = "healthy"
    DEAD = "dead"


class EngineHealthCoordinator:
    """Coordinate engine liveness, send gating, epochs, and reset emission.

    The coordinator is the only component that decides whether KV events may be
    forwarded to kvcm. It opens epochs on healthy generations, closes the gate
    on transient unhealthy events, and emits ``AllBlocksCleared`` only after an
    established generation reaches the configured failure threshold.
    """

    def __init__(
        self,
        adapter: AbstractEngineAdapter,
        kvcm_client: KvcmClient | None,
        config: SubscriberConfig,
        clock: Callable[[], float] | None = None,
    ) -> None:
        self._adapter = adapter
        self._kvcm_client = kvcm_client
        self._threshold = config.engine_health_failure_threshold
        self._defer_initial_availability = config.engine_type == "rtp_llm"
        self._clock = clock or time.time
        self._state = EngineHealthState.STARTING
        self._epoch = 0
        self._failure_count = 0
        self._reset_reported = False
        self._ready = asyncio.Event()

    def attach_kvcm_client(self, kvcm_client: KvcmClient) -> None:
        """Attach the KVCM client after construction.

        Must be called before the coordinator can send reset events
        (i.e. before the first HEALTHY -> DEAD transition).
        """
        self._kvcm_client = kvcm_client

    @property
    def state(self) -> EngineHealthState:
        """Current coarse liveness state for the engine generation."""

        return self._state

    @property
    def epoch(self) -> int:
        """Current send epoch associated with the active engine generation."""

        return self._epoch

    def capture_epoch(self) -> int | None:
        """Snapshot the current epoch if the gate is open, else None."""

        if self._ready.is_set():
            return self._epoch
        return None

    def capture_event_epoch(self) -> int:
        """Assign an engine event to its current or next healthy epoch.

        Events may arrive before the first health probe or while the ready gate
        is temporarily closed. Keeping their generation here lets the bounded
        sender queue preserve them until the gate reopens. Events captured in a
        DEAD generation retain the old epoch and are discarded after recovery.
        """

        if self._state is EngineHealthState.STARTING:
            return self._epoch + 1
        return self._epoch

    def is_epoch_current(self, snapshot: int) -> bool:
        """Return True if the snapshot still matches the active epoch."""

        return snapshot == self._epoch

    async def wait_ready_epoch(self) -> int:
        """Wait until sending is allowed and return the current epoch."""

        await self._ready.wait()
        return self._epoch

    async def watch_loop(self) -> None:
        """Consume adapter liveness events and apply coordinator state changes."""

        async for event in self._adapter.watch_liveness():
            await self.handle_liveness_event(event)

    async def handle_liveness_event(self, event: LivenessEvent) -> None:
        """Apply one liveness event to the health state machine."""

        if event is LivenessEvent.HEALTHY:
            await self._on_healthy()
            return
        await self._on_unhealthy()

    async def _on_healthy(self) -> None:
        if self._state is EngineHealthState.STARTING:
            if self._defer_initial_availability and self._kvcm_client is not None:
                await self._kvcm_client.set_engine_available(True)
            self._epoch += 1
            self._state = EngineHealthState.HEALTHY
        elif self._state is EngineHealthState.DEAD:
            if not self._reset_reported:
                # A previous HOST_DOWN may have failed because KVCM was
                # unavailable. Re-register only to retry the authoritative
                # reset, then pause again if it still cannot be delivered.
                if self._kvcm_client is not None:
                    await self._kvcm_client.set_engine_available(True)
                self._reset_reported = await self._send_all_blocks_cleared()
                if not self._reset_reported:
                    if self._kvcm_client is not None:
                        await self._kvcm_client.set_engine_available(False)
                    return
            try:
                await self._adapter.reset_generation_state()
            except Exception as exc:
                logger.warning(
                    "failed to reset engine generation state; remaining not ready",
                    step="engine_health",
                    tags={
                        "epoch": self._epoch,
                        "error": exc.__class__.__name__,
                        "message": str(exc),
                    },
                    exc_info=True,
                )
                return
            if self._kvcm_client is not None:
                await self._kvcm_client.set_engine_available(True)
            self._epoch += 1
            self._state = EngineHealthState.HEALTHY
            self._reset_reported = False
        self._failure_count = 0
        self._ready.set()

    async def _on_unhealthy(self) -> None:
        if self._state is EngineHealthState.DEAD:
            return
        self._failure_count += 1
        if self._state is EngineHealthState.STARTING:
            if self._failure_count % self._threshold == 0:
                logger.warning(
                    "engine health still unhealthy during startup",
                    step="engine_health",
                    tags={
                        "state": self._state.value,
                        "failure_count": self._failure_count,
                        "failure_threshold": self._threshold,
                    },
                )
            return
        self._ready.clear()
        if self._failure_count >= self._threshold:
            self._state = EngineHealthState.DEAD
            if self._kvcm_client is not None:
                await self._kvcm_client.set_engine_available(False)
            self._reset_reported = await self._send_all_blocks_cleared()

    async def _send_all_blocks_cleared(self) -> bool:
        if self._kvcm_client is None:
            logger.warning(
                "cannot send all blocks cleared; kvcm client not attached",
                step="engine_health",
                tags={"epoch": self._epoch},
            )
            return False
        batch = KVEventBatch(ts=self._clock(), events=[AllBlocksCleared()])
        try:
            await self._kvcm_client.send_batch([batch], self._epoch)
        except Exception as exc:
            logger.warning(
                "failed to report all blocks cleared to kvcm",
                step="engine_health",
                tags={
                    "epoch": self._epoch,
                    "error": exc.__class__.__name__,
                    "message": str(exc),
                },
                exc_info=True,
            )
            return False
        return True
