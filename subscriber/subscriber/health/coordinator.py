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
from subscriber.metrics import report_engine_state_transition, report_shutdown
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
        self._liveness_retry_interval_s = config.engine_health_interval_s
        self._host_down_timeout_s = config.subscriber_shutdown_report_timeout_s
        self._clock = clock or time.time
        self._state = EngineHealthState.STARTING
        self._epoch = 0
        self._failure_count = 0
        self._ready = asyncio.Event()
        # Guards the HostDown emit flag so concurrent callers (engine DEAD on the
        # watch loop and graceful shutdown) cannot both pass the check-and-set.
        self._host_down_lock = asyncio.Lock()
        # True once AllBlocksCleared has been emitted for the current sendable
        # epoch; reset by mark_epoch_sendable() when a new epoch opens.
        self._host_down_sent = False
        # True once any epoch became sendable. Unlike the gate (_ready), which
        # closes on unhealthy, this never clears: it is the cold-start guard for
        # HostDown emission. A generation that never opened has nothing to clear.
        self._epoch_sendable = False

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

    def is_epoch_current(self, snapshot: int) -> bool:
        """Return True if the snapshot still matches the active epoch."""

        return snapshot == self._epoch

    def mark_epoch_sendable(self, epoch: int) -> None:
        """Mark ``epoch`` as sendable, arming HostDown emission for it.

        Opening a sendable epoch means KV batches for it may reach kvcm, so a
        later engine DEAD or graceful shutdown must emit ``AllBlocksCleared``
        for it exactly once. Stale epochs (older than the current one) are
        ignored so a delayed call cannot re-arm a superseded generation.
        """

        if epoch <= 0 or epoch != self._epoch:
            return
        self._epoch_sendable = True
        self._host_down_sent = False

    async def wait_ready_epoch(self) -> int:
        """Wait until sending is allowed and return the current epoch."""

        await self._ready.wait()
        return self._epoch

    async def watch_loop(self) -> None:
        """Consume liveness streams, retrying after unexpected termination."""

        while True:
            try:
                async for event in self._adapter.watch_liveness():
                    await self.handle_liveness_event(event)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.warning(
                    "engine liveness watch failed; treating as unhealthy and retrying",
                    step="engine_health",
                    tags={
                        "error": exc.__class__.__name__,
                        "message": str(exc),
                        "retry_in_s": self._liveness_retry_interval_s,
                    },
                    exc_info=True,
                )
            else:
                logger.warning(
                    "engine liveness watch ended unexpectedly; "
                    "treating as unhealthy and retrying",
                    step="engine_health",
                    tags={"retry_in_s": self._liveness_retry_interval_s},
                )
            await self.handle_liveness_event(LivenessEvent.UNHEALTHY)
            await asyncio.sleep(self._liveness_retry_interval_s)

    async def handle_liveness_event(self, event: LivenessEvent) -> None:
        """Apply one liveness event to the health state machine."""

        if event is LivenessEvent.HEALTHY:
            await self._on_healthy()
            return
        await self._on_unhealthy()

    async def _on_healthy(self) -> None:
        if self._state is EngineHealthState.STARTING:
            self._epoch += 1
            report_engine_state_transition("starting", "healthy")
            self._state = EngineHealthState.HEALTHY
            logger.info(
                "engine healthy; sendable epoch opened",
                step="engine_health",
                tags={"from": "starting", "to": "healthy", "epoch": self._epoch},
            )
        elif self._state is EngineHealthState.DEAD:
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
            self._epoch += 1
            report_engine_state_transition("dead", "healthy")
            self._state = EngineHealthState.HEALTHY
            logger.info(
                "engine recovered; new sendable epoch opened",
                step="engine_health",
                tags={"from": "dead", "to": "healthy", "epoch": self._epoch},
            )
        else:
            # Already HEALTHY: the sendable epoch is unchanged and HostDown
            # arming must not be reset (aflap must not re-arm a cleared epoch).
            self._failure_count = 0
            self._ready.set()
            return
        self.mark_epoch_sendable(self._epoch)
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
            report_engine_state_transition("healthy", "dead")
            self._state = EngineHealthState.DEAD
            logger.error(
                "engine declared dead after consecutive failures; "
                "forwarding gate closed",
                step="engine_health",
                tags={
                    "from": "healthy",
                    "to": "dead",
                    "epoch": self._epoch,
                    "failure_count": self._failure_count,
                    "failure_threshold": self._threshold,
                },
            )
            await self.report_host_down("engine_dead")

    async def report_host_down(
        self, reason: str, timeout_s: float | None = None
    ) -> None:
        """Emit ``AllBlocksCleared`` (EVENT_HOST_DOWN) once per sendable epoch.

        Engine DEAD (on the watch loop) and graceful shutdown both call this;
        the emit flag makes it idempotent so at most one ``AllBlocksCleared``
        reaches kvcm for a given sendable epoch. No event is emitted when no
        epoch ever became sendable (cold startup failure) — there is nothing to
        clear. The send is bounded by ``timeout_s`` (default: the configured
        shutdown report timeout) and never raises; a failure or timeout is
        logged and falls back to KVCM heartbeat expiry.
        """

        if not self._epoch_sendable:
            # Cold start: no epoch ever became sendable, so no KV batch was
            # forwardable and there is no generation to clear.
            return
        async with self._host_down_lock:
            if self._host_down_sent:
                return
            # Claim emission before the await so a concurrent caller cannot
            # pass the guard while this send is in flight.
            self._host_down_sent = True
            epoch_snapshot = self._epoch
        if self._kvcm_client is None:
            logger.warning(
                "cannot send all blocks cleared; kvcm client not attached",
                step="engine_health",
                tags={"epoch": epoch_snapshot, "reason": reason},
            )
            return
        effective_timeout_s = (
            self._host_down_timeout_s if timeout_s is None else timeout_s
        )
        batch = KVEventBatch(ts=self._clock(), events=[AllBlocksCleared()])
        try:
            await asyncio.wait_for(
                self._kvcm_client.report_kv_events(
                    [batch],
                    epoch_snapshot,
                    reregister_after_host_down=False,
                ),
                timeout=effective_timeout_s,
            )
        except Exception as exc:
            report_shutdown(outcome=f"host_down_failed:{reason}")
            logger.warning(
                "failed to report all blocks cleared to kvcm",
                step="engine_health",
                tags={
                    "epoch": epoch_snapshot,
                    "reason": reason,
                    "timeout_s": effective_timeout_s,
                    "error": exc.__class__.__name__,
                    "message": str(exc),
                },
                exc_info=True,
            )
            return
        report_shutdown(outcome=f"host_down_sent:{reason}")
        logger.info(
            "all blocks cleared reported to kvcm",
            step="engine_health",
            tags={"epoch": epoch_snapshot, "reason": reason},
        )
