from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator

from subscriber.config import SubscriberConfig
from subscriber.engine.base import AbstractEngineAdapter, EngineEventBatch
from subscriber.engine.metadata import KvEventBootstrap
from subscriber.health.coordinator import EngineHealthCoordinator, EngineHealthState
from subscriber.health.events import LivenessEvent
from subscriber.kvcm.client import KvcmClient
from subscriber.metrics import BatchTelemetry
from subscriber.types import AllBlocksCleared, KVEventBatch


class FakeAdapter(AbstractEngineAdapter):
    def __init__(self, events: list[LivenessEvent] | None = None) -> None:
        self._events = events or []
        self.reset_generation_calls = 0

    async def subscribe_kv_events(self) -> AsyncGenerator[EngineEventBatch, None]:
        if False:
            yield EngineEventBatch(
                [], BatchTelemetry(pipeline="incremental"), trace_id=""
            )

    async def fetch_kv_event_bootstrap(self) -> KvEventBootstrap:
        raise AssertionError("health coordinator must not fetch engine bootstrap")

    async def subscribe_snapshot_events(self) -> AsyncGenerator[EngineEventBatch, None]:
        if False:
            yield EngineEventBatch([], BatchTelemetry(pipeline="snapshot"), trace_id="")

    async def watch_liveness(self) -> AsyncGenerator[LivenessEvent, None]:
        for event in self._events:
            yield event

    async def reset_generation_state(self) -> None:
        self.reset_generation_calls += 1

    def map_medium(self, medium: str | None) -> str:
        return ""

    def supported_mediums(self) -> list[str]:
        return []

    def storage_type(self) -> str:
        return "ST_UNSPECIFIED"

    def request_immediate_snapshot(self) -> None:
        return None


class RecordingKvcmClient(KvcmClient):
    def __init__(self) -> None:
        self.sent: list[tuple[list[KVEventBatch], int]] = []
        self.reregister_after_host_down: list[bool] = []

    async def report_kv_events(
        self,
        batches: list[KVEventBatch],
        epoch: int,
        *,
        reregister_after_host_down: bool = True,
    ) -> None:
        self.sent.append((batches, epoch))
        self.reregister_after_host_down.append(reregister_after_host_down)


def _config(threshold: int = 3, health_interval_s: float = 5.0) -> SubscriberConfig:
    return SubscriberConfig(
        engine_health_failure_threshold=threshold,
        engine_health_interval_s=health_interval_s,
    )


async def test_cold_start_unhealthy_does_not_send_reset() -> None:
    adapter = FakeAdapter()
    kvcm = RecordingKvcmClient()
    coordinator = EngineHealthCoordinator(adapter, kvcm, _config(threshold=2))

    for _ in range(5):
        await coordinator.handle_liveness_event(LivenessEvent.UNHEALTHY)

    assert coordinator.state is EngineHealthState.STARTING
    assert coordinator.epoch == 0
    assert kvcm.sent == []


async def test_cold_start_unhealthy_logs_when_threshold_reached(mocker) -> None:
    adapter = FakeAdapter()
    kvcm = RecordingKvcmClient()
    coordinator = EngineHealthCoordinator(adapter, kvcm, _config(threshold=2))
    warning = mocker.patch("subscriber.health.coordinator.logger.warning")

    await coordinator.handle_liveness_event(LivenessEvent.UNHEALTHY)
    await coordinator.handle_liveness_event(LivenessEvent.UNHEALTHY)

    warning.assert_called_once_with(
        "engine health still unhealthy during startup",
        step="engine_health",
        tags={
            "state": "starting",
            "failure_count": 2,
            "failure_threshold": 2,
        },
    )


async def test_first_healthy_opens_epoch_one_and_releases_waiters() -> None:
    adapter = FakeAdapter()
    kvcm = RecordingKvcmClient()
    coordinator = EngineHealthCoordinator(adapter, kvcm, _config())
    waiter = asyncio.create_task(coordinator.wait_ready_epoch())
    await asyncio.sleep(0)

    assert not waiter.done()

    await coordinator.handle_liveness_event(LivenessEvent.HEALTHY)

    assert await asyncio.wait_for(waiter, timeout=1.0) == 1
    assert coordinator.state is EngineHealthState.HEALTHY
    assert coordinator.epoch == 1
    assert kvcm.sent == []
    assert adapter.reset_generation_calls == 0


async def test_healthy_healthy_resets_failure_and_keeps_gate_open() -> None:
    adapter = FakeAdapter()
    kvcm = RecordingKvcmClient()
    coordinator = EngineHealthCoordinator(adapter, kvcm, _config(threshold=3))

    await coordinator.handle_liveness_event(LivenessEvent.HEALTHY)
    await coordinator.handle_liveness_event(LivenessEvent.UNHEALTHY)
    await coordinator.handle_liveness_event(LivenessEvent.UNHEALTHY)
    await coordinator.handle_liveness_event(LivenessEvent.HEALTHY)

    assert coordinator.state is EngineHealthState.HEALTHY
    assert coordinator.epoch == 1
    assert kvcm.sent == []
    assert adapter.reset_generation_calls == 0
    # Third UNHEALTHY after the reset must not immediately trip DEAD.
    await coordinator.handle_liveness_event(LivenessEvent.UNHEALTHY)
    assert coordinator.state is EngineHealthState.HEALTHY
    assert kvcm.sent == []


async def test_unhealthy_below_threshold_closes_gate_but_stays_healthy() -> None:
    adapter = FakeAdapter()
    kvcm = RecordingKvcmClient()
    coordinator = EngineHealthCoordinator(adapter, kvcm, _config(threshold=3))

    await coordinator.handle_liveness_event(LivenessEvent.HEALTHY)
    await coordinator.handle_liveness_event(LivenessEvent.UNHEALTHY)

    assert coordinator.state is EngineHealthState.HEALTHY
    assert coordinator.epoch == 1
    assert kvcm.sent == []
    waiter = asyncio.create_task(coordinator.wait_ready_epoch())
    await asyncio.sleep(0)
    assert not waiter.done()

    await coordinator.handle_liveness_event(LivenessEvent.HEALTHY)
    assert await asyncio.wait_for(waiter, timeout=1.0) == 1


async def test_healthy_to_dead_sends_reset_once_with_current_epoch() -> None:
    adapter = FakeAdapter()
    kvcm = RecordingKvcmClient()
    coordinator = EngineHealthCoordinator(
        adapter,
        kvcm,
        _config(threshold=2),
        clock=lambda: 123.0,
    )

    await coordinator.handle_liveness_event(LivenessEvent.HEALTHY)
    await coordinator.handle_liveness_event(LivenessEvent.UNHEALTHY)
    await coordinator.handle_liveness_event(LivenessEvent.UNHEALTHY)

    assert coordinator.state is EngineHealthState.DEAD
    assert coordinator.epoch == 1
    assert len(kvcm.sent) == 1
    batches, epoch = kvcm.sent[0]
    assert epoch == 1
    assert batches == [KVEventBatch(ts=123.0, events=[AllBlocksCleared()])]

    # Further UNHEALTHY events must NOT resend reset.
    await coordinator.handle_liveness_event(LivenessEvent.UNHEALTHY)
    await coordinator.handle_liveness_event(LivenessEvent.UNHEALTHY)
    assert len(kvcm.sent) == 1


async def test_reset_report_failure_is_logged_and_recovery_continues(mocker) -> None:
    class FailingResetKvcmClient(RecordingKvcmClient):
        async def report_kv_events(
            self,
            batches: list[KVEventBatch],
            epoch: int,
            *,
            reregister_after_host_down: bool = True,
        ) -> None:
            raise RuntimeError("reset report failed")

    adapter = FakeAdapter()
    kvcm = FailingResetKvcmClient()
    coordinator = EngineHealthCoordinator(adapter, kvcm, _config(threshold=1))
    warning = mocker.patch("subscriber.health.coordinator.logger.warning")

    await coordinator.handle_liveness_event(LivenessEvent.HEALTHY)
    await coordinator.handle_liveness_event(LivenessEvent.UNHEALTHY)
    await coordinator.handle_liveness_event(LivenessEvent.HEALTHY)

    assert coordinator.state is EngineHealthState.HEALTHY
    assert coordinator.epoch == 2
    warning.assert_called_once_with(
        "failed to report all blocks cleared to kvcm",
        step="engine_health",
        tags={
            "epoch": 1,
            "reason": "engine_dead",
            "timeout_s": 2.0,
            "error": "RuntimeError",
            "message": "reset report failed",
        },
        exc_info=True,
    )


async def test_dead_recovery_resets_generation_and_bumps_epoch() -> None:
    adapter = FakeAdapter()
    kvcm = RecordingKvcmClient()
    coordinator = EngineHealthCoordinator(adapter, kvcm, _config(threshold=1))

    await coordinator.handle_liveness_event(LivenessEvent.HEALTHY)
    await coordinator.handle_liveness_event(LivenessEvent.UNHEALTHY)

    assert coordinator.state is EngineHealthState.DEAD
    assert coordinator.epoch == 1
    assert len(kvcm.sent) == 1

    waiter = asyncio.create_task(coordinator.wait_ready_epoch())
    await asyncio.sleep(0)
    assert not waiter.done()

    await coordinator.handle_liveness_event(LivenessEvent.HEALTHY)

    assert coordinator.state is EngineHealthState.HEALTHY
    assert coordinator.epoch == 2
    assert adapter.reset_generation_calls == 1
    assert await asyncio.wait_for(waiter, timeout=1.0) == 2


async def test_reset_generation_runs_before_epoch_bump_and_gate_open() -> None:
    events_order: list[str] = []

    class OrderRecordingAdapter(FakeAdapter):
        def __init__(self, coordinator_ref: list[EngineHealthCoordinator]) -> None:
            super().__init__()
            self._coordinator_ref = coordinator_ref

        async def reset_generation_state(self) -> None:
            await super().reset_generation_state()
            coordinator = self._coordinator_ref[0]
            events_order.append(f"reset(epoch={coordinator.epoch})")

    coordinator_ref: list[EngineHealthCoordinator] = []
    adapter = OrderRecordingAdapter(coordinator_ref)
    kvcm = RecordingKvcmClient()
    coordinator = EngineHealthCoordinator(adapter, kvcm, _config(threshold=1))
    coordinator_ref.append(coordinator)

    await coordinator.handle_liveness_event(LivenessEvent.HEALTHY)
    await coordinator.handle_liveness_event(LivenessEvent.UNHEALTHY)
    await coordinator.handle_liveness_event(LivenessEvent.HEALTHY)

    assert events_order == ["reset(epoch=1)"]
    assert coordinator.epoch == 2


async def test_reset_generation_failure_is_logged_and_retried(mocker) -> None:
    class FlakyResetAdapter(FakeAdapter):
        async def reset_generation_state(self) -> None:
            self.reset_generation_calls += 1
            if self.reset_generation_calls == 1:
                raise RuntimeError("socket recreation failed")

    adapter = FlakyResetAdapter()
    kvcm = RecordingKvcmClient()
    coordinator = EngineHealthCoordinator(adapter, kvcm, _config(threshold=1))
    warning = mocker.patch("subscriber.health.coordinator.logger.warning")

    await coordinator.handle_liveness_event(LivenessEvent.HEALTHY)
    await coordinator.handle_liveness_event(LivenessEvent.UNHEALTHY)
    await coordinator.handle_liveness_event(LivenessEvent.HEALTHY)

    assert coordinator.state is EngineHealthState.DEAD
    assert coordinator.epoch == 1
    warning.assert_called_once_with(
        "failed to reset engine generation state; remaining not ready",
        step="engine_health",
        tags={
            "epoch": 1,
            "error": "RuntimeError",
            "message": "socket recreation failed",
        },
        exc_info=True,
    )

    await coordinator.handle_liveness_event(LivenessEvent.HEALTHY)

    assert coordinator.state is EngineHealthState.HEALTHY
    assert coordinator.epoch == 2
    assert adapter.reset_generation_calls == 2


async def test_watch_loop_consumes_adapter_events() -> None:
    completed = asyncio.Event()

    class BlockingAdapter(FakeAdapter):
        async def watch_liveness(self) -> AsyncGenerator[LivenessEvent, None]:
            for event in self._events:
                yield event
            completed.set()
            await asyncio.Event().wait()

    adapter = BlockingAdapter(
        [LivenessEvent.HEALTHY, LivenessEvent.UNHEALTHY, LivenessEvent.UNHEALTHY]
    )
    kvcm = RecordingKvcmClient()
    coordinator = EngineHealthCoordinator(adapter, kvcm, _config(threshold=2))

    watch_task = asyncio.create_task(coordinator.watch_loop())
    await asyncio.wait_for(completed.wait(), timeout=1.0)

    assert coordinator.state is EngineHealthState.DEAD
    assert coordinator.epoch == 1
    assert len(kvcm.sent) == 1
    _, epoch = kvcm.sent[0]
    assert epoch == 1

    watch_task.cancel()
    try:
        await watch_task
    except asyncio.CancelledError:
        pass


async def test_watch_loop_recovers_after_liveness_exception(mocker) -> None:
    recovered = asyncio.Event()

    class FlakyWatchAdapter(FakeAdapter):
        def __init__(self) -> None:
            super().__init__()
            self.watch_attempts = 0

        async def watch_liveness(self) -> AsyncGenerator[LivenessEvent, None]:
            self.watch_attempts += 1
            if self.watch_attempts == 1:
                yield LivenessEvent.HEALTHY
                raise RuntimeError("liveness monitor failed")
            recovered.set()
            await asyncio.Event().wait()

    adapter = FlakyWatchAdapter()
    kvcm = RecordingKvcmClient()
    coordinator = EngineHealthCoordinator(
        adapter, kvcm, _config(threshold=1, health_interval_s=0)
    )
    warning = mocker.patch("subscriber.health.coordinator.logger.warning")
    watch_task = asyncio.create_task(coordinator.watch_loop())

    await asyncio.wait_for(recovered.wait(), timeout=1.0)
    assert coordinator.state is EngineHealthState.DEAD
    assert len(kvcm.sent) == 1
    warning.assert_any_call(
        "engine liveness watch failed; treating as unhealthy and retrying",
        step="engine_health",
        tags={
            "error": "RuntimeError",
            "message": "liveness monitor failed",
            "retry_in_s": 0,
        },
        exc_info=True,
    )

    watch_task.cancel()
    try:
        await watch_task
    except asyncio.CancelledError:
        pass


async def test_watch_loop_recovers_after_liveness_stream_ends(mocker) -> None:
    recovered = asyncio.Event()

    class EndingWatchAdapter(FakeAdapter):
        def __init__(self) -> None:
            super().__init__()
            self.watch_attempts = 0

        async def watch_liveness(self) -> AsyncGenerator[LivenessEvent, None]:
            self.watch_attempts += 1
            if self.watch_attempts == 1:
                yield LivenessEvent.HEALTHY
                return
            recovered.set()
            await asyncio.Event().wait()

    adapter = EndingWatchAdapter()
    kvcm = RecordingKvcmClient()
    coordinator = EngineHealthCoordinator(
        adapter, kvcm, _config(threshold=1, health_interval_s=0)
    )
    warning = mocker.patch("subscriber.health.coordinator.logger.warning")
    watch_task = asyncio.create_task(coordinator.watch_loop())

    await asyncio.wait_for(recovered.wait(), timeout=1.0)
    assert coordinator.state is EngineHealthState.DEAD
    assert len(kvcm.sent) == 1
    warning.assert_any_call(
        "engine liveness watch ended unexpectedly; treating as unhealthy and retrying",
        step="engine_health",
        tags={"retry_in_s": 0},
    )

    watch_task.cancel()
    try:
        await watch_task
    except asyncio.CancelledError:
        pass


async def test_construct_with_kvcm_none() -> None:
    adapter = FakeAdapter()
    coordinator = EngineHealthCoordinator(adapter, None, _config())

    assert coordinator.state is EngineHealthState.STARTING
    assert coordinator.epoch == 0


async def test_attach_kvcm_client_sets_reference() -> None:
    adapter = FakeAdapter()
    coordinator = EngineHealthCoordinator(adapter, None, _config(threshold=2))
    kvcm = RecordingKvcmClient()

    coordinator.attach_kvcm_client(kvcm)

    await coordinator.handle_liveness_event(LivenessEvent.HEALTHY)
    await coordinator.handle_liveness_event(LivenessEvent.UNHEALTHY)
    await coordinator.handle_liveness_event(LivenessEvent.UNHEALTHY)

    assert coordinator.state is EngineHealthState.DEAD
    assert len(kvcm.sent) == 1


async def test_send_all_blocks_cleared_skipped_when_kvcm_none(mocker) -> None:
    adapter = FakeAdapter()
    coordinator = EngineHealthCoordinator(
        adapter, None, _config(threshold=1), clock=lambda: 42.0
    )
    warning = mocker.patch("subscriber.health.coordinator.logger.warning")

    await coordinator.handle_liveness_event(LivenessEvent.HEALTHY)
    coordinator.attach_kvcm_client(RecordingKvcmClient())
    await coordinator.handle_liveness_event(LivenessEvent.UNHEALTHY)

    assert coordinator.state is EngineHealthState.DEAD
    warning.assert_not_called()


async def test_deferred_kvcm_lifecycle() -> None:
    """Full lifecycle: start without kvcm, become healthy, attach, then go dead."""
    adapter = FakeAdapter()
    coordinator = EngineHealthCoordinator(
        adapter, None, _config(threshold=2), clock=lambda: 99.0
    )

    await coordinator.handle_liveness_event(LivenessEvent.HEALTHY)
    assert coordinator.state is EngineHealthState.HEALTHY
    assert coordinator.epoch == 1

    kvcm = RecordingKvcmClient()
    coordinator.attach_kvcm_client(kvcm)

    await coordinator.handle_liveness_event(LivenessEvent.UNHEALTHY)
    await coordinator.handle_liveness_event(LivenessEvent.UNHEALTHY)

    assert coordinator.state is EngineHealthState.DEAD
    assert len(kvcm.sent) == 1
    batches, epoch = kvcm.sent[0]
    assert epoch == 1
    assert batches == [KVEventBatch(ts=99.0, events=[AllBlocksCleared()])]


class TestReportHostDownIdempotent:
    """report_host_down emits AllBlocksCleared once per sendable epoch."""

    async def test_cold_start_emits_nothing(self) -> None:
        adapter = FakeAdapter()
        kvcm = RecordingKvcmClient()
        coordinator = EngineHealthCoordinator(adapter, kvcm, _config(threshold=2))

        await coordinator.report_host_down("shutdown")

        assert kvcm.sent == []

    async def test_sendable_epoch_emits_once(self) -> None:
        adapter = FakeAdapter()
        kvcm = RecordingKvcmClient()
        coordinator = EngineHealthCoordinator(
            adapter, kvcm, _config(threshold=2), clock=lambda: 7.0
        )
        await coordinator.handle_liveness_event(LivenessEvent.HEALTHY)

        await coordinator.report_host_down("shutdown")

        assert len(kvcm.sent) == 1
        batches, epoch = kvcm.sent[0]
        assert epoch == 1
        assert batches == [KVEventBatch(ts=7.0, events=[AllBlocksCleared()])]
        assert kvcm.reregister_after_host_down == [False]

    async def test_repeated_calls_are_idempotent(self) -> None:
        adapter = FakeAdapter()
        kvcm = RecordingKvcmClient()
        coordinator = EngineHealthCoordinator(adapter, kvcm, _config(threshold=2))
        await coordinator.handle_liveness_event(LivenessEvent.HEALTHY)

        await coordinator.report_host_down("shutdown")
        await coordinator.report_host_down("shutdown")
        await coordinator.report_host_down("engine_dead")

        assert len(kvcm.sent) == 1

    async def test_engine_dead_then_shutdown_emits_once(self) -> None:
        adapter = FakeAdapter()
        kvcm = RecordingKvcmClient()
        coordinator = EngineHealthCoordinator(adapter, kvcm, _config(threshold=1))
        await coordinator.handle_liveness_event(LivenessEvent.HEALTHY)

        # Engine DEAD emits via the watch path.
        await coordinator.handle_liveness_event(LivenessEvent.UNHEALTHY)
        assert coordinator.state is EngineHealthState.DEAD
        assert len(kvcm.sent) == 1

        # Graceful shutdown for the same epoch must not re-emit.
        await coordinator.report_host_down("shutdown")
        assert len(kvcm.sent) == 1

    async def test_recovery_rearms_new_epoch(self) -> None:
        adapter = FakeAdapter()
        kvcm = RecordingKvcmClient()
        coordinator = EngineHealthCoordinator(adapter, kvcm, _config(threshold=1))
        await coordinator.handle_liveness_event(LivenessEvent.HEALTHY)
        await coordinator.handle_liveness_event(LivenessEvent.UNHEALTHY)
        assert len(kvcm.sent) == 1

        # Recover into a new sendable epoch.
        await coordinator.handle_liveness_event(LivenessEvent.HEALTHY)
        assert coordinator.epoch == 2

        await coordinator.report_host_down("shutdown")
        assert len(kvcm.sent) == 2
        _, epoch = kvcm.sent[1]
        assert epoch == 2

    async def test_stale_mark_epoch_sendable_is_ignored(self) -> None:
        adapter = FakeAdapter()
        kvcm = RecordingKvcmClient()
        coordinator = EngineHealthCoordinator(adapter, kvcm, _config(threshold=2))

        # No epoch is current at 0; marking a stale epoch must not arm emission.
        coordinator.mark_epoch_sendable(0)
        await coordinator.report_host_down("shutdown")

        assert kvcm.sent == []

    async def test_slow_send_is_bounded_by_timeout(self) -> None:
        class SlowKvcmClient(RecordingKvcmClient):
            async def report_kv_events(
                self,
                batches: list[KVEventBatch],
                epoch: int,
                *,
                reregister_after_host_down: bool = True,
            ) -> None:
                await asyncio.sleep(10)
                await super().report_kv_events(
                    batches,
                    epoch,
                    reregister_after_host_down=reregister_after_host_down,
                )

        adapter = FakeAdapter()
        kvcm = SlowKvcmClient()
        coordinator = EngineHealthCoordinator(adapter, kvcm, _config(threshold=2))
        await coordinator.handle_liveness_event(LivenessEvent.HEALTHY)

        await asyncio.wait_for(
            coordinator.report_host_down("shutdown", timeout_s=0.01), timeout=1.0
        )

        # The bounded send did not deliver, but emission was still claimed.
        assert kvcm.sent == []

    async def test_concurrent_callers_emit_once(self) -> None:
        class GatedKvcmClient(RecordingKvcmClient):
            def __init__(self) -> None:
                super().__init__()
                self.release = asyncio.Event()

            async def report_kv_events(
                self,
                batches: list[KVEventBatch],
                epoch: int,
                *,
                reregister_after_host_down: bool = True,
            ) -> None:
                await self.release.wait()
                await super().report_kv_events(
                    batches,
                    epoch,
                    reregister_after_host_down=reregister_after_host_down,
                )

        adapter = FakeAdapter()
        kvcm = GatedKvcmClient()
        coordinator = EngineHealthCoordinator(adapter, kvcm, _config(threshold=2))
        await coordinator.handle_liveness_event(LivenessEvent.HEALTHY)

        first = asyncio.create_task(coordinator.report_host_down("engine_dead"))
        second = asyncio.create_task(coordinator.report_host_down("shutdown"))
        await asyncio.sleep(0)
        kvcm.release.set()
        await asyncio.gather(first, second)

        assert len(kvcm.sent) == 1
