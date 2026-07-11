from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator

from subscriber.config import SubscriberConfig
from subscriber.engine.base import AbstractEngineAdapter
from subscriber.health.coordinator import EngineHealthCoordinator, EngineHealthState
from subscriber.health.events import LivenessEvent
from subscriber.kvcm.client import KvcmClient
from subscriber.types import AllBlocksCleared, KVEventBatch


class FakeAdapter(AbstractEngineAdapter):
    def __init__(self, events: list[LivenessEvent] | None = None) -> None:
        self._events = events or []
        self.reset_generation_calls = 0

    async def subscribe_kv_events(self) -> AsyncGenerator[list[KVEventBatch], None]:
        if False:
            yield []

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


class RecordingKvcmClient(KvcmClient):
    def __init__(self) -> None:
        self.sent: list[tuple[list[KVEventBatch], int]] = []

    async def send_batch(self, batches: list[KVEventBatch], epoch: int) -> None:
        self.sent.append((batches, epoch))


def _config(threshold: int = 3) -> SubscriberConfig:
    return SubscriberConfig(engine_health_failure_threshold=threshold)


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
        tags={"state": "starting", "failure_count": 2},
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
        async def send_batch(self, batches: list[KVEventBatch], epoch: int) -> None:
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
    adapter = FakeAdapter(
        [
            LivenessEvent.HEALTHY,
            LivenessEvent.UNHEALTHY,
            LivenessEvent.UNHEALTHY,
        ]
    )
    kvcm = RecordingKvcmClient()
    coordinator = EngineHealthCoordinator(adapter, kvcm, _config(threshold=2))

    await coordinator.watch_loop()

    assert coordinator.state is EngineHealthState.DEAD
    assert coordinator.epoch == 1
    assert len(kvcm.sent) == 1
    _, epoch = kvcm.sent[0]
    assert epoch == 1
