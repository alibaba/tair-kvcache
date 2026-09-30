from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncGenerator

import httpx
import pytest
from pytest_mock import MockerFixture

import subscriber.__main__
from subscriber.cli import cli as real_cli
from subscriber.config import SubscriberConfig
from subscriber.engine.base import AbstractEngineAdapter, EngineEventBatch
from subscriber.engine.metadata import (
    EventTransport,
    KvCacheDescriptor,
    KvEventBootstrap,
    MetadataProtocolError,
    MetadataTemporarilyUnavailable,
    RuntimeTopology,
    SnapshotCapability,
    VllmEventSchema,
)
from subscriber.health.coordinator import EngineHealthState
from subscriber.health.events import LivenessEvent
from subscriber.health.state_reporter import DashservingStateReporter
from subscriber.kvcm.client import KvcmClient
from subscriber.kvcm.errors import KvcmReportRejectedError, KvcmUnavailableError
from subscriber.main import SubscriberLifecycle, _FatalStartupError
from subscriber.metrics import BatchTelemetry
from subscriber.trace import generate_trace_id
from subscriber.types import AllBlocksCleared, BlockSnapshot, BlockStored, KVEventBatch


def test_main_module_exposes_cli() -> None:
    assert hasattr(subscriber.__main__, "cli")


def test_main_module_cli_is_the_real_cli() -> None:
    assert subscriber.__main__.cli is real_cli


def test_main_module_cli_is_callable() -> None:
    assert callable(subscriber.__main__.cli)


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


def _kv_batch() -> list[KVEventBatch]:
    return [
        KVEventBatch(
            ts=1.0,
            events=[
                BlockStored(
                    block_hashes=[1],
                    parent_block_hash=None,
                    token_ids=[1],
                    block_size=16,
                    lora_id=None,
                    medium="GPU",
                    lora_name=None,
                )
            ],
        )
    ]


def _snapshot_batch() -> list[KVEventBatch]:
    return [
        KVEventBatch(
            ts=1.0,
            events=[
                BlockSnapshot(
                    medium="GPU",
                    block_size=16,
                    items=[],
                    snapshot_version=1,
                )
            ],
        )
    ]


def _bootstrap() -> KvEventBootstrap:
    return KvEventBootstrap(
        protocol_version=1,
        engine_kind="vllm",
        event_transport=EventTransport(
            live_endpoint="tcp://127.0.0.1:5557",
            topic="",
            replay_supported=True,
            replay_endpoint="tcp://127.0.0.1:5558",
            serialization="msgpack-v1",
        ),
        runtime_topology=RuntimeTopology(1, 1, 1, 0, 0, 0),
        snapshot=SnapshotCapability(supported=True, versioned=True),
        components=(),
        compatibility_settings=(),
        diagnostic_settings=(),
        vllm=VllmEventSchema(2, False, "none", "sha256", "v1"),
    )


class FakeAdapter(AbstractEngineAdapter):
    """Adapter whose liveness is driven through a queue.

    ``watch_liveness`` yields an initial HEALTHY then forwards whatever the test
    pushes onto ``liveness``. ``subscribe_kv_events`` forwards batches pushed
    onto ``kv_queue``. Metadata result/error are configurable.
    """

    def __init__(
        self,
        calls: list[str],
        *,
        snapshot_ends_immediately: bool = False,
        immediate_snapshot: list[KVEventBatch] | None = None,
    ) -> None:
        self._calls = calls
        self._snapshot_ends_immediately = snapshot_ends_immediately
        self._immediate_snapshot = immediate_snapshot
        self.liveness: asyncio.Queue[LivenessEvent] = asyncio.Queue()
        self.kv_queue: asyncio.Queue[list[KVEventBatch]] = asyncio.Queue()
        self.snapshot_queue: asyncio.Queue[list[KVEventBatch]] = asyncio.Queue()
        self.bootstrap_result = _bootstrap()
        self.metadata_errors: list[Exception] = []
        self.fetch_count = 0
        self.snapshot_yield_count = 0
        self.snapshot_consumed_count = 0
        self.reset_count = 0
        self.closed = False

    async def subscribe_kv_events(self) -> AsyncGenerator[EngineEventBatch, None]:
        while True:
            batches = await self.kv_queue.get()
            yield EngineEventBatch(
                batches,
                BatchTelemetry(pipeline="incremental"),
                trace_id=generate_trace_id(),
            )

    async def subscribe_snapshot_events(self) -> AsyncGenerator[EngineEventBatch, None]:
        if self._snapshot_ends_immediately:
            return
        if self._immediate_snapshot is not None:
            self.snapshot_yield_count += 1
            yield EngineEventBatch(
                self._immediate_snapshot,
                BatchTelemetry(pipeline="snapshot"),
                trace_id=generate_trace_id(),
            )
            self.snapshot_consumed_count += 1
        while True:
            batches = await self.snapshot_queue.get()
            self.snapshot_yield_count += 1
            yield EngineEventBatch(
                batches,
                BatchTelemetry(pipeline="snapshot"),
                trace_id=generate_trace_id(),
            )
            self.snapshot_consumed_count += 1

    async def watch_liveness(self) -> AsyncGenerator[LivenessEvent, None]:
        self._calls.append("watch_started")
        yield LivenessEvent.HEALTHY
        while True:
            yield await self.liveness.get()

    async def fetch_kv_event_bootstrap(self) -> KvEventBootstrap:
        self.fetch_count += 1
        self._calls.append("metadata")
        if self.metadata_errors:
            raise self.metadata_errors.pop(0)
        return self.bootstrap_result

    async def reset_generation_state(self) -> None:
        self.reset_count += 1

    def map_medium(self, medium: str | None) -> str:
        return ""

    def supported_mediums(self) -> list[str]:
        return []

    def storage_type(self) -> str:
        return "ST_UNSPECIFIED"

    def request_immediate_snapshot(self) -> None:
        return None

    async def close(self) -> None:
        self.closed = True
        self._calls.append("adapter_close")


class FakeKvcmClient(KvcmClient):
    """KVCM client double recording sends and lifecycle calls."""

    def __init__(
        self,
        calls: list[str],
        *,
        registered: bool = True,
        send_error: Exception | None = None,
        location_spec_error: MetadataProtocolError | None = None,
    ) -> None:
        self._calls = calls
        self._registered = registered
        self._send_error = send_error
        self._location_spec_error = location_spec_error
        self.sent: list[tuple[list[KVEventBatch], int]] = []
        self.snapshots: list[tuple[list[KVEventBatch], int]] = []
        self.started = False
        self.closed = False

    @property
    def is_registered(self) -> bool:
        return self._registered

    async def start(self) -> None:
        self.started = True
        self._calls.append("kvcm_start")

    def validate_location_specs(self) -> None:
        self._calls.append("kvcm_validate")
        if self._location_spec_error is not None:
            raise self._location_spec_error

    async def report_kv_events(
        self,
        batches: list[KVEventBatch],
        epoch: int,
        telemetries=None,
        trace_id=None,
        *,
        reregister_after_host_down: bool = True,
    ) -> None:
        self._calls.append("kvcm_send")
        if self._send_error is not None:
            raise self._send_error
        self.sent.append((batches, epoch))

    async def report_snapshot(
        self,
        batches: list[KVEventBatch],
        epoch: int,
        telemetry: object | None = None,
        trace_id: str | None = None,
    ) -> None:
        self._calls.append("kvcm_send_snapshot")
        if self._send_error is not None:
            raise self._send_error
        self.snapshots.append((batches, epoch))

    async def close(self) -> None:
        self.closed = True
        self._calls.append("kvcm_close")


class RecordingReporter(DashservingStateReporter):
    """State reporter double recording the logical report sequence."""

    def __init__(
        self,
        calls: list[str],
        active_event: asyncio.Event,
        *,
        fail_active_times: int = 0,
    ) -> None:
        self._calls = calls
        self._active_event = active_event
        self._fail_active_times = fail_active_times
        self._active_attempts = 0
        self.failed_reasons: list[str] = []

    async def report_active(self) -> None:
        self._active_attempts += 1
        if self._active_attempts <= self._fail_active_times:
            raise httpx.ConnectError("connection refused")
        self._calls.append("active")
        self._active_event.set()

    def start_heartbeat(self) -> None:
        self._calls.append("heartbeat_start")

    async def stop_heartbeat(self) -> None:
        self._calls.append("heartbeat_stop")

    async def report_failed(self, reason: str) -> None:
        self.failed_reasons.append(reason)
        self._calls.append("failed")

    async def report_shutdown_inactive(self, reason: str) -> None:
        self._calls.append("inactive")

    async def close(self) -> None:
        self._calls.append("reporter_close")


class BlockingActiveReporter(RecordingReporter):
    """Reporter that exposes the interval while the initial active POST waits."""

    def __init__(
        self,
        calls: list[str],
        active_event: asyncio.Event,
        active_attempted: asyncio.Event,
        release_active: asyncio.Event,
    ) -> None:
        super().__init__(calls, active_event)
        self._active_attempted = active_attempted
        self._release_active = release_active

    async def report_active(self) -> None:
        self._active_attempted.set()
        await self._release_active.wait()
        await super().report_active()


class _LifecycleBundle:
    def __init__(
        self,
        calls: list[str],
        active_event: asyncio.Event,
        *,
        threshold: int = 1,
        registered: bool = True,
        fail_active_times: int = 0,
        metadata_errors: list[Exception] | None = None,
        send_error: Exception | None = None,
        location_spec_error: MetadataProtocolError | None = None,
        snapshot_ends_immediately: bool = False,
        immediate_snapshot: list[KVEventBatch] | None = None,
        incremental_pipeline_enabled: bool = True,
        snapshot_pipeline_enabled: bool = True,
        reporter: DashservingStateReporter | None = None,
    ) -> None:
        config = SubscriberConfig(
            engine_health_failure_threshold=threshold,
            subscriber_health_enabled=True,
            incremental_kv_event_pipeline_enabled=incremental_pipeline_enabled,
            snapshot_kv_event_pipeline_enabled=snapshot_pipeline_enabled,
        )
        self.adapter = FakeAdapter(
            calls,
            snapshot_ends_immediately=snapshot_ends_immediately,
            immediate_snapshot=immediate_snapshot,
        )
        if metadata_errors:
            self.adapter.metadata_errors = list(metadata_errors)
        self.kvcm = FakeKvcmClient(
            calls,
            registered=registered,
            send_error=send_error,
            location_spec_error=location_spec_error,
        )
        self.reporter = reporter or RecordingReporter(
            calls, active_event, fail_active_times=fail_active_times
        )
        self.shutdown = asyncio.Event()
        self.lifecycle = SubscriberLifecycle(
            config,
            adapter=self.adapter,
            kvcm_client=self.kvcm,
            state_reporter=self.reporter,
            shutdown_event=self.shutdown,
            install_signal_handlers=False,
            startup_retry_delay_s=0.0,
        )
        self.calls = calls
        self.active_event = active_event

    async def start(self) -> asyncio.Task[None]:
        task = asyncio.create_task(self.lifecycle.run())
        await asyncio.wait_for(self.active_event.wait(), timeout=1.0)
        while "heartbeat_start" not in self.calls:
            await asyncio.sleep(0)
        return task

    async def stop(self, task: asyncio.Task[None]) -> None:
        self.shutdown.set()
        await asyncio.wait_for(task, timeout=1.0)


def _make_bundle(**kwargs) -> _LifecycleBundle:
    return _LifecycleBundle([], asyncio.Event(), **kwargs)


# ---------------------------------------------------------------------------
# Startup order and active prerequisites
# ---------------------------------------------------------------------------


async def test_register_kvcm_wires_adapter_snapshot_callback(
    mocker: MockerFixture,
) -> None:
    calls: list[str] = []
    adapter = FakeAdapter(calls)
    lifecycle = SubscriberLifecycle(
        SubscriberConfig(),
        adapter=adapter,
        state_reporter=RecordingReporter(calls, asyncio.Event()),
        install_signal_handlers=False,
    )
    kvcm_cls = mocker.patch("subscriber.main.KvcmClient", autospec=True)
    kvcm_cls.return_value.is_registered = True

    descriptor = KvCacheDescriptor(groups=())
    await lifecycle._register_kvcm(descriptor)

    kvcm_cls.validate_descriptor_location_specs.assert_called_once_with(
        lifecycle._config, descriptor
    )
    kvcm_cls.assert_called_once()
    assert (
        kvcm_cls.call_args.kwargs["on_snapshot_required"]
        == adapter.request_immediate_snapshot
    )


async def test_incompatible_descriptor_is_checked_before_kvcm_construction(
    mocker: MockerFixture,
) -> None:
    calls: list[str] = []
    lifecycle = SubscriberLifecycle(
        SubscriberConfig(),
        adapter=FakeAdapter(calls),
        state_reporter=RecordingReporter(calls, asyncio.Event()),
        install_signal_handlers=False,
    )
    kvcm_cls = mocker.patch("subscriber.main.KvcmClient", autospec=True)
    kvcm_cls.validate_descriptor_location_specs.side_effect = MetadataProtocolError(
        "unknown component kind"
    )

    with pytest.raises(_FatalStartupError, match="KV cache descriptor protocol error"):
        await lifecycle._register_kvcm(KvCacheDescriptor(groups=()))

    kvcm_cls.assert_not_called()


async def test_startup_exact_call_order_and_shutdown_sequence() -> None:
    bundle = _make_bundle()
    task = await bundle.start()
    await bundle.stop(task)

    assert bundle.calls == [
        "watch_started",
        "metadata",
        "kvcm_validate",
        "kvcm_start",
        "active",
        "heartbeat_start",
        "heartbeat_stop",
        "kvcm_send",  # AllBlocksCleared (HostDown)
        "inactive",
        "kvcm_close",
        "reporter_close",
        "adapter_close",
    ]


async def test_startup_logs_complete_bootstrap_info(mocker: MockerFixture) -> None:
    info = mocker.patch("subscriber.main.logger.info")
    bundle = _make_bundle()

    task = await bundle.start()
    await bundle.stop(task)

    bootstrap_log = next(
        call
        for call in info.call_args_list
        if call.args == ("KV event bootstrap fetched",)
    )
    payload = json.loads(bootstrap_log.kwargs["tags"]["bootstrap_info"])
    assert payload["event_transport"]["live_endpoint"] == "tcp://127.0.0.1:5557"
    assert payload["runtime_topology"]["data_parallel_rank"] == 0
    assert payload["vllm"]["event_schema_version"] == 2


async def test_active_not_reported_before_prerequisites() -> None:
    bundle = _make_bundle()
    task = await bundle.start()
    await bundle.stop(task)

    active_idx = bundle.calls.index("active")
    assert bundle.calls.index("metadata") < active_idx
    assert bundle.calls.index("kvcm_start") < active_idx
    assert bundle.calls.index("heartbeat_start") > active_idx


async def test_pipeline_receives_no_kv_event_while_initial_active_is_blocked() -> None:
    calls: list[str] = []
    active_event = asyncio.Event()
    active_attempted = asyncio.Event()
    release_active = asyncio.Event()
    reporter = BlockingActiveReporter(
        calls,
        active_event,
        active_attempted,
        release_active,
    )
    bundle = _LifecycleBundle(calls, active_event, reporter=reporter)
    task = asyncio.create_task(bundle.lifecycle.run())

    await asyncio.wait_for(active_attempted.wait(), timeout=1.0)
    await asyncio.sleep(0)

    assert "kvcm_send" not in calls
    assert bundle.kvcm.sent == []

    release_active.set()
    await asyncio.wait_for(active_event.wait(), timeout=1.0)
    while "heartbeat_start" not in calls:
        await asyncio.sleep(0)
    await bundle.stop(task)


# ---------------------------------------------------------------------------
# Failed-vs-transient classification
# ---------------------------------------------------------------------------


async def test_fatal_metadata_error_reports_failed_and_exits() -> None:
    bundle = _make_bundle(
        metadata_errors=[MetadataProtocolError("bad schema")],
    )
    task = asyncio.create_task(bundle.lifecycle.run())
    await asyncio.wait_for(task, timeout=1.0)

    assert "active" not in bundle.calls
    assert "failed" in bundle.calls
    assert bundle.reporter.failed_reasons == [
        "KV event bootstrap protocol error: bad schema"
    ]
    # KVCM registration never happened; cleanup still ran on the partial path.
    assert bundle.kvcm.started is False
    assert bundle.adapter.closed is True


async def test_incompatible_descriptor_reports_failed_before_kvcm_start() -> None:
    bundle = _make_bundle(
        location_spec_error=MetadataProtocolError("unknown component kind"),
    )

    await asyncio.wait_for(bundle.lifecycle.run(), timeout=1.0)

    assert bundle.reporter.failed_reasons == [
        "KV cache descriptor protocol error: unknown component kind"
    ]
    assert "kvcm_validate" in bundle.calls
    assert "kvcm_start" not in bundle.calls
    assert "active" not in bundle.calls
    assert "heartbeat_start" not in bundle.calls


async def test_transient_metadata_error_retries_then_active() -> None:
    bundle = _make_bundle(
        metadata_errors=[MetadataTemporarilyUnavailable("try again")],
    )
    task = await bundle.start()
    await bundle.stop(task)

    assert bundle.adapter.fetch_count == 2
    assert "active" in bundle.calls
    assert "failed" not in bundle.calls


async def test_transient_active_report_retries_then_active() -> None:
    bundle = _make_bundle(fail_active_times=2)
    task = await bundle.start()
    await bundle.stop(task)

    assert "active" in bundle.calls
    assert "failed" not in bundle.calls


async def test_transient_http_5xx_active_report_retries_then_active() -> None:
    """raise_for_status() raises HTTPStatusError, which is not a
    TransportError subclass; a transient 5xx must retry, not report failed."""

    class _Http5xxOnceReporter(RecordingReporter):
        def __init__(self, calls: list[str], active_event: asyncio.Event) -> None:
            super().__init__(calls, active_event)
            self._raised = False

        async def report_active(self) -> None:
            if not self._raised:
                self._raised = True
                request = httpx.Request("POST", "http://127.0.0.1:8601/state")
                response = httpx.Response(503, request=request)
                raise httpx.HTTPStatusError(
                    "503 Service Unavailable", request=request, response=response
                )
            await super().report_active()

    calls: list[str] = []
    active_event = asyncio.Event()
    bundle = _LifecycleBundle(
        calls, active_event, reporter=_Http5xxOnceReporter(calls, active_event)
    )
    task = await bundle.start()
    await bundle.stop(task)

    assert "active" in bundle.calls
    assert "failed" not in bundle.calls


# ---------------------------------------------------------------------------
# Readiness regression / KVCM outage isolation
# ---------------------------------------------------------------------------


async def test_kvcm_outage_does_not_affect_heartbeat_or_engine_health() -> None:
    bundle = _make_bundle(send_error=KvcmUnavailableError("kvcm down"))
    task = await bundle.start()

    # Push a KV batch; the sender's send_kv_events raises and the batch is dropped.
    await bundle.adapter.kv_queue.put(_kv_batch())
    while "kvcm_send" not in bundle.calls:
        await asyncio.sleep(0)

    # Engine health is driven only by liveness, not KVCM availability.
    assert bundle.lifecycle.coordinator.state is EngineHealthState.HEALTHY
    # The active heartbeat is unaffected by the KVCM outage.
    assert "active" in bundle.calls
    # No HostDown is emitted for a KVCM outage while the engine is healthy.
    assert bundle.kvcm.sent == []

    await bundle.stop(task)


async def test_metadata_protocol_drop_reports_inactive_without_exiting() -> None:
    bundle = _make_bundle(
        send_error=KvcmReportRejectedError(
            "component identity drift",
            status_code="METADATA_PROTOCOL",
            reason="metadata_protocol",
        )
    )
    task = await bundle.start()

    await bundle.adapter.kv_queue.put(_kv_batch())
    while "inactive" not in bundle.calls:
        await asyncio.sleep(0)

    assert task.done() is False
    assert bundle.calls.count("inactive") == 1
    assert bundle.lifecycle.coordinator.state is EngineHealthState.HEALTHY

    await bundle.stop(task)

    assert bundle.calls.count("inactive") == 1


async def test_snapshot_pipeline_forwards_batch_to_kvcm() -> None:
    bundle = _make_bundle()
    task = await bundle.start()

    await bundle.adapter.snapshot_queue.put(_snapshot_batch())
    while not bundle.kvcm.snapshots:
        await asyncio.sleep(0)

    assert len(bundle.kvcm.snapshots) == 1
    assert bundle.kvcm.sent == []
    await bundle.stop(task)


async def test_snapshot_only_mode_disables_incremental_pipeline() -> None:
    bundle = _make_bundle(incremental_pipeline_enabled=False)
    task = await bundle.start()

    assert bundle.lifecycle._incremental_producer_task is None
    assert bundle.lifecycle._incremental_sender_task is None
    assert bundle.lifecycle._incremental_metrics_reporter is None

    for _ in range(17):
        await bundle.adapter.snapshot_queue.put(_snapshot_batch())
    while bundle.adapter.snapshot_consumed_count != 17:
        await asyncio.sleep(0)

    await bundle.adapter.kv_queue.put(_kv_batch())
    await asyncio.sleep(0)
    assert bundle.kvcm.sent == []

    await bundle.stop(task)


async def test_snapshot_pipeline_can_be_disabled_independently() -> None:
    bundle = _make_bundle(snapshot_pipeline_enabled=False)
    task = await bundle.start()

    assert bundle.lifecycle._snapshot_producer_task is None
    assert bundle.lifecycle._snapshot_sender_task is None
    assert bundle.lifecycle._snapshot_metrics_reporter is None

    await bundle.adapter.kv_queue.put(_kv_batch())
    while not bundle.kvcm.sent:
        await asyncio.sleep(0)

    await bundle.stop(task)


async def test_pipeline_metrics_reporters_are_independent_instances() -> None:
    bundle = _make_bundle()
    task = await bundle.start()

    incremental = bundle.lifecycle._incremental_metrics_reporter
    snapshot = bundle.lifecycle._snapshot_metrics_reporter
    assert incremental is not None
    assert snapshot is not None
    # Each pipeline gets its own MetricsReporter instance with a distinct
    # background task name so pipeline-scoped flushes never interfere.
    assert incremental is not snapshot
    assert incremental._task is not None
    assert snapshot._task is not None
    assert incremental._task.get_name() == "incremental-kv-event-metrics"
    assert snapshot._task.get_name() == "snapshot-kv-event-metrics"

    await bundle.stop(task)


# ---------------------------------------------------------------------------
# Recovery without process restart
# ---------------------------------------------------------------------------


async def test_recovery_without_process_restart() -> None:
    bundle = _make_bundle(threshold=1)
    task = await bundle.start()
    coordinator = bundle.lifecycle.coordinator
    assert coordinator.epoch == 1

    # Engine DEAD: emits AllBlocksCleared for epoch 1.
    await bundle.adapter.liveness.put(LivenessEvent.UNHEALTHY)
    while coordinator.state is not EngineHealthState.DEAD:
        await asyncio.sleep(0)
    assert len(bundle.kvcm.sent) == 1

    # Engine recovers in-process: new sendable epoch, no restart.
    await bundle.adapter.liveness.put(LivenessEvent.HEALTHY)
    while coordinator.state is not EngineHealthState.HEALTHY:
        await asyncio.sleep(0)
    assert coordinator.epoch == 2
    assert bundle.adapter.reset_count == 1

    # Graceful shutdown emits HostDown for the new epoch.
    await bundle.stop(task)
    assert len(bundle.kvcm.sent) == 2
    assert bundle.kvcm.sent[1][1] == 2


# ---------------------------------------------------------------------------
# HostDown once / no KV report after HostDown
# ---------------------------------------------------------------------------


async def test_host_down_emitted_once_on_shutdown() -> None:
    bundle = _make_bundle()
    task = await bundle.start()
    await bundle.stop(task)

    host_downs = [
        batches
        for batches, _ in bundle.kvcm.sent
        if any(
            isinstance(e, AllBlocksCleared) for batch in batches for e in batch.events
        )
    ]
    assert len(host_downs) == 1


async def test_engine_dead_then_shutdown_emits_host_down_once() -> None:
    bundle = _make_bundle(threshold=1)
    task = await bundle.start()
    coordinator = bundle.lifecycle.coordinator

    await bundle.adapter.liveness.put(LivenessEvent.UNHEALTHY)
    while coordinator.state is not EngineHealthState.DEAD:
        await asyncio.sleep(0)
    assert len(bundle.kvcm.sent) == 1

    await bundle.stop(task)
    # Shutdown for the same (still current) epoch must not re-emit.
    assert len(bundle.kvcm.sent) == 1


async def test_no_kv_report_after_host_down() -> None:
    bundle = _make_bundle()
    task = await bundle.start()

    # Queue a KV batch that would be forwarded while serving.
    await bundle.adapter.kv_queue.put(_kv_batch())
    while not bundle.kvcm.sent:
        await asyncio.sleep(0)
    forwarded_before = len(bundle.kvcm.sent)

    await bundle.stop(task)

    # The only send after shutdown begins is the single AllBlocksCleared; it is
    # the last KVCM state transition and no KV batch follows it.
    last_batches, _ = bundle.kvcm.sent[-1]
    assert any(
        isinstance(e, AllBlocksCleared) for batch in last_batches for e in batch.events
    )
    assert len(bundle.kvcm.sent) == forwarded_before + 1


# ---------------------------------------------------------------------------
# Shutdown inactive sequence greater than last heartbeat
# ---------------------------------------------------------------------------


class _RecordingTransport(httpx.AsyncBaseTransport):
    def __init__(self) -> None:
        self.states: list[str] = []
        self.seqs: list[int] = []

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        await request.aread()
        payload = json.loads(request.content.decode())
        self.states.append(str(payload["state"]))
        self.seqs.append(int(payload["seq_id"]))
        return httpx.Response(
            200, json={"accepted": True, "last_seq_id": payload["seq_id"]}
        )


async def test_shutdown_inactive_seq_greater_than_last_heartbeat() -> None:
    transport = _RecordingTransport()
    config = SubscriberConfig(
        subscriber_heartbeat_interval_s=0.01,
        subscriber_health_enabled=True,
        engine_health_failure_threshold=1,
    )
    client = httpx.AsyncClient(transport=transport)
    reporter = DashservingStateReporter(config, http_client=client)

    bundle = _LifecycleBundle([], asyncio.Event(), reporter=reporter)
    task = asyncio.create_task(bundle.lifecycle.run())

    # Wait for the initial active report plus at least one heartbeat active.
    while transport.states.count("active") < 2:
        await asyncio.sleep(0.005)

    bundle.shutdown.set()
    await asyncio.wait_for(task, timeout=1.0)

    assert transport.states[-1] == "inactive"
    assert transport.seqs[-1] == max(transport.seqs)
    assert transport.seqs == sorted(transport.seqs)
    await client.aclose()


# ---------------------------------------------------------------------------
# Cleanup after a task failure
# ---------------------------------------------------------------------------


async def test_cleanup_after_sender_task_failure_reraises() -> None:
    bundle = _make_bundle(send_error=RuntimeError("sender exploded"))
    task = await bundle.start()

    # Push a KV batch so the sender calls send_kv_events and fails.
    await bundle.adapter.kv_queue.put(_kv_batch())

    with pytest.raises(RuntimeError, match="sender exploded"):
        await asyncio.wait_for(task, timeout=1.0)

    # Cleanup ran on the failure path.
    assert bundle.kvcm.closed is True
    assert bundle.adapter.closed is True
    assert bundle.kvcm.started is True


async def test_cleanup_after_snapshot_sender_task_failure_reraises(
    mocker: MockerFixture,
) -> None:
    release_failure = asyncio.Event()

    async def failing_snapshot_sender(*args: object, **kwargs: object) -> None:
        await release_failure.wait()
        raise RuntimeError("snapshot sender exploded")

    mocker.patch(
        "subscriber.main.send_snapshot_events",
        new=failing_snapshot_sender,
    )
    bundle = _make_bundle()
    task = await bundle.start()

    release_failure.set()

    with pytest.raises(RuntimeError, match="snapshot sender exploded"):
        await asyncio.wait_for(task, timeout=1.0)

    # Cleanup ran on the failure path.
    assert bundle.kvcm.closed is True
    assert bundle.adapter.closed is True
    assert bundle.kvcm.started is True


async def test_cleanup_after_snapshot_producer_ends_reraises() -> None:
    bundle = _make_bundle(snapshot_ends_immediately=True)
    task = await bundle.start()

    with pytest.raises(RuntimeError, match="snapshot-kv-event-producer ended"):
        await asyncio.wait_for(task, timeout=1.0)

    assert bundle.kvcm.closed is True
    assert bundle.adapter.closed is True
    assert bundle.kvcm.started is True


# ---------------------------------------------------------------------------
# Snapshot-before-active ordering invariant
# ---------------------------------------------------------------------------


async def test_first_snapshot_reaches_kvcm_before_active_accepted() -> None:
    """KVCM receives the initial snapshot while report_active is still in-flight.

    This proves the ordering invariant: KVCM has a baseline snapshot before
    DashServing marks the pod ready (readiness 200) and traffic starts flowing
    incremental events.
    """

    calls: list[str] = []
    active_event = asyncio.Event()
    active_attempted = asyncio.Event()
    release_active = asyncio.Event()
    reporter = BlockingActiveReporter(
        calls,
        active_event,
        active_attempted,
        release_active,
    )
    bundle = _LifecycleBundle(
        calls,
        active_event,
        reporter=reporter,
        immediate_snapshot=_snapshot_batch(),
    )
    task = asyncio.create_task(bundle.lifecycle.run())

    # Wait until report_active is attempted (startup reached step 5).
    await asyncio.wait_for(active_attempted.wait(), timeout=1.0)

    # While active is still blocked, the snapshot pipeline (started in step 4)
    # has already polled and sent the initial snapshot to KVCM.
    while "kvcm_send_snapshot" not in calls:
        await asyncio.sleep(0)
    assert "active" not in calls
    assert len(bundle.kvcm.snapshots) == 1

    # Release active; startup completes normally.
    release_active.set()
    await asyncio.wait_for(active_event.wait(), timeout=1.0)
    while "heartbeat_start" not in calls:
        await asyncio.sleep(0)
    await bundle.stop(task)

    # Snapshot was sent before active was accepted.
    assert calls.index("kvcm_send_snapshot") < calls.index("active")
