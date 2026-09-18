from __future__ import annotations

import asyncio
import signal
from collections.abc import Awaitable
from contextlib import suppress
from typing import TypeVar

import grpc.aio
import httpx

from subscriber import logger
from subscriber.config import SubscriberConfig
from subscriber.engine.base import AbstractEngineAdapter
from subscriber.engine.metadata import (
    KvCacheDescriptor,
    KvEventBootstrap,
    MetadataProtocolError,
    MetadataTemporarilyUnavailable,
)
from subscriber.forwarding import (
    consume_incremental_events,
    consume_snapshot_events,
    send_incremental_events,
    send_snapshot_events,
)
from subscriber.health.coordinator import EngineHealthCoordinator
from subscriber.health.state_reporter import DashservingStateReporter
from subscriber.kvcm.client import KvcmClient
from subscriber.metrics import MetricsReporter, init_dashlog
from subscriber.pipeline.context import PipelineContext

__all__ = [
    "PipelineContext",
    "SubscriberLifecycle",
    "consume_incremental_events",
    "consume_snapshot_events",
    "run",
    "send_incremental_events",
    "send_snapshot_events",
]

# Delay between retries of transient startup steps (metadata fetch, KVCM
# registration wait, first active report). Transient startup errors stay
# ``starting`` and keep retrying until the deployment startup/liveness policy
# replaces the Pod; these are not converted to ``failed``.
_STARTUP_RETRY_DELAY_S = 1.0
_REGISTRATION_POLL_S = 0.2
_REGISTRATION_WAIT_WARN_INTERVAL_S = 5.0

_T = TypeVar("_T")


class _FatalStartupError(Exception):
    """A pre-active startup error that must report ``failed`` and exit."""


class _ShutdownRequested(Exception):
    """Shutdown was requested during startup; exit gracefully, not ``failed``."""


class SubscriberLifecycle:
    """Own the subscriber startup order and graceful shutdown sequence.

    Startup order (exactly):
        engine alive -> metadata success -> KVCM two-step registration ->
        pipeline tasks running -> accepted active report -> heartbeat running ->
        forwarding once the engine epoch is ready.

    Fatal pre-active errors (invalid metadata protocol, unsupported adapter)
    report ``failed`` and exit; transient engine/metadata/KVCM errors remain
    ``starting`` and keep retrying. Shutdown follows §1.5 in a single ``finally``
    path; every close is idempotent so a partial startup uses the same path.

    Dependencies (adapter, KVCM client, state reporter) may be injected for
    tests; otherwise they are constructed from ``config``. Tests inject a
    ``shutdown_event`` and never send real OS signals.
    """

    def __init__(
        self,
        config: SubscriberConfig,
        *,
        adapter: AbstractEngineAdapter | None = None,
        kvcm_client: KvcmClient | None = None,
        state_reporter: DashservingStateReporter | None = None,
        shutdown_event: asyncio.Event | None = None,
        install_signal_handlers: bool = True,
        startup_retry_delay_s: float = _STARTUP_RETRY_DELAY_S,
    ) -> None:
        self._config = config
        self._adapter = adapter or AbstractEngineAdapter.create(
            config.engine_type, config
        )
        self._shutdown = shutdown_event or asyncio.Event()
        self._install_signal_handlers = install_signal_handlers
        self._startup_retry_delay_s = startup_retry_delay_s
        self._incremental_pipeline_enabled = (
            config.incremental_kv_event_pipeline_enabled
        )
        self._snapshot_pipeline_enabled = config.snapshot_kv_event_pipeline_enabled

        self._coordinator = EngineHealthCoordinator(self._adapter, None, config)
        # Reporter is only created when the health link is enabled and not
        # injected. When disabled, state reporting is skipped entirely.
        self._state_reporter: DashservingStateReporter | None = None
        if state_reporter is not None:
            self._state_reporter = state_reporter
        elif config.subscriber_health_enabled:
            self._state_reporter = DashservingStateReporter(config)
        else:
            self._state_reporter = None

        self._kvcm: KvcmClient | None = kvcm_client
        self._watch_task: asyncio.Task[None] | None = None
        self._incremental_producer_task: asyncio.Task[None] | None = None
        self._incremental_sender_task: asyncio.Task[None] | None = None
        self._snapshot_producer_task: asyncio.Task[None] | None = None
        self._snapshot_sender_task: asyncio.Task[None] | None = None
        self._incremental_metrics_reporter: MetricsReporter | None = None
        self._snapshot_metrics_reporter: MetricsReporter | None = None
        self._active_reported = False
        self._metadata_protocol_inactive_reported = False
        self._startup_failure: str | None = None
        self._task_failure: BaseException | None = None

    @property
    def coordinator(self) -> EngineHealthCoordinator:
        return self._coordinator

    @property
    def shutdown_event(self) -> asyncio.Event:
        return self._shutdown

    async def run(self) -> None:
        """Run startup, serve until shutdown, then run the shutdown sequence.

        A fatal pre-active error reports ``failed`` and returns normally. A
        non-cancellation forwarding/watch task failure is re-raised after
        cleanup so it is never suppressed.
        """

        original_handlers = self._install_signals()
        try:
            await self._graceful_startup()
            if self._startup_failure is None:
                await self._serve_until_shutdown()
        finally:
            await self._graceful_shutdown()
            self._restore_signals(original_handlers)
        if self._task_failure is not None:
            raise self._task_failure

    # ------------------------------------------------------------------
    # Startup
    # ------------------------------------------------------------------

    async def _graceful_startup(self) -> None:
        try:
            # 1. Engine alive: start the watcher, then gate on a sendable epoch.
            self._watch_task = asyncio.create_task(
                self._coordinator.watch_loop(), name="engine-health-watch"
            )
            await self._await_or_shutdown(
                self._coordinator.wait_ready_epoch(), "before engine ready"
            )
            logger.info("engine ready", step="startup", tags={})

            # 2. Bootstrap success; the adapter opens engine-owned ZMQ endpoints.
            bootstrap = await self._await_or_shutdown(
                self._fetch_kv_event_bootstrap_until_success(),
                "during KV event bootstrap fetch",
            )
            logger.info(
                "KV event bootstrap fetched",
                step="startup",
                tags={
                    "bootstrap_info": bootstrap.to_log_json(),
                },
            )

            # 3. KVCM two-step registration through the subscriber translation.
            await self._await_or_shutdown(
                self._register_kvcm(bootstrap.to_kv_cache_descriptor()),
                "during kvcm registration",
            )
            logger.info("kvcm registered", step="startup", tags={})

            # 4. Pipeline tasks running (gated until the epoch is sendable).
            await self._await_or_shutdown(
                self._start_pipeline(), "during pipeline start"
            )
            logger.info(
                "pipelines started",
                step="startup",
                tags={
                    "incremental_pipeline": self._incremental_pipeline_enabled,
                    "snapshot_pipeline": self._snapshot_pipeline_enabled,
                    "incremental_queue_maxsize": (
                        self._config.kv_event_queue_maxsize
                        if self._incremental_pipeline_enabled
                        else None
                    ),
                    "snapshot_queue_maxsize": (
                        self._config.snapshot_queue_maxsize
                        if self._snapshot_pipeline_enabled
                        else None
                    ),
                },
            )

            # 5. Accepted active report.
            await self._await_or_shutdown(
                self._report_active_until_accepted(), "during active report"
            )

            # 6. Heartbeat running.
            if self._state_reporter is not None and self._active_reported:
                self._state_reporter.start_heartbeat()

            # 7. Forwarding once the engine epoch is ready. The gate opened in
            # step 1; this re-confirms a sendable epoch before serving begins.
            await self._await_or_shutdown(
                self._coordinator.wait_ready_epoch(), "before serving"
            )

            logger.info(
                "subscriber started",
                step="startup",
                tags={
                    "engine_type": self._config.engine_type,
                    "kvcm_base_url": self._config.kvcm_base_url,
                    "kvcm_protocol": self._config.kvcm_protocol,
                    "kvcm_instance_group": self._config.kvcm_instance_group,
                    "engine_grpc_endpoint": self._config.engine_grpc_endpoint,
                    "engine_kv_event_control_uds_path": (
                        self._config.engine_kv_event_control_uds_path
                    ),
                    "host_port": self._config.host_port,
                    "kvcm_heartbeat_interval_s": (
                        self._config.kvcm_heartbeat_interval_s
                    ),
                    "kv_event_merge_max_report_events": (
                        self._config.kv_event_merge_max_report_events
                    ),
                    "kv_event_merge_max_queue_items": (
                        self._config.kv_event_merge_max_queue_items
                    ),
                    "incremental_pipeline_enabled": self._incremental_pipeline_enabled,
                    "snapshot_pipeline_enabled": self._snapshot_pipeline_enabled,
                    "engine_health_interval_s": self._config.engine_health_interval_s,
                    "engine_health_failure_threshold": (
                        self._config.engine_health_failure_threshold
                    ),
                    "subscriber_health_enabled": (
                        self._config.subscriber_health_enabled
                    ),
                },
            )
        except _ShutdownRequested as exc:
            # Graceful shutdown during startup: exit without reporting failed.
            self._startup_failure = str(exc)
            logger.info(
                "subscriber shutdown requested during startup",
                step="startup",
                tags={"reason": str(exc)},
            )
        except _FatalStartupError as exc:
            self._startup_failure = str(exc)
            logger.error(
                "subscriber startup failed; reporting failed",
                step="startup",
                tags={"reason": str(exc)},
            )
            if self._state_reporter is not None:
                await self._state_reporter.report_failed(str(exc))

    async def _await_or_shutdown(self, coro: Awaitable[_T], description: str) -> _T:
        """Race an awaitable against shutdown.

        Raises _ShutdownRequested if shutdown wins.
        """

        task = asyncio.ensure_future(coro)
        shutdown_waiter = asyncio.create_task(self._shutdown.wait())
        done, pending = await asyncio.wait(
            {task, shutdown_waiter}, return_when=asyncio.FIRST_COMPLETED
        )
        for p in pending:
            p.cancel()
            try:
                await p
            except asyncio.CancelledError:
                pass
        if shutdown_waiter in done:
            raise _ShutdownRequested(f"shutdown requested {description}")
        # Re-raise any exception from the task
        return task.result()

    async def _fetch_kv_event_bootstrap_until_success(self) -> KvEventBootstrap:
        """Fetch authoritative bootstrap, retrying transient failures.

        ``MetadataProtocolError`` is fatal and reports ``failed``. Only
        ``MetadataTemporarilyUnavailable``, gRPC
        transport errors, timeouts, and OS connection errors are transient and
        keep the subscriber ``starting``. All other exceptions are fatal.
        """

        while True:
            if self._shutdown.is_set():
                raise _ShutdownRequested("shutdown requested during bootstrap fetch")
            try:
                return await self._adapter.fetch_kv_event_bootstrap()
            except MetadataProtocolError as exc:
                raise _FatalStartupError(
                    f"KV event bootstrap protocol error: {exc}"
                ) from exc
            except (
                MetadataTemporarilyUnavailable,
                grpc.aio.AioRpcError,
                TimeoutError,
                OSError,
            ) as exc:
                logger.warning(
                    "KV event bootstrap temporarily unavailable; retrying",
                    step="startup",
                    tags={"error": exc.__class__.__name__, "message": str(exc)},
                )
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                raise _FatalStartupError(
                    f"KV event bootstrap fetch failed fatally: {exc}"
                ) from exc
            await self._sleep_or_shutdown(self._startup_retry_delay_s)

    async def _register_kvcm(self, descriptor: KvCacheDescriptor) -> None:
        if self._kvcm is None:
            try:
                KvcmClient.validate_descriptor_location_specs(self._config, descriptor)
            except MetadataProtocolError as exc:
                raise _FatalStartupError(
                    f"KV cache descriptor protocol error: {exc}"
                ) from exc
            self._kvcm = KvcmClient(
                self._config,
                medium_mapper=self._adapter.map_medium,
                storage_type=self._adapter.storage_type(),
                supported_mediums=self._adapter.supported_mediums(),
                descriptor=descriptor,
                on_snapshot_required=self._adapter.request_immediate_snapshot,
            )
        else:
            try:
                self._kvcm.validate_location_specs()
            except MetadataProtocolError as exc:
                raise _FatalStartupError(
                    f"KV cache descriptor protocol error: {exc}"
                ) from exc
        await self._kvcm.start()
        self._coordinator.attach_kvcm_client(self._kvcm)
        await self._wait_kvcm_registered()

    async def _wait_kvcm_registered(self) -> None:
        """Wait for the KVCM two-step registration to succeed.

        ``KvcmClient.start()`` may return in a not-ready state; registration is
        completed by its internal heartbeat loop. A not-yet-registered state is
        transient, so this waits (polling ``is_registered``) rather than failing.
        A periodic warning surfaces a stalled registration.
        """

        assert self._kvcm is not None
        waited_s = 0.0
        last_warn_s = 0.0
        while not self._kvcm.is_registered:
            await self._await_or_shutdown(
                self._sleep_or_shutdown(_REGISTRATION_POLL_S),
                "during kvcm registration",
            )
            waited_s += _REGISTRATION_POLL_S
            if waited_s - last_warn_s >= _REGISTRATION_WAIT_WARN_INTERVAL_S:
                last_warn_s = waited_s
                logger.warning(
                    "still waiting for kvcm registration",
                    step="startup",
                    tags={"waited_s": round(waited_s, 1)},
                )

    async def _start_pipeline(self) -> None:
        assert self._kvcm is not None
        if self._incremental_pipeline_enabled:
            self._incremental_metrics_reporter = MetricsReporter(
                task_name="incremental-kv-event-metrics",
            )
            await self._incremental_metrics_reporter.start()
            incremental_queue: asyncio.Queue[PipelineContext] = asyncio.Queue(
                maxsize=self._config.kv_event_queue_maxsize
            )
            self._incremental_producer_task = asyncio.create_task(
                consume_incremental_events(
                    self._adapter,
                    self._coordinator,
                    incremental_queue,
                    self._incremental_metrics_reporter,
                ),
                name="incremental-kv-event-producer",
            )
            self._incremental_sender_task = asyncio.create_task(
                send_incremental_events(
                    self._kvcm,
                    self._coordinator,
                    incremental_queue,
                    max_merged_report_events=(
                        self._config.kv_event_merge_max_report_events
                    ),
                    max_merged_queue_items=self._config.kv_event_merge_max_queue_items,
                    pipeline="incremental",
                    on_snapshot_required=self._adapter.request_immediate_snapshot,
                    on_metadata_protocol_error=(
                        self._report_metadata_protocol_inactive
                    ),
                ),
                name="incremental-kv-event-sender",
            )

        if self._snapshot_pipeline_enabled:
            self._snapshot_metrics_reporter = MetricsReporter(
                task_name="snapshot-kv-event-metrics",
            )
            await self._snapshot_metrics_reporter.start()
            snapshot_queue: asyncio.Queue[PipelineContext] = asyncio.Queue(
                maxsize=self._config.snapshot_queue_maxsize
            )
            self._snapshot_producer_task = asyncio.create_task(
                consume_snapshot_events(
                    self._adapter,
                    self._coordinator,
                    snapshot_queue,
                    self._snapshot_metrics_reporter,
                ),
                name="snapshot-kv-event-producer",
            )
            self._snapshot_sender_task = asyncio.create_task(
                send_snapshot_events(
                    self._kvcm,
                    self._coordinator,
                    snapshot_queue,
                    pipeline="snapshot",
                    on_metadata_protocol_error=(
                        self._report_metadata_protocol_inactive
                    ),
                ),
                name="snapshot-kv-event-sender",
            )

    async def _report_active_until_accepted(self) -> None:
        if self._state_reporter is None:
            return
        while True:
            if self._shutdown.is_set():
                raise _ShutdownRequested("shutdown requested during active report")
            try:
                await self._state_reporter.report_active()
                self._active_reported = True
                return
            except (
                httpx.TransportError,
                httpx.HTTPStatusError,
                TimeoutError,
                OSError,
            ) as exc:
                logger.warning(
                    "active state report not accepted; retrying",
                    step="startup",
                    tags={"error": exc.__class__.__name__, "message": str(exc)},
                )
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                raise _FatalStartupError(
                    f"active state report failed fatally: {exc}"
                ) from exc
            await self._sleep_or_shutdown(self._startup_retry_delay_s)

    async def _report_metadata_protocol_inactive(self) -> None:
        """Make DashServing terminally unhealthy after a local metadata breach."""

        if self._metadata_protocol_inactive_reported:
            return
        self._metadata_protocol_inactive_reported = True
        if self._state_reporter is not None:
            await self._state_reporter.report_shutdown_inactive("metadata_protocol")

    # ------------------------------------------------------------------
    # Serving
    # ------------------------------------------------------------------

    async def _serve_until_shutdown(self) -> None:
        assert self._watch_task is not None
        pipeline_tasks = [
            self._watch_task,
        ]
        if self._snapshot_pipeline_enabled:
            assert self._snapshot_producer_task is not None
            assert self._snapshot_sender_task is not None
            pipeline_tasks.extend(
                [self._snapshot_producer_task, self._snapshot_sender_task]
            )
        if self._incremental_pipeline_enabled:
            assert self._incremental_producer_task is not None
            assert self._incremental_sender_task is not None
            pipeline_tasks.extend(
                [self._incremental_producer_task, self._incremental_sender_task]
            )
        shutdown_waiter = asyncio.create_task(
            self._shutdown.wait(), name="shutdown-waiter"
        )
        tasks = [*pipeline_tasks, shutdown_waiter]
        try:
            done, _ = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
        finally:
            shutdown_waiter.cancel()
            with suppress(asyncio.CancelledError):
                await shutdown_waiter
        first_failure: BaseException | None = None
        for task in pipeline_tasks:
            if task in done and not task.cancelled():
                exc = task.exception()
                if exc is None:
                    exc = RuntimeError(f"{task.get_name()} ended unexpectedly")
                logger.error(
                    "subscriber task failed; initiating shutdown",
                    step="lifecycle",
                    tags={
                        "task": task.get_name(),
                        "error": exc.__class__.__name__,
                        "message": str(exc),
                    },
                )
                if first_failure is None:
                    first_failure = exc
        if first_failure is not None:
            self._task_failure = first_failure
            self._shutdown.set()

    # ------------------------------------------------------------------
    # Shutdown (§1.5)
    # ------------------------------------------------------------------

    async def _graceful_shutdown(self) -> None:
        logger.info("graceful shutdown started", step="shutdown", tags={})
        # 1. Set shutdown flag: no new forwarding permit is issued.
        self._shutdown.set()

        # 2. Stop/cancel pipeline tasks, await them, so no KV batch can race
        #    behind AllBlocksCleared. The watch task is stopped too so it
        #    cannot emit a concurrent HostDown for the same epoch.
        await self._cancel_task(self._incremental_producer_task)
        await self._cancel_task(self._incremental_sender_task)
        await self._cancel_task(self._snapshot_producer_task)
        await self._cancel_task(self._snapshot_sender_task)
        await self._cancel_task(self._watch_task)
        if self._incremental_metrics_reporter is not None:
            await self._incremental_metrics_reporter.stop()
            self._incremental_metrics_reporter = None
        if self._snapshot_metrics_reporter is not None:
            await self._snapshot_metrics_reporter.stop()
            self._snapshot_metrics_reporter = None
        logger.info("pipeline tasks stopped", step="shutdown", tags={})

        # 3. Stop and await the state heartbeat task.
        if self._state_reporter is not None:
            await self._state_reporter.stop_heartbeat()

        # 4. HostDown: at most one AllBlocksCleared for a sendable epoch.
        await self._coordinator.report_host_down("shutdown")
        logger.info("host_down reported", step="shutdown", tags={})

        # 5. POST higher-seq inactive with a bounded timeout (best effort).
        if (
            self._state_reporter is not None
            and not self._metadata_protocol_inactive_reported
        ):
            await self._state_reporter.report_shutdown_inactive("shutdown")
            logger.info("inactive reported", step="shutdown", tags={})

        # 6. Close KVCM, gRPC/HTTP clients, engine adapter (each idempotent).
        if self._kvcm is not None:
            await self._kvcm.close()
        if self._state_reporter is not None:
            await self._state_reporter.close()
        await self._adapter.close()
        logger.info("graceful shutdown complete", step="shutdown", tags={})

    @staticmethod
    async def _cancel_task(task: asyncio.Task[None] | None) -> None:
        if task is None or task.done():
            return
        task.cancel()
        with suppress(asyncio.CancelledError):
            await task

    async def _sleep_or_shutdown(self, delay_s: float) -> None:
        with suppress(asyncio.TimeoutError):
            await asyncio.wait_for(self._shutdown.wait(), timeout=delay_s)

    # ------------------------------------------------------------------
    # Signal handling
    # ------------------------------------------------------------------

    def _install_signals(self) -> dict[int, object]:
        original: dict[int, object] = {}
        if not self._install_signal_handlers:
            return original
        try:
            loop = asyncio.get_running_loop()
            for sig in (signal.SIGTERM, signal.SIGINT):
                original[sig] = signal.getsignal(sig)
                loop.add_signal_handler(sig, self._request_shutdown, sig)
        except (NotImplementedError, RuntimeError):
            # Platform without loop signal handlers (e.g. Windows proactor) or
            # no running loop: rely on the injected/external shutdown event.
            return {}
        return original

    def _restore_signals(self, original: dict[int, object]) -> None:
        if not original:
            return
        try:
            loop = asyncio.get_running_loop()
            for sig in original:
                loop.remove_signal_handler(sig)
            for sig, handler in original.items():
                if callable(handler):
                    signal.signal(sig, handler)
        except (NotImplementedError, RuntimeError, ValueError):
            pass

    def _request_shutdown(self, sig: signal.Signals) -> None:
        logger.info(
            "received signal; requesting graceful shutdown",
            step="lifecycle",
            tags={"signal": sig.name},
        )
        self._shutdown.set()


async def run(
    config: SubscriberConfig,
    *,
    shutdown_event: asyncio.Event | None = None,
    adapter: AbstractEngineAdapter | None = None,
    kvcm_client: KvcmClient | None = None,
    state_reporter: DashservingStateReporter | None = None,
    install_signal_handlers: bool = True,
) -> None:
    """Run the subscriber until shutdown, cancellation, or fatal startup error."""

    init_dashlog("kvcache-subscriber")
    lifecycle = SubscriberLifecycle(
        config,
        adapter=adapter,
        kvcm_client=kvcm_client,
        state_reporter=state_reporter,
        shutdown_event=shutdown_event,
        install_signal_handlers=install_signal_handlers,
    )
    await lifecycle.run()
