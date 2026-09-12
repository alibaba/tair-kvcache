"""SGLang adapter composed from shared ZMQ transport and gRPC control."""

from __future__ import annotations

from collections.abc import AsyncGenerator

from subscriber.config import DpEndpoint, SubscriberConfig
from subscriber.engine.base import AbstractEngineAdapter, EngineEventBatch
from subscriber.engine.kv_event_control_client import DashllmKvEventControlClient
from subscriber.engine.metadata import KvEventBootstrap, MetadataProtocolError
from subscriber.engine.sglang.control import SglangControl
from subscriber.engine.sglang.incremental import SglangIncrementalSource
from subscriber.engine.sglang.snapshot import SglangSnapshotSource
from subscriber.engine.worker_status_client import DashllmWorkerStatusClient
from subscriber.health.events import LivenessEvent

_KVCM_MEDIUM_MAP = {
    "GPU": "hbm",
    "CPU_PINNED": "mem",
}


@AbstractEngineAdapter.register("sglang")
class SGLangAdapter(AbstractEngineAdapter):
    """Adapt SGLang KV events and DashLLM control RPCs to subscriber contracts."""

    def __init__(self, config: SubscriberConfig) -> None:
        self._config = config
        self._status_client = DashllmWorkerStatusClient(config.engine_grpc_endpoint)
        self._kv_event_control_client = DashllmKvEventControlClient(
            config.engine_kv_event_control_uds_path
        )
        self._incremental: SglangIncrementalSource | None = None
        self._snapshot: SglangSnapshotSource | None = None
        self._control = SglangControl(
            config, self._status_client, self._kv_event_control_client
        )

    async def subscribe_kv_events(self) -> AsyncGenerator[EngineEventBatch, None]:
        """Yield SGLang live and replay batches in source order."""

        if self._incremental is None:
            raise RuntimeError("SGLang bootstrap must be fetched before subscription")
        events = self._incremental.subscribe()
        try:
            async for event_batch in events:
                yield event_batch
        finally:
            await events.aclose()

    async def subscribe_snapshot_events(self) -> AsyncGenerator[EngineEventBatch, None]:
        """Yield periodic full snapshots for KVCM reconciliation."""

        if self._snapshot is None:
            raise RuntimeError("SGLang bootstrap must be fetched before subscription")
        events = self._snapshot.subscribe()
        try:
            async for event_batch in events:
                yield event_batch
        finally:
            await events.aclose()

    async def watch_liveness(self) -> AsyncGenerator[LivenessEvent, None]:
        """Reuse DashLLM's existing GetWorkerStatus control-plane interface."""

        async for event in self._control.watch_liveness():
            yield event

    async def fetch_kv_event_bootstrap(self) -> KvEventBootstrap:
        """Fetch bootstrap and construct component-aware SGLang ZMQ sockets."""

        bootstrap = await self._control.fetch_kv_event_bootstrap()
        # TODO(multi-dp): discover and aggregate one transport per engine DP rank.
        if bootstrap.runtime_topology.data_parallel_size != 1:
            raise MetadataProtocolError(
                "subscriber currently supports only data_parallel_size=1"
            )
        if self._config.snapshot_kv_event_pipeline_enabled:
            if not bootstrap.snapshot.supported:
                raise MetadataProtocolError(
                    "SGLang snapshot transport must be supported"
                )
            if not bootstrap.snapshot.versioned:
                raise MetadataProtocolError(
                    "SGLang snapshot transport must be versioned"
                )
        if (
            self._config.incremental_kv_event_pipeline_enabled
            and not bootstrap.event_transport.replay_supported
        ):
            raise MetadataProtocolError("SGLang event replay must be supported")
        component_ids = {
            component.component_id: component.component_id
            for component in bootstrap.components
        }
        if (
            self._config.incremental_kv_event_pipeline_enabled
            and self._incremental is None
        ):
            self._incremental = SglangIncrementalSource(
                self._config,
                endpoint=DpEndpoint(
                    rank=bootstrap.runtime_topology.data_parallel_rank,
                    zmq_pub_endpoint=bootstrap.event_transport.live_endpoint,
                    zmq_replay_endpoint=bootstrap.event_transport.replay_endpoint,
                    zmq_topic=bootstrap.event_transport.topic,
                ),
                component_group_idxs=component_ids,
            )
        if self._config.snapshot_kv_event_pipeline_enabled and self._snapshot is None:
            self._snapshot = SglangSnapshotSource(
                self._config,
                self._kv_event_control_client,
                component_group_idxs=component_ids,
            )
        return bootstrap

    async def close(self) -> None:
        """Release the shared transport, control helper, and gRPC channel."""

        try:
            await self._control.close()
        finally:
            try:
                if self._snapshot is not None:
                    await self._snapshot.close()
            finally:
                try:
                    if self._incremental is not None:
                        await self._incremental.close()
                finally:
                    try:
                        await self._status_client.close()
                    finally:
                        await self._kv_event_control_client.close()

    async def reset_generation_state(self) -> None:
        """Discard source sequence state after a SGLang generation transition."""

        if self._incremental is not None:
            await self._incremental.reset_generation_state()
        if self._snapshot is not None:
            await self._snapshot.reset_generation_state()

    def request_immediate_snapshot(self) -> None:
        """Wake the SGLang full-snapshot polling loop after bootstrap."""

        if self._snapshot is not None:
            self._snapshot.request_immediate_snapshot()

    def map_medium(self, medium: str | None) -> str:
        if medium is None:
            return ""
        return _KVCM_MEDIUM_MAP.get(medium, "")

    def supported_mediums(self) -> list[str]:
        return list(_KVCM_MEDIUM_MAP.values())

    def storage_type(self) -> str:
        return self._config.kvcm_storage_type
