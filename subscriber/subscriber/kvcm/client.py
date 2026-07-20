from __future__ import annotations

import asyncio
import json
import os
import time
from collections.abc import Callable
from enum import StrEnum
from typing import Any

from subscriber import logger
from subscriber.config import SubscriberConfig
from subscriber.kvcm.base import AbstractKvCacheManagerClient
from subscriber.kvcm.manager_client import HttpKvCacheManagerClient
from subscriber.types import (
    AllBlocksCleared,
    BlockRemoved,
    BlockStored,
    KvCacheGroupSpec,
    KVEventBatch,
)
from subscriber.utils.network import resolve_host_ip_port


class KvcmReportEventType(StrEnum):
    """Wire event types accepted by KVCM ReportEvent."""

    NODE_REGISTER = "EVENT_NODE_REGISTER"
    HEARTBEAT = "EVENT_HEARTBEAT"
    BLOCK_ADD = "EVENT_BLOCK_ADD"
    BLOCK_DELETE = "EVENT_BLOCK_DELETE"
    HOST_DOWN = "EVENT_HOST_DOWN"


# Map vLLM ``KVCacheSpecKind`` values to the short kvcm spec-name prefix.
# Unmapped kinds fall back to their raw kind string.
_KIND_DISPLAY_NAMES = {
    "full_attention": "full",
    "mamba": "linear",
}


def _get_engine_config_from_env() -> dict[str, Any]:
    raw = os.environ.get("DS_LLM_ENGINE_CONFIG", "")
    if not raw:
        return {}
    try:
        parsed = json.loads(raw)
        if not isinstance(parsed, dict):
            return {}
        return parsed
    except (json.JSONDecodeError, TypeError):
        logger.warning(
            "failed to parse DS_LLM_ENGINE_CONFIG",
            step="kvcm_client_init",
            exc_info=True,
        )
        return {}


class KvcmClient:
    """Async boundary for forwarding KV event batches to kvcm.

    TODO: Define restart and idempotent-close semantics if a future caller needs
    to reuse a client after ``close()``. The current lifecycle is one
    ``start()`` followed by one ``close()``.
    """

    def __init__(
        self,
        config: SubscriberConfig,
        *,
        medium_mapper: Callable[[str | None], str],
        storage_type: str,
        supported_mediums: list[str],
        location_spec_namer: Callable[[int], str] | None = None,
        location_uri_builder: Callable[[str, str], str] | None = None,
        manager_client: AbstractKvCacheManagerClient | None = None,
        group_metadata: list[KvCacheGroupSpec] | None = None,
        learn_mode: bool = False,
    ) -> None:
        self._config = config
        self._medium_mapper = medium_mapper
        self._storage_type = storage_type
        self._supported_mediums = supported_mediums
        self._location_spec_namer = location_spec_namer or (
            lambda block_size: f"vllm_{block_size}"
        )
        self._location_uri_builder = location_uri_builder or (
            lambda host_ip_port, medium: f"vllm://{host_ip_port}/{medium}"
        )
        self._group_by_idx: dict[int, KvCacheGroupSpec] | None = (
            {spec.group_idx: spec for spec in group_metadata}
            if group_metadata is not None
            else None
        )
        self._learn_mode = learn_mode
        self._learn_mode_registration_warning_emitted = False
        self._manager_client: AbstractKvCacheManagerClient = (
            manager_client
            or HttpKvCacheManagerClient(
                self._base_url(),
                request_timeout_seconds=self._config.kvcm_request_timeout_s,
            )
        )
        self._host_ip_port_value: str | None = None
        self._heartbeat_task: asyncio.Task[None] | None = None
        self._send_lock = asyncio.Lock()
        self._engine_config: dict[str, Any] = _get_engine_config_from_env()
        self._registered = False
        self._engine_available = config.engine_type != "rtp_llm"
        self._initial_reset_pending = (
            config.engine_type == "rtp_llm" and config.rtp_reset_on_start
        )
        self._started = False

    def update_group_metadata(self, group_by_idx: dict[int, KvCacheGroupSpec]) -> None:
        """Replace the group metadata used for spec name generation.

        Called by the learn-mode pipeline when the ``GroupMetadataLearner``
        discovers new groups from live events.
        """
        self._group_by_idx = dict(group_by_idx)

    def _base_url(self) -> str:
        configured_base_url = self._config.kvcm_base_url.strip()
        if configured_base_url:
            return configured_base_url
        virtual_service_id = os.environ.get("KVCM_VSERVICE_ID", "")
        if not virtual_service_id:
            raise ValueError("Please specify kvcm_base_url or KVCM_VSERVICE_ID")
        return f"spectrum://{virtual_service_id}:6382"

    def _instance_group(self) -> str:
        return os.environ.get("KVCM_INSTANCE_GROUP", "")

    def _instance_id(self) -> str:
        return os.environ.get("SPECTRUM_DEPLOYMENT_NAME", "")

    def _host_ip_port(self) -> str:
        if self._host_ip_port_value is not None:
            return self._host_ip_port_value
        raise RuntimeError("kvcm client host identity has not been resolved")

    def _trace_id(self, operation: str) -> str:
        return f"subscriber_{operation}_{time.monotonic_ns()}"

    @staticmethod
    def _config_int(cfg: dict[str, Any], key: str, default: int = 1) -> int:
        raw = cfg.get(key)
        if isinstance(raw, bool) or not isinstance(raw, int) or raw <= 0:
            return default
        return raw

    def _location_spec_name(self, block_size: int, group_idx: int | None = None) -> str:
        if self._group_by_idx is None:
            return self._location_spec_namer(block_size)
        if group_idx is None:
            logger.warning(
                "event missing group_idx in multi-group engine; "
                "using fallback spec name",
                step="kvcm_send",
                tags={"block_size": block_size},
            )
            return self._location_spec_namer(block_size)
        spec = self._group_by_idx.get(group_idx)
        if spec is None:
            logger.warning(
                "group_idx not present in kv cache metadata; using fallback spec name",
                step="kvcm_send",
                tags={"group_idx": group_idx},
            )
            return self._location_spec_namer(block_size)
        display = _KIND_DISPLAY_NAMES.get(spec.kind, spec.kind)
        return f"{display}_{group_idx}"

    def _location_specs(
        self, block_size: int
    ) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
        """Build ``location_spec_infos`` and ``location_spec_groups``.

        With group metadata, each group becomes its own location spec named
        ``{display_kind}_{group_idx}`` sized by that group's own ``block_size``,
        and each spec gets its own single-member location spec group of the same
        name. Without metadata a single ``default`` group is used, matching
        pre-hybrid engines.
        """

        if self._group_by_idx is None:
            name = self._location_spec_name(block_size)
            infos: list[dict[str, object]] = [{"name": name, "size": block_size}]
            groups: list[dict[str, object]] = [
                {"name": "default", "spec_names": [name]}
            ]
            return infos, groups

        infos = []
        groups = []
        for spec in self._group_by_idx.values():
            name = self._location_spec_name(spec.block_size, spec.group_idx)
            infos.append({"name": name, "size": spec.block_size})
            groups.append({"name": name, "spec_names": [name]})
        return infos, groups

    def _register_instance_request(self) -> dict[str, object]:
        cfg = self._engine_config
        block_size = self._config_int(cfg, "block_size")
        location_spec_infos, location_spec_groups = self._location_specs(block_size)
        raw_dtype = cfg.get("dtype")
        dtype = (
            raw_dtype
            if isinstance(raw_dtype, str) and raw_dtype
            else self._config.model_dtype
        )
        raw_model_name = cfg.get("model_name")
        model_name = (
            raw_model_name
            if isinstance(raw_model_name, str) and raw_model_name
            else self._config.model_name
        )
        tp_size = self._config_int(
            cfg,
            "tensor_parallel_size",
            self._config.tensor_parallel_size,
        )
        dp_size = self._config_int(
            cfg,
            "data_parallel_size",
            self._config.data_parallel_size,
        )
        pp_size = self._config_int(
            cfg,
            "pipeline_parallel_size",
            self._config.pipeline_parallel_size,
        )
        return {
            "trace_id": self._trace_id("register_instance"),
            "instance_group": self._instance_group(),
            "instance_id": self._instance_id(),
            "block_size": block_size,
            "location_spec_infos": location_spec_infos,
            "model_deployment": {
                "model_name": model_name,
                "dtype": dtype,
                "use_mla": bool(cfg.get("use_mla", self._config.use_mla)),
                "tp_size": tp_size,
                "dp_size": dp_size,
                "lora_name": "",
                "pp_size": pp_size,
                "extra": "",
                "user_data": "",
            },
            "location_spec_groups": location_spec_groups,
        }

    def _node_register_event(self) -> dict[str, object]:
        return {
            "event_type": KvcmReportEventType.NODE_REGISTER,
            "node_register": {"mediums": self._supported_mediums},
        }

    def _heartbeat_event(self) -> dict[str, object]:
        return {
            "event_type": KvcmReportEventType.HEARTBEAT,
            "heartbeat": {"system_status": {}},
        }

    def _report_event_request(
        self, events: list[dict[str, object]]
    ) -> dict[str, object]:
        return {
            "trace_id": self._trace_id("report_event"),
            "instance_id": self._instance_id(),
            "host_ip_port": self._host_ip_port(),
            "events": events,
            "storage_type": self._storage_type,
        }

    def _block_specs(
        self, medium: str, group_idx: int | None = None
    ) -> list[dict[str, str]]:
        block_size = self._config_int(self._engine_config, "block_size")
        return [
            {
                "name": self._location_spec_name(block_size, group_idx),
                "uri": self._location_uri_builder(self._host_ip_port(), medium),
            }
        ]

    def _block_spec_names(self) -> list[str]:
        block_size = self._config_int(
            self._engine_config,
            "block_size",
            self._config.block_size,
        )
        return [self._location_spec_name(block_size)]

    def _report_events_for_batches(
        self, batches: list[KVEventBatch]
    ) -> list[dict[str, object]]:
        events: list[dict[str, object]] = []
        for batch in batches:
            for event in batch.events:
                if isinstance(event, BlockStored):
                    medium = self._medium_mapper(event.medium)
                    specs = self._block_specs(medium, event.group_idx)
                    for block_hash in event.block_hashes:
                        events.append(
                            {
                                "event_type": KvcmReportEventType.BLOCK_ADD,
                                "block_add": {
                                    "block_key": str(block_hash),
                                    "medium": medium,
                                    "specs": specs,
                                },
                            }
                        )
                elif isinstance(event, BlockRemoved):
                    medium = self._medium_mapper(event.medium)
                    specs = self._block_specs(medium, event.group_idx)
                    for block_hash in event.block_hashes:
                        events.append(
                            {
                                "event_type": KvcmReportEventType.BLOCK_DELETE,
                                "block_delete": {
                                    "block_key": str(block_hash),
                                    "medium": medium,
                                    "specs": specs,
                                },
                            }
                        )
                elif isinstance(event, AllBlocksCleared):
                    events.append(
                        {"event_type": KvcmReportEventType.HOST_DOWN, "host_down": {}}
                    )
        return events

    async def start(self) -> None:
        self._host_ip_port_value = (
            self._config.host_ip_port or await resolve_host_ip_port()
        )
        await self._manager_client.start()
        self._started = True
        if not self._engine_available:
            logger.info(
                "waiting for RTP cache snapshot before registering with kvcm",
                step="kvcm_register",
            )
        elif await self._manager_is_ready():
            await self._register_and_report_node()
        else:
            logger.warning(
                "kvcm has no available endpoint; starting in not-ready state",
                step="kvcm_register",
            )
        self._heartbeat_task = asyncio.create_task(self._heartbeat_loop())

    async def _manager_is_ready(self) -> bool:
        if self._manager_client is None:
            return False
        return await self._manager_client.is_ready()

    async def _register_instance(self) -> dict[str, object]:
        if not self._started or self._manager_client is None:
            raise RuntimeError("kvcm client has not been started")
        request = self._register_instance_request()
        await self._manager_client.register_instance(request)
        return request

    async def _register_and_report_node(self) -> bool:
        """Perform the two-step kvcm registration: the register_instance RPC
        followed by the NODE_REGISTER event report. ``_registered`` is left
        false unless both operations complete successfully."""

        try:
            registration_request = await self._register_instance()
        except Exception as exc:
            logger.warning(
                "kvcm register_instance failed (%s: %s)",
                type(exc).__name__,
                exc,
                step="kvcm_register",
                tags={"phase": "register_instance"},
                exc_info=exc,
            )
            return False
        try:
            await self._report_events([self._node_register_event()])
        except Exception as exc:
            logger.warning(
                "kvcm node_register report failed (%s: %s)",
                type(exc).__name__,
                exc,
                step="kvcm_register",
                tags={"phase": "node_register"},
                exc_info=exc,
            )
            return False
        self._registered = True
        if self._learn_mode and not self._learn_mode_registration_warning_emitted:
            self._learn_mode_registration_warning_emitted = True
            location_spec_infos = registration_request.get("location_spec_infos")
            registered_spec_names = (
                [
                    name
                    for spec in location_spec_infos
                    if isinstance(spec, dict)
                    and isinstance((name := spec.get("name")), str)
                ]
                if isinstance(location_spec_infos, list)
                else []
            )
            logger.warning(
                "learn-mode registered KVCM with default specs; later learned "
                "group metadata cannot update the active registration",
                step="kv_metadata_learn",
                tags={
                    "registered_spec_names": registered_spec_names,
                    "block_size": registration_request["block_size"],
                },
            )
        return True

    async def set_engine_available(self, available: bool) -> None:
        """Pause node heartbeats while the inference engine is down.

        Recovery performs a fresh NODE_REGISTER before forwarding the next
        cache generation. This prevents the heartbeat loop from making a dead
        engine node available again immediately after HOST_DOWN.
        """

        async with self._send_lock:
            self._engine_available = available
            if not available:
                return
            if self._initial_reset_pending:
                # The first authoritative RTP snapshot must clear an old host
                # generation before NODE_REGISTER makes it available again.
                return
            if (
                self._started
                and not self._registered
                and await self._manager_is_ready()
            ):
                await self._register_and_report_node()

    async def _heartbeat_loop(self) -> None:
        while True:
            await asyncio.sleep(self._config.kvcm_heartbeat_interval_s)
            async with self._send_lock:
                if not self._engine_available or self._initial_reset_pending:
                    continue
                if not self._registered:
                    if not await self._manager_is_ready():
                        continue
                    await self._register_and_report_node()
                    if not self._registered:
                        # A heartbeat must never make a node visible before
                        # both RegisterInstance and NODE_REGISTER succeed.
                        continue
                    logger.info(
                        "kvcm registration recovered",
                        step="kvcm_register",
                    )

                try:
                    await self._report_events([self._heartbeat_event()])
                except Exception as exc:
                    self._registered = False
                    logger.warning(
                        "kvcm heartbeat report failed (%s: %s)",
                        type(exc).__name__,
                        exc,
                        step="kvcm_heartbeat",
                        exc_info=exc,
                    )

    async def _report_events(self, events: list[dict[str, object]]) -> dict[str, Any]:
        if not self._started or self._manager_client is None:
            raise RuntimeError("kvcm client has not been started")
        return await self._manager_client.report_event(
            self._report_event_request(events)
        )

    async def send_batch(self, batches: list[KVEventBatch], epoch: int) -> None:
        if not self._started or self._manager_client is None:
            raise RuntimeError("kvcm client has not been started")
        events = self._report_events_for_batches(batches)
        if not events:
            if logger.is_debug_enabled():
                logger.debug(
                    "kvcm send skipped because batch group has no reportable events",
                    step="kvcm_send",
                    tags={"epoch": epoch},
                )
            return
        async with self._send_lock:
            host_down_events = [
                event
                for event in events
                if event["event_type"] == KvcmReportEventType.HOST_DOWN
            ]
            mutation_events = [
                event
                for event in events
                if event["event_type"] != KvcmReportEventType.HOST_DOWN
            ]
            initial_reset = self._initial_reset_pending and bool(host_down_events)
            if not self._registered:
                if not initial_reset or not self._engine_available:
                    raise RuntimeError("kvcm client is not ready")
                # HOST_DOWN identifies the node by instance/host and does not
                # require NODE_REGISTER. Register only the instance first so
                # stale block state is never made available during cold start.
                await self._register_instance()
            if host_down_events:
                # KVCM applies block mutations before HOST_DOWN within one
                # request. Split the reset so the new generation cannot be
                # removed by the asynchronous old-generation cleanup.
                await self._send_report_events(host_down_events, epoch)
                self._registered = False
                if initial_reset:
                    self._initial_reset_pending = False
                if not mutation_events:
                    if initial_reset and not await self._register_and_report_node():
                        raise RuntimeError(
                            "kvcm node registration failed after host reset"
                        )
                    return
                if not self._engine_available:
                    raise RuntimeError(
                        "cannot report cache snapshot while engine is down"
                    )
                if not await self._register_and_report_node():
                    raise RuntimeError("kvcm node registration failed after host reset")

            await self._send_report_events(mutation_events, epoch)

    async def _send_report_events(
        self,
        events: list[dict[str, object]],
        epoch: int,
    ) -> None:
        batch_size = self._config.kvcm_report_batch_size
        for offset in range(0, len(events), batch_size):
            response = await self._manager_client.report_event(
                self._report_event_request(events[offset : offset + batch_size]),
                check_response=True,
            )
            item_results = (
                response.get("item_results") if isinstance(response, dict) else None
            )
            if item_results:
                logger.warning(
                    "kvcm report_event returned partial item results",
                    step="kvcm_send",
                    tags={"epoch": epoch, "item_results": item_results},
                )

    async def close(self) -> None:
        if self._heartbeat_task is not None:
            self._heartbeat_task.cancel()
            try:
                await self._heartbeat_task
            except asyncio.CancelledError:
                pass
            self._heartbeat_task = None
        if self._manager_client is not None:
            await self._manager_client.close()
        self._registered = False
        self._started = False
