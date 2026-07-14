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
from subscriber.types import AllBlocksCleared, BlockRemoved, BlockStored, KVEventBatch
from subscriber.utils.network import resolve_host_ip_port


class KvcmReportEventType(StrEnum):
    """Wire event types accepted by KVCM ReportEvent."""

    NODE_REGISTER = "EVENT_NODE_REGISTER"
    HEARTBEAT = "EVENT_HEARTBEAT"
    BLOCK_ADD = "EVENT_BLOCK_ADD"
    BLOCK_DELETE = "EVENT_BLOCK_DELETE"
    HOST_DOWN = "EVENT_HOST_DOWN"


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
        manager_client: AbstractKvCacheManagerClient | None = None,
    ) -> None:
        self._config = config
        self._medium_mapper = medium_mapper
        self._storage_type = storage_type
        self._supported_mediums = supported_mediums
        self._manager_client: AbstractKvCacheManagerClient = (
            manager_client
            or HttpKvCacheManagerClient(
                self._base_url(),
                request_timeout_seconds=self._config.kvcm_request_timeout_s,
            )
        )
        self._host_ip_port_value: str | None = None
        self._heartbeat_task: asyncio.Task[None] | None = None
        self._engine_config: dict[str, Any] = _get_engine_config_from_env()
        self._registered = False
        self._started = False

    def _base_url(self) -> str:
        virtual_service_id = os.environ.get("KVCM_VSERVICE_ID", "")
        if not virtual_service_id:
            raise ValueError("Please specify KVCM_VSERVICE_ID")
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

    def _location_spec_name(self, block_size: int) -> str:
        return f"vllm_{block_size}"

    def _register_instance_request(self) -> dict[str, object]:
        cfg = self._engine_config
        block_size = self._config_int(cfg, "block_size")
        location_spec_name = self._location_spec_name(block_size)
        raw_dtype = cfg.get("dtype")
        dtype = raw_dtype if isinstance(raw_dtype, str) else ""
        tp_size = self._config_int(cfg, "tensor_parallel_size")
        dp_size = self._config_int(cfg, "data_parallel_size")
        pp_size = self._config_int(cfg, "pipeline_parallel_size")
        return {
            "trace_id": self._trace_id("register_instance"),
            "instance_group": self._instance_group(),
            "instance_id": self._instance_id(),
            "block_size": block_size,
            "location_spec_infos": [{"name": location_spec_name, "size": block_size}],
            "model_deployment": {
                "model_name": "default",
                "dtype": dtype,
                "use_mla": False,
                "tp_size": tp_size,
                "dp_size": dp_size,
                "lora_name": "",
                "pp_size": pp_size,
                "extra": "",
                "user_data": "",
            },
            "location_spec_groups": [
                {"name": "default", "spec_names": [location_spec_name]}
            ],
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

    def _block_specs(self, medium: str) -> list[dict[str, str]]:
        block_size = self._config_int(self._engine_config, "block_size")
        return [
            {
                "name": self._location_spec_name(block_size),
                "uri": f"vllm://{self._host_ip_port()}/{medium}",
            }
        ]

    def _report_events_for_batches(
        self, batches: list[KVEventBatch]
    ) -> list[dict[str, object]]:
        events: list[dict[str, object]] = []
        for batch in batches:
            for event in batch.events:
                if isinstance(event, BlockStored):
                    medium = self._medium_mapper(event.medium)
                    specs = self._block_specs(medium)
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
                    for block_hash in event.block_hashes:
                        events.append(
                            {
                                "event_type": KvcmReportEventType.BLOCK_DELETE,
                                "block_delete": {
                                    "block_key": str(block_hash),
                                    "medium": medium,
                                },
                            }
                        )
                elif isinstance(event, AllBlocksCleared):
                    events.append(
                        {"event_type": KvcmReportEventType.HOST_DOWN, "host_down": {}}
                    )
        return events

    async def start(self) -> None:
        self._host_ip_port_value = await resolve_host_ip_port()
        await self._manager_client.start()
        self._started = True
        if await self._manager_is_ready():
            try:
                await self._register_instance()
                await self._report_events([self._node_register_event()])
                self._registered = True
            except Exception as exc:
                logger.warning(
                    "kvcm initial registration failed (%s); will retry via heartbeat",
                    type(exc).__name__,
                    step="kvcm_register",
                    tags={"message": str(exc)},
                    exc_info=True,
                )
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

    async def _register_instance(self) -> None:
        if not self._started or self._manager_client is None:
            raise RuntimeError("kvcm client has not been started")
        await self._manager_client.register_instance(self._register_instance_request())
        self._registered = True

    async def _heartbeat_loop(self) -> None:
        while True:
            await asyncio.sleep(self._config.kvcm_heartbeat_interval_s)
            if not self._registered:
                if not await self._manager_is_ready():
                    continue
                try:
                    await self._register_instance()
                except Exception as exc:
                    logger.warning(
                        "kvcm registerInstance retry failed (%s: %s)",
                        type(exc).__name__,
                        exc,
                        step="kvcm_register",
                        tags={"phase": "register_instance"},
                        exc_info=True,
                    )
                    continue
                try:
                    await self._report_events([self._node_register_event()])
                except Exception as exc:
                    logger.warning(
                        "kvcm node register report failed (%s: %s)",
                        type(exc).__name__,
                        exc,
                        step="kvcm_register",
                        exc_info=True,
                    )
                    self._registered = False
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
                    exc_info=True,
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
        if not self._registered:
            raise RuntimeError("kvcm client is not ready")
        events = self._report_events_for_batches(batches)
        try:
            if not events:
                response = None
            else:
                response = await self._manager_client.report_event(
                    self._report_event_request(events), check_response=True
                )
        except Exception:
            raise
        if response is None:
            if logger.is_debug_enabled():
                logger.debug(
                    "kvcm send skipped because batch group has no reportable events",
                    step="kvcm_send",
                    tags={"epoch": epoch},
                )
            return
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
