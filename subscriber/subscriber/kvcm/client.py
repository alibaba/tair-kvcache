from __future__ import annotations

import asyncio
import os
import time
from collections.abc import Callable
from enum import StrEnum
from typing import Any

from subscriber.kvcm.manager_client import HttpKvCacheManagerClient

from subscriber import logger
from subscriber.config import SubscriberConfig
from subscriber.types import AllBlocksCleared, BlockRemoved, BlockStored, KVEventBatch
from subscriber.utils.network import resolve_host_ip_port

_SDK_PLACEHOLDER_BASE_URL = "http://127.0.0.1:1"
_MEDIUM_MAP = {"GPU": "hbm", "CPU": "mem"}
_ENGINE_STORAGE_TYPE_MAP = {"vllm": "ST_VLLM"}


class KvcmReportEventType(StrEnum):
    """Wire event types accepted by KVCM ReportEvent."""

    NODE_REGISTER = "EVENT_NODE_REGISTER"
    HEARTBEAT = "EVENT_HEARTBEAT"
    BLOCK_ADD = "EVENT_BLOCK_ADD"
    BLOCK_DELETE = "EVENT_BLOCK_DELETE"
    HOST_DOWN = "EVENT_HOST_DOWN"


class KvcmClient:
    """Async boundary for forwarding KV event batches to kvcm."""

    def __init__(
        self,
        config: SubscriberConfig,
        *,
        sdk_client_factory: Callable[[str], Any] | None = None,
    ) -> None:
        self._config = config
        self._sdk_client_factory = sdk_client_factory or HttpKvCacheManagerClient
        self._sdk_client: Any | None = None
        self._host_ip_port_value: str | None = None
        self._heartbeat_task: asyncio.Task[None] | None = None
        # TODO(kvcm): Replace the blocking wheel SDK with an async httpx client
        # and enforce request timeouts so an unavailable KVCM cannot hang a sender.
        # TODO(kvcm): After the async migration, use separate httpx.AstncClient for heartbeat
        # and ReportEvent traffic instead of sharing one request client concurrently.

    def _create_sdk_client(self) -> Any:
        return self._sdk_client_factory(_SDK_PLACEHOLDER_BASE_URL)

    def _instance_id(self) -> str:
        return os.environ.get("SPECTRUM_DEPLOYMENT_NAME", "")

    def _host_ip_port(self) -> str:
        if self._host_ip_port_value is not None:
            return self._host_ip_port_value
        return resolve_host_ip_port(
            self._config.kvcm_host_ip_port, self._config.engine_health_url
        )

    def _storage_type(self) -> str:
        return _ENGINE_STORAGE_TYPE_MAP.get(
            self._config.engine_type.lower(), "ST_UNSPECIFIED"
        )

    def _trace_id(self, operation: str) -> str:
        return f"subscriber_{operation}_{time.monotonic_ns()}"

    def _register_instance_request(self) -> dict[str, object]:
        return {
            "trace_id": self._trace_id("register_instance"),
            "instance_group": "default",
            "instance_id": self._instance_id(),
            "block_size": 1,
            "location_spec_infos": [{"name": "default", "size": 1}],
            "model_deployment": {
                "model_name": "default",
                "dtype": "",
                "use_mla": False,
                "tp_size": 1,
                "dp_size": 1,
                "lora_name": "",
                "pp_size": 1,
                "extra": "",
                "user_data": "",
            },
            "location_spec_groups": [{"name": "default", "spec_names": ["default"]}],
        }

    def _node_register_event(self) -> dict[str, object]:
        return {
            "event_type": KvcmReportEventType.NODE_REGISTER,
            "node_register": {"mediums": ["hbm", "mem"]},
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
            "storage_type": self._storage_type(),
        }

    def _map_medium(self, medium: str | None) -> str:
        if medium is None:
            return ""
        return _MEDIUM_MAP.get(medium, "")

    def _report_events_for_batches(
        self, batches: list[KVEventBatch]
    ) -> list[dict[str, object]]:
        events: list[dict[str, object]] = []
        for batch in batches:
            for event in batch.events:
                if isinstance(event, BlockStored):
                    medium = self._map_medium(event.medium)
                    for block_hash in event.block_hashes:
                        events.append(
                            {
                                "event_type": KvcmReportEventType.BLOCK_ADD,
                                "block_add": {
                                    "block_key": str(block_hash),
                                    "uri": "",
                                    "medium": medium,
                                    "specs": [],
                                },
                            }
                        )
                elif isinstance(event, BlockRemoved):
                    medium = self._map_medium(event.medium)
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
        self._host_ip_port_value = await asyncio.to_thread(
            resolve_host_ip_port,
            self._config.kvcm_host_ip_port,
            self._config.engine_health_url,
        )
        self._sdk_client = self._create_sdk_client()
        await asyncio.to_thread(
            self._sdk_client.register_instance,
            self._register_instance_request(),
        )
        await self._report_events([self._node_register_event()])
        self._heartbeat_task = asyncio.create_task(self._heartbeat_loop())

    async def _heartbeat_loop(self) -> None:
        while True:
            await asyncio.sleep(self._config.kvcm_heartbeat_interval_s)
            try:
                await self._report_events([self._heartbeat_event()])
            except Exception:
                logger.warning(
                    "kvcm heartbeat report failed",
                    step="kvcm_heartbeat",
                    exc_info=True,
                )

    async def _report_events(self, events: list[dict[str, object]]) -> dict[str, Any]:
        if self._sdk_client is None:
            raise RuntimeError("kvcm client has not been started")
        sdk_client = self._sdk_client

        def _build_and_report() -> Any:
            return sdk_client.report_event(self._report_event_request(events))

        return await asyncio.to_thread(_build_and_report)

    async def send_batch(self, batches: list[KVEventBatch], epoch: int) -> None:
        if self._sdk_client is None:
            raise RuntimeError("kvcm client has not been started")
        sdk_client = self._sdk_client

        def _build_and_report() -> Any | None:
            events = self._report_events_for_batches(batches)
            if not events:
                return None
            return sdk_client.report_event(self._report_event_request(events))

        response = await asyncio.to_thread(_build_and_report)
        if response is None:
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
        if self._sdk_client is not None:
            await asyncio.to_thread(self._sdk_client.close)
            self._sdk_client = None
