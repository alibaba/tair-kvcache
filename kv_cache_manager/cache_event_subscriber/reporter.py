from __future__ import annotations

import itertools
import threading
import time
from collections.abc import Callable, Iterable
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from kv_cache_manager.py_connector.common.manager_client import KvCacheManagerClient

from .models import BlockRecord, EngineUpdate, LocationSpec


class KvcmEventReporter:
    """Translate normalized engine state into acknowledged KVCM mutations."""

    def __init__(
        self,
        manager_client: KvCacheManagerClient,
        *,
        instance_id: str,
        host_ip_port: str,
        spec_factory: Callable[[BlockRecord], Iterable[LocationSpec]],
        storage_type: str = "ST_EVENT_REPORT",
        max_events_per_request: int = 4096,
    ) -> None:
        if not instance_id or not host_ip_port:
            raise ValueError("instance_id and host_ip_port are required")
        if max_events_per_request <= 0:
            raise ValueError("max_events_per_request must be positive")
        self._client = manager_client
        self._instance_id = instance_id
        self._host_ip_port = host_ip_port
        self._spec_factory = spec_factory
        self._storage_type = storage_type
        self._max_events = max_events_per_request
        self._request_lock = threading.Lock()

    def _trace_id(self, operation: str) -> str:
        return f"cache_subscriber_{operation}_{time.monotonic_ns()}"

    def _request(self, events: list[dict[str, Any]]) -> dict[str, Any]:
        return {
            "trace_id": self._trace_id("report"),
            "instance_id": self._instance_id,
            "host_ip_port": self._host_ip_port,
            "storage_type": self._storage_type,
            "events": events,
        }

    def register_node(self, mediums: Iterable[str]) -> None:
        event = {
            "event_type": "EVENT_NODE_REGISTER",
            "node_register": {"mediums": sorted(set(mediums))},
        }
        self._report([event])

    def register_instance(
        self,
        *,
        instance_group: str,
        block_size: int,
        spec_sizes: dict[str, int],
        model_name: str,
        dtype: str,
        tp_size: int = 1,
        dp_size: int = 1,
        pp_size: int = 1,
        use_mla: bool = False,
        location_spec_groups: list[dict[str, Any]] | None = None,
    ) -> None:
        request = {
            "trace_id": self._trace_id("register_instance"),
            "instance_group": instance_group,
            "instance_id": self._instance_id,
            "block_size": block_size,
            "location_spec_infos": [
                {"name": name, "size": size} for name, size in sorted(spec_sizes.items())
            ],
            "model_deployment": {
                "model_name": model_name,
                "dtype": dtype,
                "use_mla": use_mla,
                "tp_size": tp_size,
                "dp_size": dp_size,
                "pp_size": pp_size,
                "lora_name": "",
                "extra": "",
                "user_data": "",
            },
            "location_spec_groups": location_spec_groups or [],
        }
        with self._request_lock:
            self._client.register_instance(request)

    def heartbeat(self, system_status: dict[str, str] | None = None) -> None:
        event = {
            "event_type": "EVENT_HEARTBEAT",
            "heartbeat": {"system_status": system_status or {}},
        }
        self._report([event])

    def host_down(self) -> None:
        event = {"event_type": "EVENT_HOST_DOWN", "host_down": {}}
        self._report([event])

    def send(self, update: EngineUpdate) -> None:
        if update.full_snapshot:
            if len(update.blocks) > self._max_events:
                # A single authoritative event is preferable for small
                # snapshots. Large snapshots use clear-then-upsert chunks to
                # stay below HTTP/protobuf limits. Retrying the same update
                # starts with another clear and therefore always converges.
                self._report(
                    [
                        {
                            "event_type": "EVENT_BLOCK_SNAPSHOT",
                            "block_snapshot": {"blocks": []},
                        }
                    ]
                )
                events = [self._upsert_event(block) for block in update.blocks]
                for chunk in _chunks(events, self._max_events):
                    self._report(chunk)
                return
            blocks = [self._snapshot_block(block) for block in update.blocks]
            event = {
                "event_type": "EVENT_BLOCK_SNAPSHOT",
                "block_snapshot": {"blocks": blocks},
            }
            self._report([event])
            return

        events = [self._upsert_event(block) for block in update.upserts]
        events.extend(self._remove_event(block) for block in update.removals)
        for chunk in _chunks(events, self._max_events):
            self._report(chunk)

    def _report(self, events: list[dict[str, Any]]) -> None:
        with self._request_lock:
            self._client.report_event(self._request(events))

    def _specs(self, block: BlockRecord) -> list[dict[str, str]]:
        specs = [{"name": spec.name, "uri": spec.uri} for spec in self._spec_factory(block)]
        if not specs:
            raise ValueError(f"block {block.block_key} has no location specs")
        return specs

    def _snapshot_block(self, block: BlockRecord) -> dict[str, Any]:
        return {
            "block_key": str(block.block_key),
            "medium": block.medium,
            "specs": self._specs(block),
        }

    def _upsert_event(self, block: BlockRecord) -> dict[str, Any]:
        return {
            "event_type": "EVENT_BLOCK_ADD",
            "block_add": self._snapshot_block(block),
        }

    @staticmethod
    def _remove_event(block: BlockRecord) -> dict[str, Any]:
        return {
            "event_type": "EVENT_BLOCK_DELETE",
            "block_delete": {
                "block_key": str(block.block_key),
                "medium": block.medium,
            },
        }


def _chunks(items: list[dict[str, Any]], size: int) -> Iterable[list[dict[str, Any]]]:
    iterator = iter(items)
    while chunk := list(itertools.islice(iterator, size)):
        yield chunk
