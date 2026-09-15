"""vLLM decoder for the shared DashLLM gRPC snapshot source."""

from __future__ import annotations

import msgspec

from subscriber.config import SubscriberConfig
from subscriber.engine.kv_event_control_client import DashllmKvEventControlClient
from subscriber.engine.snapshot import GrpcSnapshotSource
from subscriber.types import BlockSnapshotItem


class VllmSnapshotSource(GrpcSnapshotSource):
    """Poll vLLM's group-indexed HBM snapshot over the shared lifecycle."""

    def __init__(
        self,
        config: SubscriberConfig,
        grpc_client: DashllmKvEventControlClient,
    ) -> None:
        super().__init__(config, grpc_client, self._decode_vllm_snapshot)

    @staticmethod
    def _decode_vllm_snapshot(raw_snapshot: bytes) -> list[BlockSnapshotItem]:
        items = msgspec.msgpack.decode(
            raw_snapshot,
            type=list[tuple[bytes | int, int, int, int]],
        )
        return [
            BlockSnapshotItem(block_hash=block_hash, group_idx=group_idx)
            for block_hash, group_idx, _, _ in items
        ]
