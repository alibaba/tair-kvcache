"""SGLang decoder for the shared DashLLM gRPC snapshot source."""

from __future__ import annotations

import msgspec

from subscriber.config import SubscriberConfig
from subscriber.engine.kv_event_control_client import DashllmKvEventControlClient
from subscriber.engine.snapshot import GrpcSnapshotSource, SnapshotSchemaError
from subscriber.types import BlockSnapshotItem


class SglangSnapshotSource(GrpcSnapshotSource):
    """Poll SGLang's native-hash, component-indexed HBM snapshot."""

    def __init__(
        self,
        config: SubscriberConfig,
        grpc_client: DashllmKvEventControlClient,
        *,
        component_group_idxs: dict[int, int],
    ) -> None:
        self._component_group_idxs = dict(component_group_idxs)
        super().__init__(config, grpc_client, self._decode_sglang_snapshot)

    def _decode_sglang_snapshot(self, raw_snapshot: bytes) -> list[BlockSnapshotItem]:
        items = msgspec.msgpack.decode(
            raw_snapshot,
            type=list[tuple[int, int]],
        )
        decoded = []
        for block_hash, component_id in items:
            try:
                group_idx = self._component_group_idxs[component_id]
            except KeyError as exc:
                raise SnapshotSchemaError(
                    f"snapshot contains undeclared component_id={component_id}"
                ) from exc
            decoded.append(
                BlockSnapshotItem(block_hash=block_hash, group_idx=group_idx)
            )
        return decoded
