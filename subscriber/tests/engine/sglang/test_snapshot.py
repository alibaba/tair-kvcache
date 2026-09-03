from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import msgspec
import pytest

from subscriber.config import SubscriberConfig
from subscriber.engine.base import EngineEventBatch
from subscriber.engine.sglang.snapshot import SglangSnapshotSource
from subscriber.proto import engine_service_rpc_pb2
from subscriber.types import BlockSnapshot, BlockSnapshotItem


def _response(
    items: object,
    *,
    snapshot_version: int = 3,
) -> engine_service_rpc_pb2.KvCacheBlockListPB:
    return engine_service_rpc_pb2.KvCacheBlockListPB(
        raw_snapshot=msgspec.msgpack.encode(items),
        snapshot_version=snapshot_version,
    )


async def _next(source: SglangSnapshotSource) -> EngineEventBatch:
    return await asyncio.wait_for(anext(source.subscribe()), timeout=1.0)


@pytest.mark.parametrize(
    "items",
    [
        [(1,)],
        [(1, 0, 2)],
        [(b"not-an-int", 0)],
        [(1, "full")],
    ],
)
async def test_rejects_malformed_engine_specific_schema(
    items: object,
    mocker: Any,
) -> None:
    client = MagicMock()
    client.get_all_kv_cache_blocks = AsyncMock(return_value=_response(items))
    source = SglangSnapshotSource(
        SubscriberConfig(), client, component_group_idxs={0: 0, 2: 2}
    )
    mocker.patch.object(source, "_wait_interval", AsyncMock())

    event = await _next(source)

    assert event.batches == []
    assert event.telemetry.drop_reason == "schema_mismatch"


async def test_decodes_signed_hashes_and_mixed_component_ids(
    mocker: Any,
) -> None:
    client = MagicMock()
    response = _response([(-11, 0), (22, 2)], snapshot_version=9)
    client.get_all_kv_cache_blocks = AsyncMock(return_value=response)
    source = SglangSnapshotSource(
        SubscriberConfig(), client, component_group_idxs={0: 0, 2: 2}
    )
    mocker.patch.object(source, "_wait_interval", AsyncMock())

    event = await _next(source)

    assert event.batches[0].events == [
        BlockSnapshot(
            medium="GPU",
            block_size=0,
            items=[
                BlockSnapshotItem(block_hash=-11, group_idx=0),
                BlockSnapshotItem(block_hash=22, group_idx=2),
            ],
            snapshot_version=9,
        )
    ]


async def test_rejects_component_not_declared_by_bootstrap(mocker: Any) -> None:
    client = MagicMock()
    client.get_all_kv_cache_blocks = AsyncMock(return_value=_response([(11, 1)]))
    source = SglangSnapshotSource(
        SubscriberConfig(), client, component_group_idxs={0: 0, 2: 2}
    )
    mocker.patch.object(source, "_wait_interval", AsyncMock())

    event = await _next(source)

    assert event.batches == []
    assert event.telemetry.drop_reason == "schema_mismatch"


async def test_msgpack_empty_list_is_authoritative_empty_snapshot(
    mocker: Any,
) -> None:
    client = MagicMock()
    client.get_all_kv_cache_blocks = AsyncMock(return_value=_response([]))
    source = SglangSnapshotSource(
        SubscriberConfig(), client, component_group_idxs={0: 0, 2: 2}
    )
    mocker.patch.object(source, "_wait_interval", AsyncMock())

    event = await _next(source)

    assert event.telemetry.drop_reason is None
    assert event.batches[0].events[0].items == []
