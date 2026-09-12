from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import ANY, AsyncMock, MagicMock

import grpc
import msgspec.msgpack
import pytest

from subscriber.config import SubscriberConfig
from subscriber.engine import snapshot as snapshot_module
from subscriber.engine.base import EngineEventBatch
from subscriber.engine.vllm.snapshot import VllmSnapshotSource
from subscriber.metrics import BatchTelemetry
from subscriber.proto import engine_service_rpc_pb2
from subscriber.types import BlockSnapshot, BlockSnapshotItem


@pytest.fixture
def config() -> SubscriberConfig:
    return SubscriberConfig()


def _snapshot_response(
    *,
    block_hashes: list[bytes] | None = None,
    block_hashes_uint64: list[int] | None = None,
    block_size: int = 16,
    snapshot_version: int = 0,
) -> engine_service_rpc_pb2.KvCacheBlockListPB:
    items: list[tuple] = []
    idx = 0
    for block_hash in block_hashes or []:
        items.append((block_hash, idx, 0, 0))
        idx += 1
    for block_hash in block_hashes_uint64 or []:
        items.append((block_hash, idx, 0, 0))
        idx += 1
    return engine_service_rpc_pb2.KvCacheBlockListPB(
        raw_snapshot=msgspec.msgpack.encode(items),
        block_size=block_size,
        snapshot_version=snapshot_version,
    )


def _make_grpc_error() -> grpc.aio.AioRpcError:
    return grpc.aio.AioRpcError(
        code=grpc.StatusCode.UNAVAILABLE,
        initial_metadata=None,
        trailing_metadata=None,
        details="engine down",
        debug_error_string=None,
    )


async def _collect_events(source: VllmSnapshotSource, n: int) -> list[EngineEventBatch]:
    results: list[EngineEventBatch] = []

    async def _run() -> None:
        async for event in source.subscribe():
            results.append(event)
            if len(results) >= n:
                break

    await asyncio.wait_for(_run(), timeout=1.0)
    return results


async def test_subscribe_yields_snapshot_with_version(
    config: SubscriberConfig, mocker: Any
) -> None:
    grpc_client = MagicMock()
    grpc_client.get_all_kv_cache_blocks = AsyncMock(
        return_value=_snapshot_response(
            block_hashes=[b"\x01", b"\x02"], snapshot_version=7
        )
    )
    mocker.patch.object(VllmSnapshotSource, "_wait_interval", AsyncMock())
    source = VllmSnapshotSource(config, grpc_client)

    events = await _collect_events(source, 1)

    grpc_client.get_all_kv_cache_blocks.assert_awaited_once_with(
        timeout_s=config.engine_kvcache_snapshot_timeout_ms / 1000,
    )
    assert len(events) == 1
    batches = events[0].batches
    assert len(batches) == 1
    assert batches[0].events == [
        BlockSnapshot(
            medium="GPU",
            block_size=16,
            items=[
                BlockSnapshotItem(block_hash=b"\x01", group_idx=0),
                BlockSnapshotItem(block_hash=b"\x02", group_idx=1),
            ],
            snapshot_version=7,
        )
    ]
    spans = events[0].telemetry.spans
    assert [span.name for span in spans] == [
        "snapshot_fetch",
        "decode",
        "snapshot_build",
    ]
    assert all(span.duration_s >= 0 for span in spans)
    assert events[0].telemetry.drop_reason is None


async def test_subscribe_yields_uint64_block_hashes(
    config: SubscriberConfig, mocker: Any
) -> None:
    grpc_client = MagicMock()
    grpc_client.get_all_kv_cache_blocks = AsyncMock(
        return_value=_snapshot_response(
            block_hashes_uint64=[10400093841463714284, 2], snapshot_version=3
        )
    )
    mocker.patch.object(VllmSnapshotSource, "_wait_interval", AsyncMock())
    source = VllmSnapshotSource(config, grpc_client)

    events = await _collect_events(source, 1)

    assert events[0].batches[0].events == [
        BlockSnapshot(
            medium="GPU",
            block_size=16,
            items=[
                BlockSnapshotItem(block_hash=10400093841463714284, group_idx=0),
                BlockSnapshotItem(block_hash=2, group_idx=1),
            ],
            snapshot_version=3,
        )
    ]


async def test_success_records_latency_payload_and_fetch_count(
    config: SubscriberConfig, mocker: Any
) -> None:
    response = _snapshot_response(block_hashes=[b"\x01", b"\x02"])
    grpc_client = MagicMock()
    grpc_client.get_all_kv_cache_blocks = AsyncMock(return_value=response)
    mocker.patch.object(VllmSnapshotSource, "_wait_interval", AsyncMock())
    source = VllmSnapshotSource(config, grpc_client)

    events = await _collect_events(source, 1)

    telemetry = events[0].telemetry
    assert telemetry.counters == {}
    assert telemetry.gauges["full_snapshot_payload_bytes"] == response.ByteSize()


async def test_debug_log_reports_snapshot_fetch_span_not_build_span(
    config: SubscriberConfig, mocker: Any
) -> None:
    grpc_client = MagicMock()
    grpc_client.get_all_kv_cache_blocks = AsyncMock(
        return_value=_snapshot_response(block_hashes=[b"\x01"])
    )
    telemetry = BatchTelemetry(
        pipeline="snapshot",
        clock=MagicMock(side_effect=[0.0, 0.05, 0.1, 0.2]),
    )
    mocker.patch.object(snapshot_module, "BatchTelemetry", return_value=telemetry)
    mocker.patch.object(VllmSnapshotSource, "_wait_interval", AsyncMock())
    mocker.patch(
        "subscriber.engine.snapshot.logger.is_debug_enabled", return_value=True
    )
    debug = mocker.patch("subscriber.engine.snapshot.logger.debug")
    source = VllmSnapshotSource(config, grpc_client)

    await _collect_events(source, 1)

    debug.assert_called_once_with(
        "snapshot poll completed",
        step="grpc_snapshot",
        tags={
            "block_count": 1,
            "snapshot_version": 0,
            "payload_bytes": _snapshot_response(block_hashes=[b"\x01"]).ByteSize(),
            "fetch_ms": 50.0,
            "trace_id": ANY,
        },
    )


async def test_subscribe_retries_on_grpc_error(
    config: SubscriberConfig, mocker: Any
) -> None:
    grpc_client = MagicMock()
    grpc_client.get_all_kv_cache_blocks = AsyncMock(
        side_effect=[
            _make_grpc_error(),
            _snapshot_response(block_hashes=[b"\x09"]),
        ]
    )
    warning = mocker.patch("subscriber.engine.snapshot.logger.warning")
    wait_mock = mocker.patch.object(VllmSnapshotSource, "_wait_interval", AsyncMock())
    source = VllmSnapshotSource(config, grpc_client)

    # Two events are now yielded: one drop-marker for the failed poll and
    # one successful snapshot from the retry.
    events = await _collect_events(source, 2)

    assert grpc_client.get_all_kv_cache_blocks.await_count == 2
    warning.assert_called_once()
    assert warning.call_args.kwargs["tags"]["code"] == "UNAVAILABLE"

    failed = events[0].telemetry
    assert events[0].batches == []
    assert failed.drop_reason == "fetch_failed"
    assert failed.counters == {}
    assert [s.name for s in failed.spans] == ["snapshot_fetch"]
    assert "full_snapshot_payload_bytes" not in failed.gauges

    success = events[1].telemetry
    assert success.drop_reason is None
    assert success.counters == {}
    assert success.gauges["full_snapshot_payload_bytes"] == (
        _snapshot_response(block_hashes=[b"\x09"]).ByteSize()
    )

    # The failed poll must wait one full interval before retrying.
    wait_mock.assert_awaited_with(config.engine_snapshot_full_sync_interval_s)


async def test_reset_discards_inflight_result(
    config: SubscriberConfig, mocker: Any
) -> None:
    grpc_client = MagicMock()
    poll_started = asyncio.Event()
    release_poll = asyncio.Event()

    async def slow_poll(timeout_s: float) -> object:
        poll_started.set()
        await release_poll.wait()
        return _snapshot_response(block_hashes=[b"\xaa"])

    grpc_client.get_all_kv_cache_blocks = AsyncMock(side_effect=slow_poll)
    mocker.patch.object(VllmSnapshotSource, "_wait_interval", AsyncMock())
    source = VllmSnapshotSource(config, grpc_client)

    gen = source.subscribe()
    discarded_task = asyncio.create_task(gen.__anext__())
    await poll_started.wait()

    await source.reset_generation_state()
    release_poll.set()

    # First yield is the discarded stale result (drop_reason set, no batches).
    discarded_event = await asyncio.wait_for(discarded_task, timeout=1.0)
    assert discarded_event.batches == []
    assert discarded_event.telemetry.drop_reason == "generation_reset"

    # After the discard the source polls again and yields the fresh snapshot.
    grpc_client.get_all_kv_cache_blocks = AsyncMock(
        return_value=_snapshot_response(block_hashes=[b"\xbb"])
    )
    fresh_event = await asyncio.wait_for(gen.__anext__(), timeout=1.0)
    assert fresh_event.batches[0].events == [
        BlockSnapshot(
            medium="GPU",
            block_size=16,
            items=[BlockSnapshotItem(block_hash=b"\xbb", group_idx=0)],
            snapshot_version=0,
        )
    ]
    assert fresh_event.telemetry.drop_reason is None
    await gen.aclose()


async def test_stale_response_marks_generation_reset_drop(
    config: SubscriberConfig, mocker: Any
) -> None:
    grpc_client = MagicMock()
    poll_started = asyncio.Event()
    release_poll = asyncio.Event()

    async def slow_poll(timeout_s: float) -> object:
        poll_started.set()
        await release_poll.wait()
        return _snapshot_response(block_hashes=[b"\xaa"])

    grpc_client.get_all_kv_cache_blocks = AsyncMock(side_effect=slow_poll)
    mocker.patch.object(VllmSnapshotSource, "_wait_interval", AsyncMock())
    source = VllmSnapshotSource(config, grpc_client)

    gen = source.subscribe()
    discarded_task = asyncio.create_task(gen.__anext__())
    await poll_started.wait()

    await source.reset_generation_state()
    release_poll.set()

    discarded = await asyncio.wait_for(discarded_task, timeout=1.0)
    grpc_client.get_all_kv_cache_blocks = AsyncMock(
        return_value=_snapshot_response(block_hashes=[b"\xbb"])
    )
    fresh = await asyncio.wait_for(gen.__anext__(), timeout=1.0)
    await gen.aclose()

    assert discarded.telemetry.drop_reason == "generation_reset"
    # Discarded telemetry still records fetch_count, latency, and payload
    # bytes: the response arrived (so its size is a real measurement), the
    # subscriber is only choosing to discard the *content*. Reporting the
    # size lets dashboards distinguish an empty snapshot from a missing
    # observation.
    assert discarded.telemetry.counters == {}
    assert [s.name for s in discarded.telemetry.spans] == ["snapshot_fetch"]
    assert (
        discarded.telemetry.gauges["full_snapshot_payload_bytes"]
        == _snapshot_response(block_hashes=[b"\xaa"]).ByteSize()
    )
    assert fresh.telemetry.drop_reason is None


async def test_fixed_interval_from_config(
    config: SubscriberConfig, mocker: Any
) -> None:
    config.engine_snapshot_full_sync_interval_s = 42.0
    grpc_client = MagicMock()
    grpc_client.get_all_kv_cache_blocks = AsyncMock(
        return_value=_snapshot_response(block_hashes=[b"\x01"])
    )
    wait_mock = mocker.patch.object(VllmSnapshotSource, "_wait_interval", AsyncMock())
    source = VllmSnapshotSource(config, grpc_client)

    await _collect_events(source, 3)

    # The collector breaks on the third yield before its trailing wait, so two
    # waits are observed. Every poll must wait the configured fixed interval
    # (no dynamic backoff).
    assert wait_mock.await_count == 2
    for call in wait_mock.await_args_list:
        assert call.args == (42.0,)


async def test_subscribe_retries_on_unexpected_error(
    config: SubscriberConfig, mocker: Any
) -> None:
    grpc_client = MagicMock()
    grpc_client.get_all_kv_cache_blocks = AsyncMock(
        side_effect=[
            ConnectionResetError("peer reset"),
            _snapshot_response(block_hashes=[b"\x01"]),
        ]
    )
    warning = mocker.patch("subscriber.engine.snapshot.logger.warning")
    wait_mock = mocker.patch.object(VllmSnapshotSource, "_wait_interval", AsyncMock())
    source = VllmSnapshotSource(config, grpc_client)

    events = await _collect_events(source, 2)

    assert grpc_client.get_all_kv_cache_blocks.await_count == 2
    warning.assert_called_once()
    assert warning.call_args.kwargs["tags"]["error"] == "ConnectionResetError"
    assert events[0].batches == []
    assert events[0].telemetry.drop_reason == "fetch_failed"
    assert events[1].telemetry.drop_reason is None
    wait_mock.assert_awaited_with(config.engine_snapshot_full_sync_interval_s)


async def test_sleeps_interval_after_successful_poll(
    config: SubscriberConfig, mocker: Any
) -> None:
    config.engine_snapshot_full_sync_interval_s = 10.0
    grpc_client = MagicMock()
    grpc_client.get_all_kv_cache_blocks = AsyncMock(
        return_value=_snapshot_response(block_hashes=[b"\x01"])
    )
    wait_mock = mocker.patch.object(VllmSnapshotSource, "_wait_interval", AsyncMock())
    source = VllmSnapshotSource(config, grpc_client)

    await _collect_events(source, 2)

    wait_mock.assert_awaited_once_with(10.0)
