from __future__ import annotations

import asyncio
from typing import Any

from subscriber.config import SubscriberConfig
from subscriber.engine.rtp_llm import (
    CacheDiffTracker,
    CacheSnapshot,
    CacheStatusPB,
    CacheVersionPB,
    RtpGrpcCacheStatusSource,
    RtpLlmAdapter,
)
from subscriber.health.events import LivenessEvent
from subscriber.metrics import StageTimer
from subscriber.types import AllBlocksCleared, BlockRemoved, BlockStored


class FakeSource:
    async def fetch_snapshot(self) -> CacheSnapshot:
        raise AssertionError("unexpected snapshot fetch")

    async def close(self) -> None:
        return None


def _config(**overrides: Any) -> SubscriberConfig:
    values: dict[str, Any] = {
        "engine_type": "rtp_llm",
        "rtp_endpoints": "127.0.0.1:8089",
    }
    values.update(overrides)
    config = SubscriberConfig(**values)
    config.validate()
    return config


def test_cache_status_proto_subset_round_trips() -> None:
    request = CacheVersionPB(latest_cache_version=-1, need_cache_keys=True)
    decoded_request = CacheVersionPB.FromString(request.SerializeToString())

    response = CacheStatusPB(block_size=64, version=7)
    response.cache_keys[11] = True
    response.cache_keys[12] = False
    decoded_response = CacheStatusPB.FromString(response.SerializeToString())

    assert decoded_request.latest_cache_version == -1
    assert decoded_request.need_cache_keys is True
    assert decoded_response.block_size == 64
    assert decoded_response.version == 7
    assert dict(decoded_response.cache_keys) == {11: True, 12: False}


def test_grpc_source_serializes_concrete_dynamic_request(mocker) -> None:
    channel = mocker.Mock()
    mocker.patch(
        "subscriber.engine.rtp_llm.grpc.aio.insecure_channel",
        return_value=channel,
    )
    source = RtpGrpcCacheStatusSource(("rank-0:1",), 1.0)

    source._ensure_connections()

    serializer = channel.unary_unary.call_args.kwargs["request_serializer"]
    encoded = serializer(CacheVersionPB(latest_cache_version=-1, need_cache_keys=True))
    decoded = CacheVersionPB.FromString(encoded)
    assert decoded.latest_cache_version == -1
    assert decoded.need_cache_keys is True


async def test_grpc_source_merges_only_present_keys_from_all_endpoints() -> None:
    first = CacheStatusPB(block_size=64, version=9)
    first.cache_keys[1] = True
    first.cache_keys[2] = False
    second = CacheStatusPB(block_size=64, version=8)
    second.cache_keys[2] = True
    second.cache_keys[3] = True
    requests: list[tuple[int, bool, float]] = []

    def call_for(response: Any):
        async def call(request: Any, *, timeout: float) -> Any:
            requests.append(
                (
                    int(request.latest_cache_version),
                    bool(request.need_cache_keys),
                    timeout,
                )
            )
            return response

        return call

    source = RtpGrpcCacheStatusSource(("rank-0:1", "rank-1:2"), 2.5)
    source._calls = [call_for(first), call_for(second)]

    snapshot = await source.fetch_snapshot()

    assert snapshot == CacheSnapshot(frozenset({1, 2, 3}), 64, 8)
    assert requests == [(-1, True, 2.5), (-1, True, 2.5)]


async def test_grpc_source_waits_for_every_endpoint_before_rejecting_poll() -> None:
    release_second_endpoint = asyncio.Event()

    async def failed_call(_request: Any, *, timeout: float) -> Any:
        assert timeout == 1.0
        raise RuntimeError("rank unavailable")

    async def pending_call(_request: Any, *, timeout: float) -> Any:
        assert timeout == 1.0
        await release_second_endpoint.wait()
        return CacheStatusPB(block_size=64, version=1)

    source = RtpGrpcCacheStatusSource(("rank-0:1", "rank-1:2"), 1.0)
    source._calls = [failed_call, pending_call]

    fetch = asyncio.create_task(source.fetch_snapshot())
    await asyncio.sleep(0)
    assert not fetch.done()

    release_second_endpoint.set()
    try:
        await fetch
    except RuntimeError as exc:
        assert "rank-0:1" in str(exc)
        assert "rank unavailable" in str(exc)
    else:
        raise AssertionError("expected one failed DP endpoint to reject the poll")


async def test_grpc_source_rejects_uninitialized_cache_block_size() -> None:
    response = CacheStatusPB(block_size=0, version=-1)

    async def call(_request: Any, *, timeout: float) -> Any:
        assert timeout == 1.0
        return response

    source = RtpGrpcCacheStatusSource(("rank-0:1",), 1.0)
    source._calls = [call]

    try:
        await source.fetch_snapshot()
    except RuntimeError as exc:
        assert "invalid cache block size" in str(exc)
    else:
        raise AssertionError("expected invalid block size to reject the snapshot")


def test_diff_tracker_commits_only_after_ack() -> None:
    tracker = CacheDiffTracker(deletion_confirmations=2)
    diff = tracker.plan(frozenset({1, 2}))

    assert diff.added == (1, 2)
    assert tracker.acknowledged_keys == frozenset()

    tracker.commit(diff)

    assert tracker.acknowledged_keys == frozenset({1, 2})
    assert tracker.plan(frozenset({2})).removed == ()
    assert tracker.plan(frozenset({2})).removed == (1,)


async def test_initial_snapshot_resets_then_reports_all_blocks() -> None:
    adapter = RtpLlmAdapter(_config(block_size=64), source=FakeSource())
    snapshot = CacheSnapshot(frozenset({7, 9}), 64, 1)

    event = adapter._event_for_snapshot(snapshot, StageTimer())

    assert event is not None
    assert isinstance(event.batches[0].events[0], AllBlocksCleared)
    stored = event.batches[0].events[1]
    assert isinstance(stored, BlockStored)
    assert stored.block_hashes == [7, 9]
    assert event.on_delivery is not None

    await event.on_delivery(True)

    assert adapter.tracker.acknowledged_keys == frozenset({7, 9})


async def test_failed_delivery_retries_the_same_adds() -> None:
    adapter = RtpLlmAdapter(
        _config(rtp_reset_on_start=False, block_size=32),
        source=FakeSource(),
    )
    snapshot = CacheSnapshot(frozenset({5}), 32, 1)
    first = adapter._event_for_snapshot(snapshot, StageTimer())
    assert first is not None
    assert first.on_delivery is not None

    await first.on_delivery(False)
    retry = adapter._event_for_snapshot(snapshot, StageTimer())

    assert retry is not None
    stored = retry.batches[0].events[0]
    assert isinstance(stored, BlockStored)
    assert stored.block_hashes == [5]


async def test_removal_requires_configured_number_of_snapshots() -> None:
    adapter = RtpLlmAdapter(
        _config(
            rtp_reset_on_start=False,
            rtp_deletion_confirmations=2,
            block_size=16,
        ),
        source=FakeSource(),
    )
    initial = adapter._event_for_snapshot(
        CacheSnapshot(frozenset({1}), 16, 1),
        StageTimer(),
    )
    assert initial is not None
    assert initial.on_delivery is not None
    await initial.on_delivery(True)

    assert (
        adapter._event_for_snapshot(CacheSnapshot(frozenset(), 16, 2), StageTimer())
        is None
    )
    removed_event = adapter._event_for_snapshot(
        CacheSnapshot(frozenset(), 16, 3),
        StageTimer(),
    )

    assert removed_event is not None
    removed = removed_event.batches[0].events[0]
    assert isinstance(removed, BlockRemoved)
    assert removed.block_hashes == [1]


def test_snapshot_block_size_must_match_kvcm_registration() -> None:
    adapter = RtpLlmAdapter(_config(block_size=64), source=FakeSource())

    try:
        adapter._event_for_snapshot(
            CacheSnapshot(frozenset({1}), 32, 1),
            StageTimer(),
        )
    except RuntimeError as exc:
        assert "does not match KVCM registration" in str(exc)
    else:
        raise AssertionError("expected block-size mismatch to reject the snapshot")


async def test_invalid_snapshot_metadata_never_reports_engine_healthy() -> None:
    class InvalidSnapshotSource:
        async def fetch_snapshot(self) -> CacheSnapshot:
            return CacheSnapshot(frozenset({1}), 32, 1)

        async def close(self) -> None:
            return None

    adapter = RtpLlmAdapter(
        _config(block_size=64, rtp_poll_interval_s=0.001),
        source=InvalidSnapshotSource(),
    )
    events = adapter.watch_liveness()
    try:
        first = await asyncio.wait_for(events.__anext__(), timeout=1)
        second = await asyncio.wait_for(events.__anext__(), timeout=1)
    finally:
        await events.aclose()

    assert first is LivenessEvent.UNHEALTHY
    assert second is LivenessEvent.UNHEALTHY


def test_rtp_kvcm_location_metadata() -> None:
    adapter = RtpLlmAdapter(_config(), source=FakeSource())

    assert adapter.map_medium("hbm") == "hbm"
    assert adapter.supported_mediums() == ["hbm"]
    assert adapter.storage_type() == "ST_EVENT_REPORT"
    assert adapter.location_spec_name(64) == "rtp_llm_64"
    assert adapter.location_uri("10.0.0.1:8088", "hbm") == (
        "rtp-llm://10.0.0.1:8088/hbm"
    )
