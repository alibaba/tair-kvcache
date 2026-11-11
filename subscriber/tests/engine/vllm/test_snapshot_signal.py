"""Tests for VllmSnapshotSource.request_immediate_snapshot signal mechanism."""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import msgspec.msgpack
import pytest

from subscriber.config import SubscriberConfig
from subscriber.engine import snapshot as snapshot_module
from subscriber.engine.vllm.snapshot import VllmSnapshotSource
from subscriber.proto import engine_service_rpc_pb2


@pytest.fixture
def config() -> SubscriberConfig:
    cfg = SubscriberConfig()
    cfg.engine_snapshot_full_sync_interval_s = 30.0
    return cfg


def _snapshot_response(
    *, block_hashes: list[bytes] | None = None
) -> engine_service_rpc_pb2.KvCacheBlockListPB:
    items: list[tuple] = []
    for idx, bh in enumerate(block_hashes or []):
        items.append((bh, idx, 0, 0))
    return engine_service_rpc_pb2.KvCacheBlockListPB(
        raw_snapshot=msgspec.msgpack.encode(items),
        block_size=16,
        snapshot_version=1,
    )


def _make_source(
    config: SubscriberConfig,
    responses: list[Any] | None = None,
) -> tuple[VllmSnapshotSource, MagicMock]:
    grpc_client = MagicMock()
    if responses is not None:
        grpc_client.get_all_kv_cache_blocks = AsyncMock(side_effect=responses)
    else:
        grpc_client.get_all_kv_cache_blocks = AsyncMock(
            return_value=_snapshot_response(block_hashes=[b"\x01"])
        )
    source = VllmSnapshotSource(config, grpc_client)
    return source, grpc_client


async def test_signal_wakes_poll_before_interval(config: SubscriberConfig) -> None:
    """request_immediate_snapshot wakes the source without waiting full interval."""
    source, grpc_client = _make_source(config)

    gen = source.subscribe()
    first_task = asyncio.create_task(gen.__anext__())
    first_event = await asyncio.wait_for(first_task, timeout=1.0)
    assert first_event.telemetry.drop_reason is None

    # Source is now in _wait_interval. Signal it.
    source.request_immediate_snapshot()

    # The next poll should complete quickly (not after 30s).
    second_task = asyncio.create_task(gen.__anext__())
    second_event = await asyncio.wait_for(second_task, timeout=1.0)
    assert second_event.telemetry.drop_reason is None
    assert grpc_client.get_all_kv_cache_blocks.await_count == 2
    await gen.aclose()


async def test_coalesce_multiple_signals_into_one_poll(
    config: SubscriberConfig,
) -> None:
    """Multiple signals before consumption yield only one extra poll."""
    source, grpc_client = _make_source(config)

    gen = source.subscribe()
    first = await asyncio.wait_for(gen.__anext__(), timeout=1.0)
    assert first.telemetry.drop_reason is None

    # Signal multiple times while source is waiting.
    source.request_immediate_snapshot()
    source.request_immediate_snapshot()
    source.request_immediate_snapshot()

    # Only one extra poll should happen.
    second = await asyncio.wait_for(gen.__anext__(), timeout=1.0)
    assert second.telemetry.drop_reason is None
    assert grpc_client.get_all_kv_cache_blocks.await_count == 2

    # After the coalesced poll, source waits full interval again.
    # Signal once more to get a third poll to prove no extra poll was queued.
    source.request_immediate_snapshot()
    await asyncio.wait_for(gen.__anext__(), timeout=1.0)
    assert grpc_client.get_all_kv_cache_blocks.await_count == 3
    await gen.aclose()


async def test_signal_during_poll_triggers_one_extra_poll(
    config: SubscriberConfig,
) -> None:
    """Signal arriving while gRPC poll is in-flight causes one extra poll after."""
    poll_started = asyncio.Event()
    release_poll = asyncio.Event()
    call_count = 0

    async def controlled_poll(timeout_s: float) -> Any:
        nonlocal call_count
        call_count += 1
        if call_count == 2:
            poll_started.set()
            await release_poll.wait()
        return _snapshot_response(block_hashes=[b"\x01"])

    source, grpc_client = _make_source(config)
    grpc_client.get_all_kv_cache_blocks = AsyncMock(side_effect=controlled_poll)

    gen = source.subscribe()
    # First poll completes normally.
    await asyncio.wait_for(gen.__anext__(), timeout=1.0)

    # Signal to trigger second poll, which will block.
    source.request_immediate_snapshot()
    second_task = asyncio.create_task(gen.__anext__())
    await asyncio.wait_for(poll_started.wait(), timeout=1.0)

    # Signal while poll is in-flight — should coalesce into one extra poll.
    source.request_immediate_snapshot()
    release_poll.set()

    second = await asyncio.wait_for(second_task, timeout=1.0)
    assert second.telemetry.drop_reason is None

    # One extra poll should happen (from the in-flight signal).
    third = await asyncio.wait_for(gen.__anext__(), timeout=1.0)
    assert third.telemetry.drop_reason is None
    assert call_count == 3
    await gen.aclose()


async def test_signal_during_error_backoff_wakes_immediately(
    config: SubscriberConfig,
) -> None:
    """Signal during error-path wait interrupts the backoff interval."""
    import grpc

    grpc_error = grpc.aio.AioRpcError(
        code=grpc.StatusCode.UNAVAILABLE,
        initial_metadata=None,
        trailing_metadata=None,
        details="down",
        debug_error_string=None,
    )
    source, grpc_client = _make_source(
        config,
        responses=[
            grpc_error,
            _snapshot_response(block_hashes=[b"\x01"]),
        ],
    )

    gen = source.subscribe()
    # First poll fails and enters error backoff wait.
    first_task = asyncio.create_task(gen.__anext__())
    first = await asyncio.wait_for(first_task, timeout=1.0)
    assert first.telemetry.drop_reason == "fetch_failed"

    # Source is now in error-path _wait_interval. Signal it.
    source.request_immediate_snapshot()

    # Retry should happen quickly.
    second_task = asyncio.create_task(gen.__anext__())
    second = await asyncio.wait_for(second_task, timeout=1.0)
    assert second.telemetry.drop_reason is None
    await gen.aclose()


async def test_rate_limited_warning_on_repeated_signal(
    config: SubscriberConfig, mocker: Any
) -> None:
    """Warning is logged (rate-limited) when signal arrives with event already set."""
    warning_mock = mocker.patch("subscriber.engine.snapshot.logger.warning")
    source, _ = _make_source(config)

    # Signal twice without consumption.
    source.request_immediate_snapshot()
    source.request_immediate_snapshot()

    # At least one warning about coalesced signal.
    warning_mock.assert_called()
    call_messages = [str(c.args[0]) for c in warning_mock.call_args_list]
    assert any("snapshot signal" in msg for msg in call_messages)


async def test_no_signal_preserves_interval_behavior(
    config: SubscriberConfig,
) -> None:
    """Without signal, source still polls on the configured interval."""
    source, grpc_client = _make_source(config)

    gen = source.subscribe()
    first = await asyncio.wait_for(gen.__anext__(), timeout=1.0)
    assert first.telemetry.drop_reason is None

    # Without signal, the next poll should NOT complete within 0.1s
    # (interval is 30s). We verify by timing out.
    second_task = asyncio.create_task(gen.__anext__())
    with pytest.raises(asyncio.TimeoutError):
        await asyncio.wait_for(asyncio.shield(second_task), timeout=0.1)

    # Cleanup: signal to unblock, then close.
    source.request_immediate_snapshot()
    await asyncio.wait_for(second_task, timeout=1.0)
    await gen.aclose()


async def test_no_signal_polls_after_wait_interval_timeout(
    config: SubscriberConfig, mocker: Any
) -> None:
    """The normal polling path resumes only after its interval wait times out."""

    config.engine_snapshot_full_sync_interval_s = 0.01
    source, grpc_client = _make_source(config)
    real_wait_for = asyncio.wait_for
    interval_timed_out = asyncio.Event()

    async def observe_wait_for(awaitable: Any, timeout: float | None = None) -> Any:
        try:
            return await real_wait_for(awaitable, timeout)
        except TimeoutError:
            interval_timed_out.set()
            raise

    mocker.patch.object(
        snapshot_module.asyncio, "wait_for", side_effect=observe_wait_for
    )
    gen = source.subscribe()
    try:
        async with asyncio.timeout(1.0):
            first = await gen.__anext__()
        async with asyncio.timeout(1.0):
            second = await gen.__anext__()

        assert first.telemetry.drop_reason is None
        assert second.telemetry.drop_reason is None
        assert interval_timed_out.is_set()
        assert grpc_client.get_all_kv_cache_blocks.await_count == 2
    finally:
        await gen.aclose()
