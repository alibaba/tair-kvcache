from __future__ import annotations

import asyncio
import unittest
from types import SimpleNamespace

from kv_cache_manager.cache_event_subscriber.__main__ import _parse_rtp_endpoints
from kv_cache_manager.cache_event_subscriber.key_codec import to_signed_i64
from kv_cache_manager.cache_event_subscriber.models import BlockRecord, EngineUpdate, LocationSpec, SourceNotReady
from kv_cache_manager.cache_event_subscriber.reporter import KvcmEventReporter
from kv_cache_manager.cache_event_subscriber.rtp_source import RtpCacheSource
from kv_cache_manager.cache_event_subscriber.service import CacheEventSubscriberService
from kv_cache_manager.cache_event_subscriber.vllm_source import VllmCacheSource, VllmHistoryGap


def _rtp_response(
    *,
    head: int,
    next_cursor: int,
    reset: bool = False,
    more: bool = False,
    events: list[object] | None = None,
    snapshot: list[object] | None = None,
    generation: int = 7,
    protocol: int = 2,
    cache_keys: dict[int, bool] | None = None,
    block_size: int = 64,
) -> SimpleNamespace:
    return SimpleNamespace(
        cache_event_protocol_version=protocol,
        cache_event_generation=generation,
        cache_event_reset_required=reset,
        cache_event_has_more=more,
        cache_event_version=head,
        next_cache_event_version=next_cursor,
        cache_events=events or [],
        cache_event_snapshot=snapshot or [],
        cache_keys=cache_keys or {},
        version=head,
        block_size=block_size,
    )


def _event(version: int, event_type: int, key: int, groups: tuple[int, ...] = (0,)) -> SimpleNamespace:
    return SimpleNamespace(
        version=version,
        event_type=event_type,
        cache_key=key,
        group_ids=groups,
    )


class BlockKeyCodecTest(unittest.TestCase):
    def test_vllm_bytes_and_legacy_integer_have_the_same_key(self) -> None:
        digest = bytes(range(32))
        legacy_integer = int.from_bytes(digest, "big") & ((1 << 64) - 1)
        self.assertEqual(to_signed_i64(legacy_integer), to_signed_i64(digest))

    def test_empty_bytes_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            to_signed_i64(b"")


class EndpointConfigTest(unittest.TestCase):
    def test_rtp_requires_one_unique_endpoint_per_dp_rank(self) -> None:
        self.assertEqual(
            ("rank-0:8089", "rank-1:8089"),
            _parse_rtp_endpoints("rank-0:8089, rank-1:8089", 2),
        )
        with self.assertRaisesRegex(ValueError, "expected 2, got 1"):
            _parse_rtp_endpoints("rank-0:8089", 2)
        with self.assertRaisesRegex(ValueError, "must be unique"):
            _parse_rtp_endpoints("rank-0:8089,rank-0:8089", 2)


class FakeRtpTransport:
    def __init__(self, responses: list[object]) -> None:
        self.responses = responses
        self.requests: list[object] = []

    async def call(self, endpoint: str, request: object, timeout_s: float) -> object:
        self.requests.append(request)
        if not self.responses:
            raise AssertionError("unexpected RTP request")
        return self.responses.pop(0)

    async def close(self) -> None:
        pass


class EndpointRtpTransport:
    def __init__(self, responses: dict[str, list[object | Exception]]) -> None:
        self.responses = responses
        self.requests: list[tuple[str, object]] = []

    async def call(self, endpoint: str, request: object, timeout_s: float) -> object:
        self.requests.append((endpoint, request))
        response = self.responses[endpoint].pop(0)
        if isinstance(response, Exception):
            raise response
        return response

    async def close(self) -> None:
        pass


class RtpSourceTest(unittest.IsolatedAsyncioTestCase):
    async def test_snapshot_then_paginated_delta_uses_explicit_next_cursor(self) -> None:
        transport = FakeRtpTransport(
            [
                _rtp_response(
                    head=0,
                    next_cursor=0,
                    reset=True,
                    snapshot=[SimpleNamespace(cache_key=1, group_ids=(0,))],
                ),
                _rtp_response(
                    head=2,
                    next_cursor=1,
                    more=True,
                    events=[_event(1, 0, 2)],
                ),
                _rtp_response(
                    head=2,
                    next_cursor=2,
                    events=[_event(2, 1, 1)],
                ),
            ]
        )
        source = RtpCacheSource(["worker:123"], transport=transport)

        initial = await source.prepare()
        self.assertTrue(initial.full_snapshot)
        self.assertEqual((BlockRecord(1, "hbm", (0,)),), initial.blocks)
        source.commit(initial)

        delta = await source.prepare()
        self.assertFalse(delta.full_snapshot)
        self.assertEqual((BlockRecord(2, "hbm", (0,)),), delta.upserts)
        self.assertEqual((BlockRecord(1, "hbm", (0,)),), delta.removals)
        self.assertEqual(0, transport.requests[1].latest_cache_version)
        self.assertEqual(1, transport.requests[2].latest_cache_version)

    async def test_abort_does_not_advance_cursor(self) -> None:
        transport = FakeRtpTransport(
            [
                _rtp_response(head=0, next_cursor=0, reset=True, snapshot=[]),
                _rtp_response(head=1, next_cursor=1, events=[_event(1, 0, 9)]),
                _rtp_response(head=1, next_cursor=1, events=[_event(1, 0, 9)]),
            ]
        )
        source = RtpCacheSource(["worker:123"], transport=transport)
        initial = await source.prepare()
        source.commit(initial)
        failed = await source.prepare()
        source.abort(failed)
        retried = await source.prepare()
        self.assertEqual(failed.upserts, retried.upserts)
        self.assertEqual(0, transport.requests[1].latest_cache_version)
        self.assertEqual(0, transport.requests[2].latest_cache_version)

    async def test_multi_dp_failure_does_not_commit_partial_candidate(self) -> None:
        rank_0_snapshot = _rtp_response(
            head=0,
            next_cursor=0,
            reset=True,
            snapshot=[SimpleNamespace(cache_key=1, group_ids=(0,))],
        )
        transport = EndpointRtpTransport(
            {
                "rank-0:123": [rank_0_snapshot, rank_0_snapshot],
                "rank-1:123": [
                    RuntimeError("rank 1 unavailable"),
                    _rtp_response(
                        head=0,
                        next_cursor=0,
                        reset=True,
                        snapshot=[SimpleNamespace(cache_key=2, group_ids=(0,))],
                    ),
                ],
            }
        )
        source = RtpCacheSource(["rank-0:123", "rank-1:123"], transport=transport)

        with self.assertRaisesRegex(RuntimeError, "rank 1 unavailable"):
            await source.prepare()

        recovered = await source.prepare()
        self.assertTrue(recovered.full_snapshot)
        self.assertEqual(
            (BlockRecord(1, "hbm", (0,)), BlockRecord(2, "hbm", (0,))),
            recovered.blocks,
        )
        rank_0_requests = [
            request for endpoint, request in transport.requests if endpoint == "rank-0:123"
        ]
        self.assertEqual([-1, -1], [request.latest_cache_version for request in rank_0_requests])

    async def test_snapshot_rejects_unregistered_cache_group(self) -> None:
        transport = FakeRtpTransport(
            [
                _rtp_response(
                    head=0,
                    next_cursor=0,
                    reset=True,
                    snapshot=[SimpleNamespace(cache_key=1, group_ids=(1,))],
                )
            ]
        )
        source = RtpCacheSource(
            ["worker:123"], cache_group_count=1, transport=transport
        )

        with self.assertRaisesRegex(RuntimeError, "unregistered cache group"):
            await source.prepare()

    async def test_event_rejects_empty_cache_groups(self) -> None:
        transport = FakeRtpTransport(
            [
                _rtp_response(
                    head=0,
                    next_cursor=0,
                    reset=True,
                    snapshot=[SimpleNamespace(cache_key=1, group_ids=())],
                )
            ]
        )
        source = RtpCacheSource(["worker:123"], transport=transport)

        with self.assertRaisesRegex(RuntimeError, "no cache group ids"):
            await source.prepare()

    async def test_rtp_block_size_must_match_registration(self) -> None:
        transport = FakeRtpTransport(
            [_rtp_response(head=0, next_cursor=0, reset=True, block_size=32)]
        )
        source = RtpCacheSource(
            ["worker:123"], expected_block_size=64, transport=transport
        )

        with self.assertRaisesRegex(RuntimeError, "block size does not match"):
            await source.prepare()

    async def test_legacy_full_key_map_uses_delta_between_reconciliations(self) -> None:
        transport = FakeRtpTransport(
            [
                _rtp_response(
                    head=4,
                    next_cursor=0,
                    protocol=0,
                    cache_keys={1: True},
                ),
                _rtp_response(
                    head=5,
                    next_cursor=0,
                    protocol=0,
                    cache_keys={2: True},
                ),
            ]
        )
        source = RtpCacheSource(["worker:123"], transport=transport)

        initial = await source.prepare()
        self.assertTrue(initial.full_snapshot)
        self.assertEqual((BlockRecord(1, "hbm", (0,)),), initial.blocks)
        source.commit(initial)

        delta = await source.prepare()
        self.assertFalse(delta.full_snapshot)
        self.assertEqual((BlockRecord(2, "hbm", (0,)),), delta.upserts)
        self.assertEqual((BlockRecord(1, "hbm", (0,)),), delta.removals)

    async def test_legacy_full_key_map_rejects_multiple_cache_groups(self) -> None:
        transport = FakeRtpTransport(
            [
                _rtp_response(
                    head=1,
                    next_cursor=0,
                    protocol=0,
                    cache_keys={1: True},
                )
            ]
        )
        source = RtpCacheSource(
            ["worker:123"], cache_group_count=2, transport=transport
        )

        with self.assertRaisesRegex(RuntimeError, "has no cache-group identity"):
            await source.prepare()


class FakeManagerClient:
    def __init__(self) -> None:
        self.reports: list[dict[str, object]] = []
        self.registrations: list[dict[str, object]] = []

    def report_event(self, request: dict[str, object]) -> None:
        self.reports.append(request)

    def register_instance(self, request: dict[str, object]) -> None:
        self.registrations.append(request)


class ReporterTest(unittest.TestCase):
    def test_full_snapshot_is_one_authoritative_event(self) -> None:
        client = FakeManagerClient()
        reporter = KvcmEventReporter(
            client,  # type: ignore[arg-type]
            instance_id="instance",
            host_ip_port="host:1",
            spec_factory=lambda block: [LocationSpec("group_0", "event-report://host:1/hbm")],
        )
        reporter.send(EngineUpdate(True, blocks=(BlockRecord(5, "hbm", (0,)),)))
        events = client.reports[0]["events"]
        assert isinstance(events, list)
        self.assertEqual("ST_EVENT_REPORT", client.reports[0]["storage_type"])
        self.assertEqual("EVENT_BLOCK_SNAPSHOT", events[0]["event_type"])
        self.assertEqual("5", events[0]["block_snapshot"]["blocks"][0]["block_key"])

    def test_incremental_events_are_chunked(self) -> None:
        client = FakeManagerClient()
        reporter = KvcmEventReporter(
            client,  # type: ignore[arg-type]
            instance_id="instance",
            host_ip_port="host:1",
            max_events_per_request=2,
            spec_factory=lambda block: [LocationSpec("group_0", "event-report://host:1/hbm")],
        )
        reporter.send(
            EngineUpdate(
                False,
                upserts=tuple(BlockRecord(key, "hbm", (0,)) for key in range(3)),
            )
        )
        self.assertEqual(2, len(client.reports))

    def test_large_snapshot_clears_then_upserts_in_chunks(self) -> None:
        client = FakeManagerClient()
        reporter = KvcmEventReporter(
            client,  # type: ignore[arg-type]
            instance_id="instance",
            host_ip_port="host:1",
            max_events_per_request=2,
            spec_factory=lambda block: [LocationSpec("group_0", "event-report://host:1/hbm")],
        )
        reporter.send(
            EngineUpdate(
                True,
                blocks=tuple(BlockRecord(key, "hbm", (0,)) for key in range(3)),
            )
        )
        self.assertEqual(3, len(client.reports))
        first_events = client.reports[0]["events"]
        assert isinstance(first_events, list)
        self.assertEqual([], first_events[0]["block_snapshot"]["blocks"])


class FlakySource:
    def __init__(self) -> None:
        self.attempts = 0
        self.closed = False
        self.initialized = False
        self.on_commit = lambda: None

    async def prepare(self) -> EngineUpdate:
        self.attempts += 1
        if self.attempts == 1:
            raise RuntimeError("transient source failure")
        return EngineUpdate(not self.initialized)

    def commit(self, update: EngineUpdate) -> None:
        self.initialized = True
        self.on_commit()

    def abort(self, update: EngineUpdate) -> None:
        pass

    async def close(self) -> None:
        self.closed = True


class FailingSource(FlakySource):
    async def prepare(self) -> EngineUpdate:
        self.attempts += 1
        raise RuntimeError("persistent source failure")


class NotReadySource(FlakySource):
    async def prepare(self) -> EngineUpdate:
        self.attempts += 1
        if self.attempts < 3:
            raise SourceNotReady("waiting for sequence zero")
        return EngineUpdate(not self.initialized)


class IdleSource(FlakySource):
    async def prepare(self) -> EngineUpdate:
        self.attempts += 1
        return EngineUpdate(not self.initialized)


class FakeServiceReporter:
    def __init__(self) -> None:
        self.registrations = 0
        self.host_down_reports = 0
        self.heartbeats = 0
        self.updates = 0

    def register_node(self, mediums: tuple[str, ...]) -> None:
        self.registrations += 1

    def send(self, update: EngineUpdate) -> None:
        self.updates += 1

    def heartbeat(self) -> None:
        self.heartbeats += 1

    def host_down(self) -> None:
        self.host_down_reports += 1


class FlakyInitialSnapshotReporter(FakeServiceReporter):
    def send(self, update: EngineUpdate) -> None:
        super().send(update)
        if self.updates == 1:
            raise RuntimeError("reject first snapshot")


class SubscriberServiceTest(unittest.IsolatedAsyncioTestCase):
    async def test_source_not_ready_does_not_trip_failure_threshold(self) -> None:
        source = NotReadySource()
        reporter = FakeServiceReporter()
        service = CacheEventSubscriberService(
            source,
            reporter,  # type: ignore[arg-type]
            mediums=("hbm",),
            poll_interval_s=0,
            retry_interval_s=0,
            heartbeat_interval_s=60,
            source_failure_threshold=1,
        )
        source.on_commit = service.stop

        await asyncio.wait_for(service.run(), timeout=1)

        self.assertEqual(3, source.attempts)
        self.assertEqual(1, reporter.registrations)
        self.assertEqual(1, reporter.updates)
        self.assertEqual(1, reporter.host_down_reports)
        self.assertTrue(source.closed)

    async def test_initial_snapshot_failure_unadvertises_before_retry(self) -> None:
        source = IdleSource()
        reporter = FlakyInitialSnapshotReporter()
        service = CacheEventSubscriberService(
            source,
            reporter,  # type: ignore[arg-type]
            mediums=("hbm",),
            poll_interval_s=0,
            retry_interval_s=0,
            heartbeat_interval_s=60,
        )
        source.on_commit = service.stop

        await asyncio.wait_for(service.run(), timeout=1)

        self.assertEqual(1, source.attempts)
        self.assertEqual(2, reporter.registrations)
        self.assertEqual(2, reporter.updates)
        # One report removes the rejected bootstrap registration; the second
        # is the normal graceful shutdown after the committed retry.
        self.assertEqual(2, reporter.host_down_reports)

    async def test_transient_source_failure_is_retried(self) -> None:
        source = FlakySource()
        reporter = FakeServiceReporter()
        service = CacheEventSubscriberService(
            source,
            reporter,  # type: ignore[arg-type]
            mediums=("hbm",),
            poll_interval_s=0,
            retry_interval_s=0,
            heartbeat_interval_s=60,
        )
        source.on_commit = service.stop

        await asyncio.wait_for(service.run(), timeout=1)

        self.assertEqual(2, source.attempts)
        self.assertEqual(1, reporter.registrations)
        self.assertEqual(1, reporter.host_down_reports)
        self.assertTrue(source.closed)

    async def test_persistent_source_failure_reports_down_and_exits(self) -> None:
        source = FailingSource()
        reporter = FakeServiceReporter()
        service = CacheEventSubscriberService(
            source,
            reporter,  # type: ignore[arg-type]
            mediums=("hbm",),
            poll_interval_s=0,
            retry_interval_s=0,
            heartbeat_interval_s=60,
            source_failure_threshold=2,
        )

        with self.assertRaisesRegex(RuntimeError, "failure threshold"):
            await service.run()

        self.assertEqual(2, source.attempts)
        self.assertEqual(0, reporter.registrations)
        self.assertEqual(0, reporter.host_down_reports)
        self.assertTrue(source.closed)

    async def test_failed_health_probe_stops_heartbeats_and_exits(self) -> None:
        source = IdleSource()
        reporter = FakeServiceReporter()
        service = CacheEventSubscriberService(
            source,
            reporter,  # type: ignore[arg-type]
            mediums=("hbm",),
            poll_interval_s=0.001,
            retry_interval_s=0,
            heartbeat_interval_s=0.001,
            health_probe=lambda: False,
            health_failure_threshold=1,
        )

        # Health probes run through the default thread executor. Leave enough
        # headroom for a loaded build host without weakening the service's
        # millisecond-level failure threshold used by this test.
        await asyncio.wait_for(service.run(), timeout=5)

        self.assertEqual(0, reporter.heartbeats)
        self.assertEqual(1, reporter.host_down_reports)
        self.assertTrue(source.closed)


class FakeVllmTransport:
    def __init__(self, batches: list[list[tuple[int, bytes]]]) -> None:
        self.batches = batches

    async def next_sequences(self, expected_sequence: int) -> list[tuple[int, bytes]]:
        return self.batches.pop(0)

    async def close(self) -> None:
        pass


class Batch:
    def __init__(self, events: list[object]) -> None:
        self.events = events


class BlockStored:
    def __init__(self, key: int) -> None:
        self.block_hashes = [key]
        self.medium = "GPU"
        self.group_idx = 0


class BlockRemoved:
    def __init__(self, key: int) -> None:
        self.block_hashes = [key]
        self.medium = "GPU"
        self.group_idx = 0
        self.remaining_copy_counts = [0]


class AllBlocksCleared:
    pass


class VllmSourceTest(unittest.IsolatedAsyncioTestCase):
    async def test_empty_publisher_waits_for_authoritative_baseline(self) -> None:
        source = VllmCacheSource(
            "pub",
            "replay",
            transport=FakeVllmTransport([[]]),
            decoder=lambda payload: payload,
        )

        with self.assertRaises(SourceNotReady):
            await source.prepare()

    async def test_sequence_zero_builds_snapshot_then_delta(self) -> None:
        decoded = {b"stored": Batch([BlockStored(11)]), b"removed": Batch([BlockRemoved(11)])}
        source = VllmCacheSource(
            "unused",
            "unused",
            transport=FakeVllmTransport([[(0, b"stored")], [(1, b"removed")]]),
            decoder=decoded.__getitem__,
        )
        initial = await source.prepare()
        self.assertTrue(initial.full_snapshot)
        self.assertEqual((BlockRecord(11, "hbm", (0,)),), initial.blocks)
        source.commit(initial)
        delta = await source.prepare()
        self.assertEqual((BlockRecord(11, "hbm", (0,)),), delta.removals)

    async def test_pruned_cold_history_is_rejected(self) -> None:
        source = VllmCacheSource(
            "unused",
            "unused",
            transport=FakeVllmTransport([[(5, b"stored")]]),
            decoder=lambda payload: Batch([BlockStored(11)]),
        )
        with self.assertRaises(VllmHistoryGap):
            await source.prepare()

    async def test_clear_marker_recovers_from_pruned_cold_history(self) -> None:
        decoded = {
            b"reset": Batch([AllBlocksCleared(), BlockStored(12)]),
            b"delta": Batch([BlockStored(13)]),
        }
        source = VllmCacheSource(
            "unused",
            "unused",
            transport=FakeVllmTransport([[(5, b"reset")], [(6, b"delta")]]),
            decoder=decoded.__getitem__,
        )

        initial = await source.prepare()
        self.assertTrue(initial.full_snapshot)
        self.assertEqual((BlockRecord(12, "hbm", (0,)),), initial.blocks)
        source.commit(initial)

        delta = await source.prepare()
        self.assertFalse(delta.full_snapshot)
        self.assertEqual((BlockRecord(13, "hbm", (0,)),), delta.upserts)

    async def test_idle_stream_emits_periodic_full_snapshot(self) -> None:
        now = [0.0]
        source = VllmCacheSource(
            "unused",
            "unused",
            transport=FakeVllmTransport([[(0, b"stored")], []]),
            decoder=lambda payload: Batch([BlockStored(14)]),
            full_refresh_interval_s=10,
            clock=lambda: now[0],
        )
        initial = await source.prepare()
        source.commit(initial)
        now[0] = 11

        refresh = await source.prepare()

        self.assertTrue(refresh.full_snapshot)
        self.assertEqual((BlockRecord(14, "hbm", (0,)),), refresh.blocks)

    async def test_block_size_mismatch_is_rejected(self) -> None:
        stored = BlockStored(15)
        stored.block_size = 32
        source = VllmCacheSource(
            "unused",
            "unused",
            transport=FakeVllmTransport([[(0, b"stored")]]),
            decoder=lambda payload: Batch([stored]),
            expected_block_size=16,
        )

        with self.assertRaisesRegex(RuntimeError, "block size"):
            await source.prepare()

    async def test_sequence_zero_resets_a_restarted_publisher(self) -> None:
        decoded = {
            b"old": Batch([BlockStored(16)]),
            b"new": Batch([BlockStored(17)]),
        }
        source = VllmCacheSource(
            "unused",
            "unused",
            transport=FakeVllmTransport([[(0, b"old")], [(0, b"new")]]),
            decoder=decoded.__getitem__,
        )
        old = await source.prepare()
        source.commit(old)

        restarted = await source.prepare()

        self.assertTrue(restarted.full_snapshot)
        self.assertEqual((BlockRecord(17, "hbm", (0,)),), restarted.blocks)

    async def test_removed_copy_counts_must_match_hashes(self) -> None:
        removed = BlockRemoved(18)
        removed.block_hashes.append(19)
        source = VllmCacheSource(
            "unused",
            "unused",
            transport=FakeVllmTransport([[(0, b"reset")]]),
            decoder=lambda payload: Batch([AllBlocksCleared(), removed]),
        )

        with self.assertRaisesRegex(RuntimeError, "remaining_copy_counts length"):
            await source.prepare()


if __name__ == "__main__":
    unittest.main()
