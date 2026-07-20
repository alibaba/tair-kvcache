from __future__ import annotations

import asyncio
from collections import Counter
from collections.abc import AsyncGenerator

import httpx
import msgspec
import zmq
import zmq.asyncio

from subscriber import logger
from subscriber.config import SubscriberConfig
from subscriber.engine.base import AbstractEngineAdapter, EngineEventBatch
from subscriber.engine.dashllm_grpc import DashllmGrpcClient
from subscriber.health.events import LivenessEvent
from subscriber.metrics import StageTimer, ZmqQueueMetricsReporter
from subscriber.proto import kv_cache_group_metadata_pb2
from subscriber.types import (
    BlockRemoved,
    BlockStored,
    ExternalBlockHash,
    KvCacheGroupSpec,
    KVEventBatch,
)
from subscriber.utils.msgpack_helper import KVEventBatchMsgpackHelper

_END_SEQ = (-1).to_bytes(8, "big", signed=True)
_MAX_DEBUG_BLOCK_HASHES = 32

# The subscriber is co-located with the engine, so control endpoints default to
# localhost addresses. URLs are configurable via SubscriberConfig.
_METADATA_RETRY_BASE_S = 0.5
_METADATA_RETRY_MAX_S = 30.0

MEDIUM_VLLM_GPU = "GPU"
MEDIUM_VLLM_CPU = "CPU"

_KVCM_MEDIUM_MAP = {
    MEDIUM_VLLM_GPU: "hbm",
    MEDIUM_VLLM_CPU: "mem",
}


class _RetryableMetadataResponseError(RuntimeError):
    """A DashLLM response that asks the subscriber to retry metadata fetch."""

    def __init__(self, err_code: int, err_msg: str) -> None:
        self.err_code = err_code
        super().__init__(f"DashLLM metadata error {err_code}: {err_msg}")


def _metadata_response_error(payload: object) -> tuple[int, str] | None:
    """Return an explicit metadata response error, if present."""

    err_code = getattr(payload, "err_code", 0)
    if not isinstance(err_code, int) or err_code == (
        kv_cache_group_metadata_pb2.KV_CACHE_GROUP_METADATA_OK
    ):
        return None
    err_msg = getattr(payload, "err_msg", "")
    return err_code, err_msg if isinstance(err_msg, str) else ""


def _format_block_hash(block_hash: ExternalBlockHash) -> str:
    """Render a block hash safely for dashlog's structured JSON output."""

    if isinstance(block_hash, bytes):
        return block_hash.hex()
    return str(block_hash)


def _event_for_debug(event: BlockStored | BlockRemoved) -> dict[str, object]:
    """Copy an event for logging without token IDs or mutating forwarding data."""

    event_data = msgspec.structs.asdict(event)
    event_data.pop("token_ids", None)
    event_data["block_hashes"] = [
        _format_block_hash(block_hash) for block_hash in event.block_hashes
    ]
    if isinstance(event, BlockStored) and event.parent_block_hash is not None:
        event_data["parent_block_hash"] = _format_block_hash(event.parent_block_hash)
    return event_data


def _summarize_batch(batch: KVEventBatch) -> dict[str, object]:
    event_type_counts = Counter(type(event).__name__ for event in batch.events)
    stored_blocks: list[dict[str, object]] = []
    removed_blocks: list[dict[str, object]] = []
    for event in batch.events:
        if isinstance(event, BlockStored):
            if len(stored_blocks) < _MAX_DEBUG_BLOCK_HASHES:
                stored_blocks.append(_event_for_debug(event))
        elif isinstance(event, BlockRemoved):
            if len(removed_blocks) < _MAX_DEBUG_BLOCK_HASHES:
                removed_blocks.append(_event_for_debug(event))
    return {
        "event_count": len(batch.events),
        "event_types": ",".join(
            f"{name}:{count}" for name, count in sorted(event_type_counts.items())
        ),
        "data_parallel_rank": batch.data_parallel_rank,
        "stored_block_count": len(stored_blocks),
        "stored_blocks": stored_blocks,
        "stored_blocks_truncated": sum(
            1 for e in batch.events if isinstance(e, BlockStored)
        )
        > len(stored_blocks),
        "removed_block_count": len(removed_blocks),
        "removed_blocks": removed_blocks,
        "removed_blocks_truncated": sum(
            1 for e in batch.events if isinstance(e, BlockRemoved)
        )
        > len(removed_blocks),
    }


def _parse_kv_cache_group_metadata(payload: object) -> list[KvCacheGroupSpec] | None:
    """Parse gRPC metadata into per-group specs.

    ``GetKvCacheGroupsMetadata`` returns repeated entries with ``group_idx``,
    ``kind``, ``block_size``, and ``sliding_window``. ``-1`` encodes no sliding
    window. Returns ``None`` when the response has no group topology.
    """

    raw = getattr(payload, "items", None)
    if raw is None:
        return None
    specs: list[KvCacheGroupSpec] = []
    for entry in raw:
        group_idx = getattr(entry, "group_idx", None)
        kind = getattr(entry, "kind", None)
        block_size = getattr(entry, "block_size", None)
        if (
            not isinstance(group_idx, int)
            or not isinstance(kind, str)
            or not isinstance(block_size, int)
        ):
            return None
        sliding_window = getattr(entry, "sliding_window", None)
        specs.append(
            KvCacheGroupSpec(
                group_idx=group_idx,
                kind=kind,
                block_size=block_size,
                sliding_window=sliding_window
                if isinstance(sliding_window, int) and sliding_window != -1
                else None,
            )
        )
    return specs or None


async def _probe_health(client: httpx.AsyncClient, url: str) -> LivenessEvent:
    """Map one HTTP health probe result to an engine liveness event."""

    try:
        response = await client.get(url)
    except (httpx.ReadTimeout, httpx.ConnectTimeout) as exc:
        logger.warning(
            "engine health probe timed out",
            step="engine_health",
            tags={"error": type(exc).__name__, "url": url},
        )
        return LivenessEvent.UNHEALTHY
    except httpx.ConnectError as exc:
        logger.warning(
            "engine health probe connection refused",
            step="engine_health",
            tags={"error": type(exc).__name__, "url": url},
        )
        return LivenessEvent.UNHEALTHY
    except httpx.HTTPError as exc:
        logger.warning(
            "engine health probe failed",
            step="engine_health",
            tags={
                "error": type(exc).__name__,
                "message": str(exc),
                "url": url,
            },
        )
        return LivenessEvent.UNHEALTHY
    if response.status_code == 200:
        return LivenessEvent.HEALTHY
    logger.warning(
        "engine health probe returned non-200",
        step="engine_health",
        tags={"status_code": response.status_code, "url": url},
    )
    return LivenessEvent.UNHEALTHY


@AbstractEngineAdapter.register("vllm")
class VllmAdapter(AbstractEngineAdapter):
    """Engine adapter for vLLM.

    Uses ZMQ SUB + DEALER for KV event subscription and replay. Liveness is
    reported through an HTTP /health polling loop. DashLLM control-plane RPCs
    use the adapter-owned gRPC client.
    """

    def __init__(self, config: SubscriberConfig) -> None:
        """Open the vLLM subscription and replay sockets from subscriber config."""

        if config.data_parallel_size != 1:
            raise ValueError(
                "vLLM adapter currently supports exactly one DP endpoint; "
                "multi-DP must not silently subscribe to rank 0 only"
            )
        self._config = config
        self._endpoint = config.dp_endpoints[0]
        self._ctx = zmq.asyncio.Context.instance()
        self._msgpack_helper = KVEventBatchMsgpackHelper()
        self._last_seq = -1
        self._generation = 0
        self._closed = False
        self._dashllm_grpc_client = DashllmGrpcClient(self._config.engine_grpc_endpoint)
        if logger.is_debug_enabled():
            logger.debug(
                "connecting vLLM ZMQ sockets",
                step="zmq_connect",
                tags={
                    "pub_endpoint": self._endpoint.zmq_pub_endpoint,
                    "replay_endpoint": self._endpoint.zmq_replay_endpoint,
                    "topic": self._endpoint.zmq_topic,
                    "reconnect_ivl_ms": self._config.zmq_reconnect_ivl_ms,
                    "reconnect_ivl_max_ms": self._config.zmq_reconnect_ivl_max_ms,
                },
            )
        self._sub = self._open_sub_socket()
        self._dealer = self._open_dealer_socket()
        self._zmq_queue_metrics = ZmqQueueMetricsReporter(
            state_reader=self._zmq_queue_state,
        )

    def _open_sub_socket(self) -> zmq.asyncio.Socket:
        sub = self._ctx.socket(zmq.SUB)
        self._configure_sub_socket(sub)
        sub.connect(self._endpoint.zmq_pub_endpoint)
        sub.setsockopt_string(zmq.SUBSCRIBE, self._endpoint.zmq_topic)
        return sub

    def _open_dealer_socket(self) -> zmq.asyncio.Socket:
        dealer = self._ctx.socket(zmq.DEALER)
        dealer.connect(self._endpoint.zmq_replay_endpoint)
        return dealer

    def _replace_replay_socket(
        self, failed_dealer: zmq.asyncio.Socket, generation: int
    ) -> None:
        """Discard a failed replay socket so a stale reply cannot be reused."""

        if generation != self._generation or failed_dealer is not self._dealer:
            return
        failed_dealer.close(linger=0)
        self._dealer = self._open_dealer_socket()

    def _configure_sub_socket(self, sub: zmq.asyncio.Socket) -> None:
        """Apply reconnect and TCP keepalive options to the SUB socket."""

        sub.setsockopt(zmq.RECONNECT_IVL, self._config.zmq_reconnect_ivl_ms)
        sub.setsockopt(
            zmq.RECONNECT_IVL_MAX,
            self._config.zmq_reconnect_ivl_max_ms,
        )
        if self._config.zmq_tcp_keepalive:
            sub.setsockopt(zmq.TCP_KEEPALIVE, 1)
            sub.setsockopt(
                zmq.TCP_KEEPALIVE_IDLE,
                self._config.zmq_tcp_keepalive_idle_s,
            )
            sub.setsockopt(
                zmq.TCP_KEEPALIVE_INTVL,
                self._config.zmq_tcp_keepalive_intvl_s,
            )
            sub.setsockopt(
                zmq.TCP_KEEPALIVE_CNT,
                self._config.zmq_tcp_keepalive_cnt,
            )
        else:
            sub.setsockopt(zmq.TCP_KEEPALIVE, 0)

    async def subscribe_kv_events(self) -> AsyncGenerator[EngineEventBatch, None]:
        """Subscribe to vLLM KV events and repair sequence gaps with replay.

        The generator yields replayed batches first when a gap is detected, then
        yields the live batch that exposed the gap. Sequence tracking is local
        to this adapter generation and is reset by ``reset_generation_state``.

        Each yield carries a :class:`StageTimer`: live events measure a
        ``decode`` stage, replayed events measure a ``replay_fetch`` stage
        (DEALER round-trip plus batch decode).
        """

        await self._zmq_queue_metrics.start()
        try:
            while True:
                received = await self._recv_live_message()
                if received is None:
                    continue
                seq, payload = received
                generation = self._generation

                if seq <= self._last_seq:
                    logger.warning(
                        "dropping stale or duplicate kv event sequence",
                        step="zmq_subscribe",
                        tags={"last_seq": self._last_seq, "current_seq": seq},
                    )
                    continue

                if seq > self._last_seq + 1:
                    missed = seq - self._last_seq - 1
                    self._zmq_queue_metrics.record_sequence_gap(
                        missed_message_count=missed
                    )
                    logger.warning(
                        "kv event sequence gap detected, triggering replay",
                        step="zmq_replay",
                        tags={
                            "last_seq": self._last_seq,
                            "current_seq": seq,
                            "missed": missed,
                        },
                    )
                    replay_timer = StageTimer()
                    replay_batches = await self._replay_missing_batches(seq, generation)
                    if generation != self._generation:
                        continue
                    if replay_batches:
                        replay_timer.mark("replay_fetch")
                        yield EngineEventBatch(replay_batches, replay_timer)
                    elif replay_batches == []:
                        logger.warning(
                            "replay returned no batches, "
                            "publisher buffer may have been pruned",
                            step="zmq_replay",
                            tags={"gap_start_seq": self._last_seq + 1},
                        )

                timer = StageTimer()
                batch = self._msgpack_helper.decode(
                    payload,
                    step="zmq_subscribe",
                    tags={"seq": seq},
                )
                if batch is None:
                    continue
                timer.mark("decode")
                if logger.is_debug_enabled():
                    logger.debug(
                        "decoded vLLM KV event batch",
                        step="zmq_subscribe",
                        tags={"seq": seq, **_summarize_batch(batch)},
                    )
                if generation != self._generation:
                    continue
                self._last_seq = seq
                yield EngineEventBatch([batch], timer)
        finally:
            await self._zmq_queue_metrics.stop()
            self._sub.close(linger=0)
            self._dealer.close(linger=0)

    async def _recv_live_message(self) -> tuple[int, bytes] | None:
        generation = self._generation
        sub = self._sub
        try:
            frames = await sub.recv_multipart()
        except Exception as exc:
            logger.warning(
                "failed to receive kv event message",
                step="zmq_subscribe",
                tags={"error": exc.__class__.__name__, "message": str(exc)},
            )
            return None
        if generation != self._generation:
            return None
        self._zmq_queue_metrics.record_message(
            message_bytes=sum(len(frame) for frame in frames),
            queue_nonempty_after_receive=self._sub_queue_is_readable(sub),
        )
        if len(frames) != 3:
            logger.warning(
                "dropping malformed kv event message frames",
                step="zmq_subscribe",
                tags={"frame_count": len(frames)},
            )
            return None
        topic, seq_bytes, payload = frames
        if len(seq_bytes) != 8:
            logger.warning(
                "dropping kv event message with invalid sequence frame",
                step="zmq_subscribe",
                tags={"seq_frame_len": len(seq_bytes)},
            )
            return None
        seq = int.from_bytes(seq_bytes, "big")
        if logger.is_debug_enabled():
            logger.debug(
                "received vLLM ZMQ live message",
                step="zmq_subscribe",
                tags={
                    "topic": topic.decode(errors="replace"),
                    "seq": seq,
                    "payload_bytes": len(payload),
                },
            )
        return seq, payload

    def _zmq_queue_state(self) -> dict[str, bool | int]:
        """Return the stable libzmq queue signals available for the SUB socket."""

        receive_high_water_mark = self._sub.getsockopt(zmq.RCVHWM)
        if not isinstance(receive_high_water_mark, int):
            raise TypeError("ZMQ RCVHWM must be an integer")
        return {
            "zmq_sub_readable": self._sub_queue_is_readable(self._sub),
            "zmq_sub_rcvhwm": receive_high_water_mark,
            "zmq_exact_queue_depth_available": False,
        }

    @staticmethod
    def _sub_queue_is_readable(sub: zmq.asyncio.Socket) -> bool:
        """Whether libzmq currently has at least one message ready to receive."""

        try:
            events = sub.getsockopt(zmq.EVENTS)
            return isinstance(events, int) and bool(events & zmq.POLLIN)
        except Exception:
            return False

    async def _replay_missing_batches(
        self, current_seq: int, generation: int
    ) -> list[KVEventBatch] | None:
        gap_start_seq = self._last_seq + 1
        if logger.is_debug_enabled():
            logger.debug(
                "requesting vLLM ZMQ replay",
                step="zmq_replay",
                tags={"gap_start_seq": gap_start_seq, "current_seq": current_seq},
            )
        dealer = self._dealer
        try:
            async with asyncio.timeout(self._config.zmq_replay_timeout_s):
                await dealer.send_multipart([b"", gap_start_seq.to_bytes(8, "big")])
        except TimeoutError:
            self._replace_replay_socket(dealer, generation)
            logger.warning(
                "kv event replay unavailable; sequence gap remains; "
                "forwarding live batch",
                step="zmq_replay",
                tags={
                    "gap_start_seq": gap_start_seq,
                    "current_seq": current_seq,
                    "error": "TimeoutError",
                    "message": "replay timed out",
                },
            )
            return None
        except Exception as exc:
            self._replace_replay_socket(dealer, generation)
            logger.warning(
                "kv event replay unavailable; sequence gap remains; "
                "forwarding live batch",
                step="zmq_replay",
                tags={
                    "gap_start_seq": gap_start_seq,
                    "current_seq": current_seq,
                    "error": exc.__class__.__name__,
                    "message": str(exc),
                },
            )
            return None
        if generation != self._generation:
            return None

        replay_batches: list[KVEventBatch] = []
        while True:
            try:
                async with asyncio.timeout(self._config.zmq_replay_timeout_s):
                    frames = await dealer.recv_multipart()
            except TimeoutError:
                self._replace_replay_socket(dealer, generation)
                logger.warning(
                    "kv event replay unavailable; sequence gap remains; "
                    "forwarding live batch",
                    step="zmq_replay",
                    tags={
                        "gap_start_seq": gap_start_seq,
                        "current_seq": current_seq,
                        "error": "TimeoutError",
                        "message": "replay timed out",
                    },
                )
                return None
            except Exception as exc:
                self._replace_replay_socket(dealer, generation)
                logger.warning(
                    "kv event replay unavailable; sequence gap remains; "
                    "forwarding live batch",
                    step="zmq_replay",
                    tags={
                        "gap_start_seq": gap_start_seq,
                        "current_seq": current_seq,
                        "error": exc.__class__.__name__,
                        "message": str(exc),
                    },
                )
                return None
            if generation != self._generation:
                return None
            if len(frames) != 3:
                logger.warning(
                    "dropping malformed kv event replay frames",
                    step="zmq_replay",
                    tags={
                        "gap_start_seq": gap_start_seq,
                        "current_seq": current_seq,
                        "frame_count": len(frames),
                    },
                )
                return None
            if frames[1] == _END_SEQ:
                if logger.is_debug_enabled():
                    logger.debug(
                        "completed vLLM ZMQ replay",
                        step="zmq_replay",
                        tags={
                            "gap_start_seq": gap_start_seq,
                            "current_seq": current_seq,
                            "batch_count": len(replay_batches),
                        },
                    )
                return replay_batches
            replay_seq = int.from_bytes(frames[1], "big")
            batch = self._msgpack_helper.decode(
                frames[2],
                step="zmq_replay",
                tags={"gap_start_seq": gap_start_seq, "current_seq": current_seq},
            )
            if batch is None:
                return None
            if logger.is_debug_enabled():
                logger.debug(
                    "decoded vLLM KV event batch",
                    step="zmq_replay",
                    tags={
                        "gap_start_seq": gap_start_seq,
                        "current_seq": current_seq,
                        "replay_seq": replay_seq,
                        **_summarize_batch(batch),
                    },
                )
            replay_batches.append(batch)

    async def watch_liveness(self) -> AsyncGenerator[LivenessEvent, None]:
        """Poll the configured vLLM health endpoint as liveness events."""

        while True:
            try:
                async with httpx.AsyncClient(
                    timeout=self._config.engine_health_timeout_s
                ) as client:
                    while True:
                        try:
                            event = await _probe_health(
                                client, self._config.engine_health_url
                            )
                        except Exception as exc:
                            logger.warning(
                                "engine health probe failed unexpectedly",
                                step="engine_health",
                                tags={
                                    "error": exc.__class__.__name__,
                                    "message": str(exc),
                                },
                            )
                            event = LivenessEvent.UNHEALTHY
                        yield event
                        await asyncio.sleep(self._config.engine_health_interval_s)
            except Exception as exc:
                logger.warning(
                    "engine health watch loop failed unexpectedly",
                    step="engine_health",
                    tags={"error": exc.__class__.__name__, "message": str(exc)},
                )
                yield LivenessEvent.UNHEALTHY
                await asyncio.sleep(self._config.engine_health_interval_s)

    async def fetch_kv_cache_group_metadata(self) -> list[KvCacheGroupSpec] | None:
        """Fetch per-group metadata from the engine's gRPC endpoint.

        Retries with exponential backoff up to ``engine_kv_group_metadata_max_retries``
        attempts. The engine is already confirmed healthy by the coordinator
        before this method is called, so failures are transient transport
        errors rather than engine unavailability.
        """

        max_retries = self._config.engine_kv_group_metadata_max_retries
        delay = _METADATA_RETRY_BASE_S
        for attempt in range(1, max_retries + 1):
            try:
                payload = await self._dashllm_grpc_client.get_kv_cache_group_metadata(
                    self._config.engine_health_timeout_s
                )
                response_error = _metadata_response_error(payload)
                if response_error is not None:
                    err_code, err_msg = response_error
                    if err_code == (
                        kv_cache_group_metadata_pb2.KV_CACHE_GROUP_METADATA_UNAVAILABLE
                    ):
                        raise _RetryableMetadataResponseError(err_code, err_msg)
                    logger.warning(
                        "received non-retryable kv cache group metadata error; "
                        "falling back to learn-mode",
                        step="kv_metadata",
                        tags={
                            "err_code": err_code,
                            "err_msg": err_msg,
                            "target": self._config.engine_grpc_endpoint,
                        },
                    )
                    return None
            # TODO: Retry only _RetryableMetadataResponseError and explicitly
            # retryable gRPC transport statuses instead of every Exception.
            except Exception as exc:
                if attempt >= max_retries:
                    logger.warning(
                        "failed to fetch kv cache group metadata; "
                        "max retries exhausted, falling back to learn-mode",
                        step="kv_metadata",
                        tags={
                            "error": type(exc).__name__,
                            "message": str(exc),
                            "target": self._config.engine_grpc_endpoint,
                            "attempts": attempt,
                            "max_retries": max_retries,
                            "err_code": getattr(exc, "err_code", None),
                        },
                    )
                    return None
                logger.warning(
                    "failed to fetch kv cache group metadata; retrying",
                    step="kv_metadata",
                    tags={
                        "error": type(exc).__name__,
                        "message": str(exc),
                        "target": self._config.engine_grpc_endpoint,
                        "retry_in_s": delay,
                        "attempt": attempt,
                        "max_retries": max_retries,
                        "err_code": getattr(exc, "err_code", None),
                    },
                )
                await asyncio.sleep(delay)
                delay = min(delay * 2, _METADATA_RETRY_MAX_S)
                continue
            metadata = _parse_kv_cache_group_metadata(payload)
            if logger.is_debug_enabled():
                logger.debug(
                    "fetched kv cache group metadata",
                    step="kv_metadata",
                    tags={
                        "group_count": len(metadata) if metadata else 0,
                        "groups": [
                            {
                                "kind": spec.kind,
                                "sliding_window": spec.sliding_window,
                            }
                            for spec in metadata
                        ]
                        if metadata
                        else None,
                    },
                )
            return metadata
        raise AssertionError("unreachable")

    async def close(self) -> None:
        """Release the DashLLM gRPC client and vLLM ZMQ sockets."""

        if self._closed:
            return
        self._closed = True
        try:
            await self._dashllm_grpc_client.close()
        finally:
            self._sub.close(linger=0)
            self._dealer.close(linger=0)

    async def reset_generation_state(self) -> None:
        """Clear sequence state and recreate sockets after engine recovery."""
        self._generation += 1
        self._last_seq = -1
        self._sub.close(linger=0)
        self._dealer.close(linger=0)
        self._sub = self._open_sub_socket()
        self._dealer = self._open_dealer_socket()

    def map_medium(self, medium: str | None) -> str:
        if medium is None:
            return ""
        return _KVCM_MEDIUM_MAP.get(medium, "")

    def supported_mediums(self) -> list[str]:
        return list(_KVCM_MEDIUM_MAP.values())

    def storage_type(self) -> str:
        return "ST_EVENT_REPORT"

    def location_spec_name(self, block_size: int) -> str:
        return f"vllm_{block_size}"

    def location_uri(self, host_ip_port: str, medium: str) -> str:
        return f"vllm://{host_ip_port}/{medium}"
