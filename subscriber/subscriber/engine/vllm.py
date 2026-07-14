from __future__ import annotations

import asyncio
from collections import Counter
from collections.abc import AsyncGenerator

import httpx
import zmq
import zmq.asyncio

from subscriber import logger
from subscriber.config import SubscriberConfig
from subscriber.engine.base import AbstractEngineAdapter
from subscriber.health.events import LivenessEvent
from subscriber.types import BlockRemoved, BlockStored, KVEventBatch
from subscriber.utils.msgpack_helper import KVEventBatchMsgpackHelper

_END_SEQ = (-1).to_bytes(8, "big", signed=True)
_MAX_DEBUG_BLOCK_HASHES = 32

MEDIUM_VLLM_GPU = "GPU"
MEDIUM_VLLM_CPU = "CPU"

_KVCM_MEDIUM_MAP = {
    MEDIUM_VLLM_GPU: "hbm",
    MEDIUM_VLLM_CPU: "mem",
}


def _summarize_batch(batch: KVEventBatch) -> dict[str, object]:
    event_type_counts = Counter(type(event).__name__ for event in batch.events)
    stored_block_hash_count = 0
    removed_block_hash_count = 0
    stored_block_hashes: list[int] = []
    removed_block_hashes: list[int] = []
    for event in batch.events:
        if isinstance(event, BlockStored):
            for block_hash in event.block_hashes:
                stored_block_hash_count += 1
                if len(stored_block_hashes) < _MAX_DEBUG_BLOCK_HASHES:
                    stored_block_hashes.append(block_hash)
        elif isinstance(event, BlockRemoved):
            for block_hash in event.block_hashes:
                removed_block_hash_count += 1
                if len(removed_block_hashes) < _MAX_DEBUG_BLOCK_HASHES:
                    removed_block_hashes.append(block_hash)
    return {
        "event_count": len(batch.events),
        "event_types": ",".join(
            f"{name}:{count}" for name, count in sorted(event_type_counts.items())
        ),
        "data_parallel_rank": batch.data_parallel_rank,
        "stored_block_hash_count": stored_block_hash_count,
        "stored_block_hashes": stored_block_hashes,
        "stored_block_hashes_truncated": stored_block_hash_count
        > len(stored_block_hashes),
        "removed_block_hash_count": removed_block_hash_count,
        "removed_block_hashes": removed_block_hashes,
        "removed_block_hashes_truncated": removed_block_hash_count
        > len(removed_block_hashes),
    }


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
            tags={"error": type(exc).__name__, "url": url},
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
    reported through an HTTP /health polling loop.
    """

    def __init__(self, config: SubscriberConfig) -> None:
        """Open the vLLM subscription and replay sockets from subscriber config."""

        self._config = config
        self._endpoint = config.dp_endpoints[0]
        self._ctx = zmq.asyncio.Context.instance()
        self._msgpack_helper = KVEventBatchMsgpackHelper()
        self._last_seq = -1
        self._generation = 0
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

    async def subscribe_kv_events(self) -> AsyncGenerator[list[KVEventBatch], None]:
        """Subscribe to vLLM KV events and repair sequence gaps with replay.

        The generator yields replayed batches first when a gap is detected, then
        yields the live batch that exposed the gap. Sequence tracking is local
        to this adapter generation and is reset by ``reset_generation_state``.
        """

        try:
            while True:
                received = await self._recv_live_message()
                if received is None:
                    continue
                seq, payload = received
                generation = self._generation

                if seq > self._last_seq + 1:
                    missed = seq - self._last_seq - 1
                    logger.warning(
                        "kv event sequence gap detected, triggering replay",
                        step="zmq_replay",
                        tags={
                            "last_seq": self._last_seq,
                            "current_seq": seq,
                            "missed": missed,
                        },
                    )
                    replay_batches = await self._replay_missing_batches(seq, generation)
                    if generation != self._generation:
                        continue
                    if replay_batches:
                        yield replay_batches
                    elif replay_batches == []:
                        logger.warning(
                            "replay returned no batches, "
                            "publisher buffer may have been pruned",
                            step="zmq_replay",
                            tags={"gap_start_seq": self._last_seq + 1},
                        )

                batch = self._msgpack_helper.decode(
                    payload,
                    step="zmq_subscribe",
                    tags={"seq": seq},
                )
                if batch is None:
                    continue
                if logger.is_debug_enabled():
                    logger.debug(
                        "decoded vLLM KV event batch",
                        step="zmq_subscribe",
                        tags={"seq": seq, **_summarize_batch(batch)},
                    )
                if generation != self._generation:
                    continue
                self._last_seq = seq
                yield [batch]
        finally:
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
                        await asyncio.sleep(self._config.engine_health_interval_s)
                        yield event
            except Exception as exc:
                logger.warning(
                    "engine health watch loop failed unexpectedly",
                    step="engine_health",
                    tags={"error": exc.__class__.__name__, "message": str(exc)},
                )
                await asyncio.sleep(self._config.engine_health_interval_s)
                yield LivenessEvent.UNHEALTHY

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
