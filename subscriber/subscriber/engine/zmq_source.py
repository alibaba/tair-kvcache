"""Shared ZMQ incremental-event transport for engine adapters."""

from __future__ import annotations

import asyncio
import time
from abc import ABC, abstractmethod
from collections.abc import AsyncGenerator, Callable, Mapping
from typing import Any

import zmq
import zmq.asyncio

from subscriber import logger
from subscriber.config import DpEndpoint, SubscriberConfig
from subscriber.engine.base import EngineEventBatch
from subscriber.metrics import BatchTelemetry, report_zmq_message
from subscriber.trace import generate_trace_id
from subscriber.types import KVEventBatch

_END_SEQ = (-1).to_bytes(8, "big", signed=True)
_RECV_FAILURE_WARN_INTERVAL_S = 5.0
_RECV_FAILURE_BACKOFF_S = 0.1


class _ZmqSequenceGapDiagnostics:
    """Accumulate best-effort SUB queue observations until a sequence gap."""

    def __init__(
        self,
        *,
        state_reader: Callable[[], Mapping[str, object]],
    ) -> None:
        self._state_reader = state_reader
        self._received_message_bytes = 0
        self._queue_nonempty_observation_count = 0

    def record_message(
        self,
        *,
        message_bytes: int,
        queue_nonempty_after_receive: bool,
    ) -> None:
        self._received_message_bytes += message_bytes
        if queue_nonempty_after_receive:
            self._queue_nonempty_observation_count += 1

    def snapshot_and_reset(self) -> dict[str, object]:
        tags: dict[str, object] = {}
        try:
            tags.update(self._state_reader())
        except Exception as exc:
            tags["zmq_queue_state_error"] = type(exc).__name__
        tags.update(
            {
                "zmq_received_message_bytes": self._received_message_bytes,
                "zmq_queue_nonempty_observation_count": (
                    self._queue_nonempty_observation_count
                ),
            }
        )
        self._received_message_bytes = 0
        self._queue_nonempty_observation_count = 0
        return tags


class ZmqKvEventSource(ABC):
    """Own the engine-independent ZMQ live/replay state machine.

    Subclasses decode their own msgpack schema while this class keeps the
    generation-local sequence/replay contract shared by vLLM and SGLang.
    """

    def __init__(
        self,
        config: SubscriberConfig,
        *,
        endpoint: DpEndpoint,
    ) -> None:
        self._config = config
        self._endpoint = endpoint
        self._ctx = zmq.asyncio.Context.instance()
        self._last_seq = -1
        self._generation = 0
        self._closed = False
        self._recv_failure_count = 0
        self._last_recv_warn_s: float | None = None
        self._logger.info(
            self._connect_log_message,
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
        self._zmq_gap_diagnostics = _ZmqSequenceGapDiagnostics(
            state_reader=self._zmq_queue_state,
        )

    @property
    def _logger(self) -> Any:
        return logger

    @property
    def _engine_label(self) -> str:
        return "engine"

    @property
    def _connect_log_message(self) -> str:
        return "connecting engine ZMQ sockets"

    @property
    def _live_message_log_message(self) -> str:
        return "received engine ZMQ live message"

    @property
    def _decoded_batch_log_message(self) -> str:
        return "decoded engine KV event batch"

    @property
    def _replay_request_log_message(self) -> str:
        return "requesting engine ZMQ replay"

    @property
    def _replay_complete_log_message(self) -> str:
        return "completed engine ZMQ replay"

    @property
    def _reset_log_message(self) -> str:
        return "reset engine ZMQ generation state; sockets recreated"

    def _new_trace_id(self) -> str:
        return generate_trace_id()

    def _report_zmq_message(self) -> None:
        report_zmq_message()

    @abstractmethod
    def _decode_payload(
        self,
        payload: bytes,
        *,
        step: str,
        tags: dict[str, object],
    ) -> KVEventBatch | None:
        """Decode one engine payload into the subscriber's internal event batch."""

    def _summarize_batch(self, batch: KVEventBatch) -> dict[str, object]:
        return {
            "event_count": len(batch.events),
            "data_parallel_rank": batch.data_parallel_rank,
        }

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

    async def subscribe(self) -> AsyncGenerator[EngineEventBatch, None]:
        """Yield replayed batches before the live batch that exposed a gap."""

        try:
            while True:
                received = await self._recv_live_message()
                if received is None:
                    continue
                seq, payload, trace_id = received
                generation = self._generation

                if seq > self._last_seq + 1:
                    missed = seq - self._last_seq - 1
                    replay_trace_id = self._new_trace_id()
                    diagnostic_tags = self._zmq_gap_diagnostics.snapshot_and_reset()
                    self._logger.warning(
                        "kv event sequence gap detected, triggering replay",
                        step="zmq_replay",
                        tags={
                            "last_seq": self._last_seq,
                            "current_seq": seq,
                            "missed": missed,
                            "trace_id": replay_trace_id,
                            **diagnostic_tags,
                        },
                    )
                    replay_telemetry = BatchTelemetry(pipeline="incremental")
                    replay_telemetry.count("zmq_sequence_gap_count", 1)
                    replay_telemetry.count("zmq_missed_message_count", missed)
                    replay_batches = await self._replay_missing_batches(
                        seq, generation, replay_trace_id
                    )
                    if generation != self._generation:
                        continue
                    if replay_batches is not None:
                        self._last_seq = seq - 1
                    if replay_batches:
                        replay_telemetry.mark("replay_fetch")
                        yield EngineEventBatch(
                            replay_batches,
                            replay_telemetry,
                            trace_id=replay_trace_id,
                        )
                    elif replay_batches == []:
                        self._logger.warning(
                            "replay returned no batches, publisher buffer may "
                            "have been pruned",
                            step="zmq_replay",
                            tags={
                                "gap_start_seq": self._last_seq + 1,
                                "trace_id": replay_trace_id,
                            },
                        )

                telemetry = BatchTelemetry(pipeline="incremental")
                batch = self._decode_payload(
                    payload,
                    step="zmq_subscribe",
                    tags={"seq": seq, "trace_id": trace_id},
                )
                if batch is None:
                    continue
                telemetry.mark("decode")
                if self._logger.is_debug_enabled():
                    self._logger.debug(
                        self._decoded_batch_log_message,
                        step="zmq_subscribe",
                        tags={
                            "seq": seq,
                            "trace_id": trace_id,
                            **self._summarize_batch(batch),
                        },
                    )
                if generation != self._generation:
                    continue
                self._last_seq = seq
                yield EngineEventBatch([batch], telemetry, trace_id=trace_id)
        finally:
            self._sub.close(linger=0)
            self._dealer.close(linger=0)

    async def _recv_live_message(self) -> tuple[int, bytes, str] | None:
        generation = self._generation
        sub = self._sub
        try:
            frames = await sub.recv_multipart()
        except Exception as exc:
            self._recv_failure_count += 1
            now_s = time.monotonic()
            if (
                self._last_recv_warn_s is None
                or now_s - self._last_recv_warn_s >= _RECV_FAILURE_WARN_INTERVAL_S
            ):
                self._last_recv_warn_s = now_s
                self._logger.warning(
                    "failed to receive kv event message",
                    step="zmq_subscribe",
                    tags={
                        "error": exc.__class__.__name__,
                        "message": str(exc),
                        "consecutive_failures": self._recv_failure_count,
                    },
                )
            await asyncio.sleep(_RECV_FAILURE_BACKOFF_S)
            return None
        if self._recv_failure_count > 0:
            self._logger.info(
                "kv event message receive recovered",
                step="zmq_subscribe",
                tags={"previous_consecutive_failures": self._recv_failure_count},
            )
            self._recv_failure_count = 0
            self._last_recv_warn_s = None
        if generation != self._generation:
            return None
        self._report_zmq_message()
        self._zmq_gap_diagnostics.record_message(
            message_bytes=sum(len(frame) for frame in frames),
            queue_nonempty_after_receive=self._sub_queue_is_readable(sub),
        )
        if len(frames) != 3:
            self._logger.warning(
                "dropping malformed kv event message frames",
                step="zmq_subscribe",
                tags={"frame_count": len(frames)},
            )
            return None
        topic, seq_bytes, payload = frames
        if len(seq_bytes) != 8:
            self._logger.warning(
                "dropping kv event message with invalid sequence frame",
                step="zmq_subscribe",
                tags={"seq_frame_len": len(seq_bytes)},
            )
            return None
        seq = int.from_bytes(seq_bytes, "big")
        trace_id = self._new_trace_id()
        if self._logger.is_debug_enabled():
            self._logger.debug(
                self._live_message_log_message,
                step="zmq_subscribe",
                tags={
                    "topic": topic.decode(errors="replace"),
                    "seq": seq,
                    "payload_bytes": len(payload),
                    "trace_id": trace_id,
                },
            )
        return seq, payload, trace_id

    def _zmq_queue_state(self) -> dict[str, bool | int]:
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
        try:
            events = sub.getsockopt(zmq.EVENTS)
            return isinstance(events, int) and bool(events & zmq.POLLIN)
        except Exception:
            return False

    async def _replay_missing_batches(
        self, current_seq: int, generation: int, trace_id: str
    ) -> list[KVEventBatch] | None:
        gap_start_seq = self._last_seq + 1
        if self._logger.is_debug_enabled():
            self._logger.debug(
                self._replay_request_log_message,
                step="zmq_replay",
                tags={
                    "gap_start_seq": gap_start_seq,
                    "current_seq": current_seq,
                    "trace_id": trace_id,
                },
            )
        dealer = self._dealer
        try:
            async with asyncio.timeout(self._config.zmq_replay_timeout_s):
                await dealer.send_multipart([b"", gap_start_seq.to_bytes(8, "big")])
        except TimeoutError:
            self._replace_replay_socket(dealer, generation)
            self._log_replay_unavailable(
                gap_start_seq,
                current_seq,
                error="TimeoutError",
                message="replay timed out",
                trace_id=trace_id,
            )
            return None
        except Exception as exc:
            self._replace_replay_socket(dealer, generation)
            self._log_replay_unavailable(
                gap_start_seq,
                current_seq,
                error=exc.__class__.__name__,
                message=str(exc),
                trace_id=trace_id,
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
                self._log_replay_unavailable(
                    gap_start_seq,
                    current_seq,
                    error="TimeoutError",
                    message="replay timed out",
                    trace_id=trace_id,
                )
                return None
            except Exception as exc:
                self._replace_replay_socket(dealer, generation)
                self._log_replay_unavailable(
                    gap_start_seq,
                    current_seq,
                    error=exc.__class__.__name__,
                    message=str(exc),
                    trace_id=trace_id,
                )
                return None
            if generation != self._generation:
                return None
            if len(frames) == 4:
                _, _, replay_seq_bytes, replay_payload = frames
            elif len(frames) == 3:
                _, replay_seq_bytes, replay_payload = frames
            else:
                self._replace_replay_socket(dealer, generation)
                self._logger.warning(
                    "dropping malformed kv event replay frames",
                    step="zmq_replay",
                    tags={
                        "gap_start_seq": gap_start_seq,
                        "current_seq": current_seq,
                        "frame_count": len(frames),
                        "trace_id": trace_id,
                    },
                )
                return None
            if replay_seq_bytes == _END_SEQ:
                if self._logger.is_debug_enabled():
                    self._logger.debug(
                        self._replay_complete_log_message,
                        step="zmq_replay",
                        tags={
                            "gap_start_seq": gap_start_seq,
                            "current_seq": current_seq,
                            "batch_count": len(replay_batches),
                            "trace_id": trace_id,
                        },
                    )
                return replay_batches
            replay_seq = int.from_bytes(replay_seq_bytes, "big")
            if replay_seq < gap_start_seq or replay_seq >= current_seq:
                continue
            batch = self._decode_payload(
                replay_payload,
                step="zmq_replay",
                tags={
                    "gap_start_seq": gap_start_seq,
                    "current_seq": current_seq,
                    "trace_id": trace_id,
                },
            )
            if batch is None:
                self._replace_replay_socket(dealer, generation)
                return None
            if self._logger.is_debug_enabled():
                self._logger.debug(
                    self._decoded_batch_log_message,
                    step="zmq_replay",
                    tags={
                        "gap_start_seq": gap_start_seq,
                        "current_seq": current_seq,
                        "replay_seq": replay_seq,
                        "trace_id": trace_id,
                        **self._summarize_batch(batch),
                    },
                )
            replay_batches.append(batch)

    def _log_replay_unavailable(
        self,
        gap_start_seq: int,
        current_seq: int,
        *,
        error: str,
        message: str,
        trace_id: str,
    ) -> None:
        self._logger.warning(
            "kv event replay unavailable; sequence gap remains; forwarding live batch",
            step="zmq_replay",
            tags={
                "gap_start_seq": gap_start_seq,
                "current_seq": current_seq,
                "error": error,
                "message": message,
                "trace_id": trace_id,
            },
        )

    async def close(self) -> None:
        """Close the current ZMQ sockets once."""

        if self._closed:
            return
        self._closed = True
        self._sub.close(linger=0)
        self._dealer.close(linger=0)

    async def reset_generation_state(self) -> None:
        """Invalidate in-flight waits and recreate the ZMQ sockets."""

        self._generation += 1
        self._last_seq = -1
        self._sub.close(linger=0)
        self._dealer.close(linger=0)
        self._sub = self._open_sub_socket()
        self._dealer = self._open_dealer_socket()
        self._logger.info(
            self._reset_log_message,
            step="zmq_connect",
            tags={
                "generation": self._generation,
                "pub_endpoint": self._endpoint.zmq_pub_endpoint,
                "replay_endpoint": self._endpoint.zmq_replay_endpoint,
            },
        )
