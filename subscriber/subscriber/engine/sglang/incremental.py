"""SGLang msgpack decoding on the shared ZMQ event transport."""

from __future__ import annotations

import time
from collections.abc import Mapping
from typing import Any

import msgspec

from subscriber import logger
from subscriber.config import DpEndpoint, SubscriberConfig
from subscriber.engine.debug import summarize_kv_event_batch_for_debug
from subscriber.engine.zmq_source import ZmqKvEventSource
from subscriber.types import (
    AllBlocksCleared,
    BlockRemoved,
    BlockSnapshot,
    BlockStored,
    ExternalBlockHash,
    KVEventBatch,
)


class _SglangWireEvent(
    msgspec.Struct,
    array_like=True,
    omit_defaults=True,
    gc=False,
    tag=True,
):
    pass


class _SglangWireBlockStored(_SglangWireEvent, tag="BlockStored"):
    block_hashes: list[ExternalBlockHash]
    parent_block_hash: ExternalBlockHash | None
    token_ids: list[int | tuple[int, int]]
    block_size: int
    lora_id: int | None
    component_type: str
    component_id: int | None = None
    snapshot_version: int = 0
    medium: str | None = None


class _SglangWireBlockRemoved(_SglangWireEvent, tag="BlockRemoved"):
    block_hashes: list[ExternalBlockHash]
    component_type: str
    component_id: int | None = None
    snapshot_version: int = 0
    medium: str | None = None


class _SglangWireAllBlocksCleared(_SglangWireEvent, tag="AllBlocksCleared"):
    pass


class _SglangWireBatch(msgspec.Struct, array_like=True, gc=False):
    ts: float
    events: list[
        _SglangWireBlockStored | _SglangWireBlockRemoved | _SglangWireAllBlocksCleared
    ]
    attn_dp_rank: int | None = None


_SUPPORTED_MEDIA = {"GPU", "CPU_PINNED"}
_DROP_LOG_INTERVAL_S = 60.0


class _DropLogLimiter:
    """Emit a first detailed drop log and one periodic aggregate per reason."""

    def __init__(self) -> None:
        self._last_log_s: dict[str, float] = {}
        self._suppressed_counts: dict[str, int] = {}

    def record(self, reason: str) -> int | None:
        """Return aggregated count when this drop should be logged."""

        self._suppressed_counts[reason] = self._suppressed_counts.get(reason, 0) + 1
        now_s = time.monotonic()
        last_log_s = self._last_log_s.get(reason)
        if last_log_s is not None and now_s - last_log_s < _DROP_LOG_INTERVAL_S:
            return None
        self._last_log_s[reason] = now_s
        count = self._suppressed_counts[reason]
        self._suppressed_counts[reason] = 0
        return count


class SglangIncrementalSource(ZmqKvEventSource):
    """Decode SGLang component events with descriptor-owned component mapping."""

    def __init__(
        self,
        config: SubscriberConfig,
        *,
        component_group_idxs: Mapping[int, int],
        endpoint: DpEndpoint,
    ) -> None:
        self._component_group_idxs = dict(component_group_idxs)
        self._decoder = msgspec.msgpack.Decoder(type=_SglangWireBatch)
        self._drop_log_limiter = _DropLogLimiter()
        super().__init__(config, endpoint=endpoint)

    @property
    def _logger(self) -> Any:
        return logger

    @property
    def _engine_label(self) -> str:
        return "SGLang"

    @property
    def _connect_log_message(self) -> str:
        return "connecting SGLang ZMQ sockets"

    @property
    def _live_message_log_message(self) -> str:
        return "received SGLang ZMQ live message"

    @property
    def _decoded_batch_log_message(self) -> str:
        return "decoded SGLang KV event batch"

    @property
    def _replay_request_log_message(self) -> str:
        return "requesting SGLang ZMQ replay"

    @property
    def _replay_complete_log_message(self) -> str:
        return "completed SGLang ZMQ replay"

    @property
    def _reset_log_message(self) -> str:
        return "reset SGLang ZMQ generation state; sockets recreated"

    def set_component_group_idxs(self, component_group_idxs: Mapping[int, int]) -> None:
        """Replace the descriptor-derived component map before subscription starts."""

        self._component_group_idxs = dict(component_group_idxs)

    def _summarize_batch(self, batch: KVEventBatch) -> dict[str, object]:
        return summarize_kv_event_batch_for_debug(batch)

    def _log_event_drop(
        self,
        reason: str,
        message: str,
        *,
        step: str,
        tags: dict[str, object],
    ) -> None:
        dropped_count = self._drop_log_limiter.record(reason)
        if dropped_count is not None:
            logger.warning(
                message,
                step=step,
                tags={**tags, "reason": reason, "dropped_count": dropped_count},
            )

    def _decode_payload(
        self,
        payload: bytes,
        *,
        step: str,
        tags: dict[str, object],
    ) -> KVEventBatch | None:
        try:
            wire_batch = self._decoder.decode(payload)
        except (msgspec.DecodeError, msgspec.ValidationError, TypeError) as exc:
            logger.warning(
                "failed to decode SGLang KV event msgpack payload",
                step=step,
                tags={
                    **tags,
                    "error": exc.__class__.__name__,
                    "message": str(exc),
                },
            )
            return None

        events: list[BlockStored | BlockRemoved | AllBlocksCleared | BlockSnapshot] = []
        for event in wire_batch.events:
            if isinstance(event, _SglangWireAllBlocksCleared):
                events.append(AllBlocksCleared())
                continue
            # SGLang publishes component_id. Map that engine-native identity to
            # the common pipeline group_idx consumed by the KVCM translation.
            component_id = event.component_id
            group_idx = (
                self._component_group_idxs.get(component_id)
                if component_id is not None
                else None
            )
            if group_idx is None:
                self._log_event_drop(
                    "unknown_component_type",
                    "dropping SGLang KV event with unknown component type",
                    step=step,
                    tags={
                        **tags,
                        "component_id": event.component_id,
                        "component_type": event.component_type,
                    },
                )
                continue
            if event.medium not in _SUPPORTED_MEDIA:
                self._log_event_drop(
                    "unsupported_medium",
                    "dropping SGLang KV event with unsupported medium",
                    step=step,
                    tags={**tags, "medium": event.medium},
                )
                continue
            if isinstance(event, _SglangWireBlockStored):
                token_ids = [
                    token
                    for value in event.token_ids
                    for token in (value if isinstance(value, tuple) else (value,))
                ]
                events.append(
                    BlockStored(
                        block_hashes=event.block_hashes,
                        parent_block_hash=event.parent_block_hash,
                        token_ids=token_ids,
                        block_size=event.block_size,
                        lora_id=event.lora_id,
                        medium=event.medium,
                        lora_name=None,
                        group_idx=group_idx,
                        component_id=event.component_id,
                        kv_cache_spec_kind=event.component_type,
                        snapshot_version=event.snapshot_version,
                    )
                )
            else:
                events.append(
                    BlockRemoved(
                        block_hashes=event.block_hashes,
                        medium=event.medium,
                        group_idx=group_idx,
                        component_id=event.component_id,
                        remaining_copy_counts=None,
                        snapshot_version=event.snapshot_version,
                    )
                )
        if not events:
            return None
        return KVEventBatch(
            ts=wire_batch.ts,
            events=events,
            data_parallel_rank=wire_batch.attn_dp_rank,
        )
