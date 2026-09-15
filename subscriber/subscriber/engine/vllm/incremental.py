from __future__ import annotations

from typing import Any

from subscriber import logger
from subscriber.config import DpEndpoint, SubscriberConfig
from subscriber.engine.debug import summarize_kv_event_batch_for_debug
from subscriber.engine.zmq_source import ZmqKvEventSource
from subscriber.metrics import report_zmq_message
from subscriber.trace import generate_trace_id
from subscriber.types import (
    AllBlocksCleared,
    BlockRemoved,
    BlockSnapshot,
    BlockStored,
    KVEventBatch,
)
from subscriber.utils.msgpack_helper import KVEventBatchMsgpackHelper


class VllmIncrementalSource(ZmqKvEventSource):
    """Decode vLLM payloads on the shared ZMQ live/replay transport."""

    def __init__(
        self,
        config: SubscriberConfig,
        *,
        endpoint: DpEndpoint,
        valid_component_ids: set[int] | None = None,
    ) -> None:
        self._msgpack_helper = KVEventBatchMsgpackHelper()
        self._valid_component_ids = valid_component_ids
        super().__init__(config, endpoint=endpoint)

    @property
    def _logger(self) -> Any:
        return logger

    @property
    def _engine_label(self) -> str:
        return "vLLM"

    @property
    def _connect_log_message(self) -> str:
        return "connecting vLLM ZMQ sockets"

    @property
    def _live_message_log_message(self) -> str:
        return "received vLLM ZMQ live message"

    @property
    def _decoded_batch_log_message(self) -> str:
        return "decoded vLLM KV event batch"

    @property
    def _replay_request_log_message(self) -> str:
        return "requesting vLLM ZMQ replay"

    @property
    def _replay_complete_log_message(self) -> str:
        return "completed vLLM ZMQ replay"

    @property
    def _reset_log_message(self) -> str:
        return "reset vLLM ZMQ generation state; sockets recreated"

    def _new_trace_id(self) -> str:
        return generate_trace_id()

    def _report_zmq_message(self) -> None:
        report_zmq_message()

    def _decode_payload(
        self,
        payload: bytes,
        *,
        step: str,
        tags: dict[str, object],
    ) -> KVEventBatch | None:
        batch = self._msgpack_helper.decode(payload, step=step, tags=tags)
        if batch is None or self._valid_component_ids is None:
            return batch
        filtered_events: list[
            BlockStored | BlockRemoved | AllBlocksCleared | BlockSnapshot
        ] = []
        for event in batch.events:
            if not isinstance(event, (BlockStored, BlockRemoved)):
                filtered_events.append(event)
                continue
            # vLLM's engine-native event identity is group_idx. Bootstrap uses
            # the engine-neutral name component_id for that same numeric ID.
            group_idx = event.group_idx
            if group_idx is None:
                logger.warning(
                    "dropping vLLM KV event without group_idx",
                    step=step,
                    tags=tags,
                )
                continue
            if group_idx not in self._valid_component_ids:
                logger.warning(
                    "dropping vLLM KV event with unknown group_idx",
                    step=step,
                    tags={**tags, "group_idx": group_idx},
                )
                continue
            filtered_events.append(event)
        if not filtered_events:
            return None
        if len(filtered_events) == len(batch.events):
            return batch
        return KVEventBatch(
            ts=batch.ts,
            events=filtered_events,
            data_parallel_rank=batch.data_parallel_rank,
        )

    def _summarize_batch(self, batch: KVEventBatch) -> dict[str, object]:
        return summarize_kv_event_batch_for_debug(batch)
