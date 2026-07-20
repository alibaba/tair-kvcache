"""Integration test: real ZMQ sockets, no mocks.

Exercises VllmAdapter against a FakeVllmPublisher that reproduces the wire
protocol of vllm.distributed.kv_events.ZmqEventPublisher (PUB 3-frame
multipart with msgspec.msgpack payloads, ROUTER/DEALER replay with an
8-byte big-endian sequence number and END_SEQ sentinel) without depending
on the vllm package itself.
"""

from __future__ import annotations

import asyncio
import contextlib
from unittest.mock import MagicMock

import msgspec
import pytest
import zmq
import zmq.asyncio

from subscriber.config import SubscriberConfig
from subscriber.engine.vllm import VllmAdapter
from subscriber.types import AllBlocksCleared, BlockStored, KVEventBatch

pytestmark = pytest.mark.integration


class FakeVllmPublisher:
    """Minimal stand-in for ZmqEventPublisher that speaks its exact wire
    protocol over real ZMQ sockets."""

    END_SEQ = (-1).to_bytes(8, "big", signed=True)

    def __init__(self, topic: str = "") -> None:
        self._ctx = zmq.asyncio.Context.instance()
        self._pub: zmq.asyncio.Socket = self._ctx.socket(zmq.PUB)
        self._pub.bind("tcp://127.0.0.1:0")
        self._router: zmq.asyncio.Socket = self._ctx.socket(zmq.ROUTER)
        self._router.bind("tcp://127.0.0.1:0")
        self._topic_bytes = topic.encode("utf-8")
        self._buffer: list[tuple[int, bytes]] = []
        self._replay_task: asyncio.Task[None] | None = None

    @property
    def pub_endpoint(self) -> str:
        return self._pub.getsockopt_string(zmq.LAST_ENDPOINT)

    @property
    def replay_endpoint(self) -> str:
        return self._router.getsockopt_string(zmq.LAST_ENDPOINT)

    def start_replay_server(self) -> None:
        self._replay_task = asyncio.create_task(self._serve_replay_loop())

    async def _serve_replay_loop(self) -> None:
        while True:
            client_id, _, start_seq_bytes = await self._router.recv_multipart()
            start_seq = int.from_bytes(start_seq_bytes, "big")
            for seq, payload in self._buffer:
                if seq >= start_seq:
                    await self._router.send_multipart(
                        [client_id, b"", seq.to_bytes(8, "big"), payload]
                    )
            await self._router.send_multipart([client_id, b"", self.END_SEQ, b""])

    async def publish(self, seq: int, batch: KVEventBatch) -> None:
        """Send live on the wire and record in the replay buffer."""
        payload = msgspec.msgpack.encode(batch)
        self._buffer.append((seq, payload))
        await self._pub.send_multipart(
            [self._topic_bytes, seq.to_bytes(8, "big"), payload]
        )

    def record_dropped(self, seq: int, batch: KVEventBatch) -> None:
        """Simulate a message lost in transit: kept in the replay buffer
        only, never sent live, so a later gap must be filled via replay."""
        payload = msgspec.msgpack.encode(batch)
        self._buffer.append((seq, payload))

    async def publish_until_delivered(
        self,
        seq: int,
        batch: KVEventBatch,
        delivered: asyncio.Event,
        interval: float = 0.02,
    ) -> None:
        """Work around the PUB/SUB slow-joiner race by resending until the
        subscriber confirms receipt. Safe because ZMQ PUB never queues a
        message for a socket that connects after it was sent, so at most
        one resend actually reaches the subscriber."""
        payload = msgspec.msgpack.encode(batch)
        if not any(existing_seq == seq for existing_seq, _ in self._buffer):
            self._buffer.append((seq, payload))
        while not delivered.is_set():
            await self._pub.send_multipart(
                [self._topic_bytes, seq.to_bytes(8, "big"), payload]
            )
            with contextlib.suppress(TimeoutError):
                await asyncio.wait_for(delivered.wait(), timeout=interval)

    async def close(self) -> None:
        if self._replay_task is not None:
            self._replay_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._replay_task
        self._pub.close(linger=0)
        self._router.close(linger=0)


async def _stop_consumer(consumer: asyncio.Task[None]) -> None:
    consumer.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await consumer


async def test_real_zmq_publisher_delivers_single_event() -> None:
    publisher = FakeVllmPublisher()
    config = SubscriberConfig(
        zmq_pub_endpoint=publisher.pub_endpoint,
        zmq_replay_endpoint=publisher.replay_endpoint,
    )
    adapter = VllmAdapter(config)

    batch = KVEventBatch(
        ts=1.0,
        events=[
            BlockStored(
                block_hashes=[1, 2],
                parent_block_hash=None,
                token_ids=[10, 20, 30],
                block_size=16,
                lora_id=None,
                medium="GPU",
                lora_name=None,
            )
        ],
    )

    delivered = asyncio.Event()
    results: list[list[KVEventBatch]] = []

    async def _consume() -> None:
        async for received in adapter.subscribe_kv_events():
            results.append(received.batches)
            delivered.set()
            break

    consumer = asyncio.create_task(_consume())
    try:
        await asyncio.wait_for(
            publisher.publish_until_delivered(0, batch, delivered), timeout=5.0
        )
        await asyncio.wait_for(consumer, timeout=5.0)
    finally:
        await _stop_consumer(consumer)
        await publisher.close()

    assert results == [[batch]]


async def test_sequence_gap_triggers_real_replay() -> None:
    publisher = FakeVllmPublisher()
    publisher.start_replay_server()
    config = SubscriberConfig(
        zmq_pub_endpoint=publisher.pub_endpoint,
        zmq_replay_endpoint=publisher.replay_endpoint,
    )
    adapter = VllmAdapter(config)

    batch0 = KVEventBatch(ts=1.0, events=[AllBlocksCleared()])
    dropped_batch1 = KVEventBatch(ts=2.0, events=[AllBlocksCleared()])
    batch2 = KVEventBatch(ts=3.0, events=[AllBlocksCleared()])

    delivered = asyncio.Event()
    results: list[list[KVEventBatch]] = []

    async def _consume() -> None:
        async for received in adapter.subscribe_kv_events():
            results.append(received.batches)
            delivered.set()
            if len(results) >= 3:
                break

    consumer = asyncio.create_task(_consume())
    try:
        await asyncio.wait_for(
            publisher.publish_until_delivered(0, batch0, delivered), timeout=5.0
        )
        publisher.record_dropped(1, dropped_batch1)
        await publisher.publish(2, batch2)
        await asyncio.wait_for(consumer, timeout=5.0)
    finally:
        await _stop_consumer(consumer)
        await publisher.close()

    # Replay has no upper bound: the publisher's buffer returns every batch
    # with seq >= start_seq, so it includes both the dropped batch and the
    # one that triggered the gap. The adapter then also re-yields the
    # triggering live message on its own.
    assert results == [[batch0], [dropped_batch1, batch2], [batch2]]


async def test_real_zmq_backlog_is_recorded_after_one_message_is_received() -> None:
    publisher = FakeVllmPublisher()
    config = SubscriberConfig(
        zmq_pub_endpoint=publisher.pub_endpoint,
        zmq_replay_endpoint=publisher.replay_endpoint,
    )
    adapter = VllmAdapter(config)
    batch = KVEventBatch(ts=1.0, events=[AllBlocksCleared()])
    warmup_received = asyncio.Event()

    async def _receive_warmup() -> tuple[int, bytes] | None:
        received = await adapter._recv_live_message()
        warmup_received.set()
        return received

    warmup = asyncio.create_task(_receive_warmup())
    try:
        await asyncio.wait_for(
            publisher.publish_until_delivered(0, batch, warmup_received), timeout=5.0
        )
        assert (await asyncio.wait_for(warmup, timeout=5.0))[0] == 0

        queue_metrics = MagicMock()
        adapter._zmq_queue_metrics = queue_metrics
        await publisher.publish(1, batch)
        await publisher.publish(2, batch)
        await asyncio.sleep(0.05)

        assert (await adapter._recv_live_message())[0] == 1
        assert (await adapter._recv_live_message())[0] == 2
    finally:
        await _stop_consumer(warmup)
        adapter._sub.close(linger=0)
        adapter._dealer.close(linger=0)
        await publisher.close()

    assert (
        queue_metrics.record_message.call_args_list[0].kwargs[
            "queue_nonempty_after_receive"
        ]
        is True
    )
