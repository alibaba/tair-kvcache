from __future__ import annotations

import asyncio
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Protocol

from .key_codec import to_signed_i64
from .models import BlockRecord, EngineUpdate, SourceNotReady


class VllmHistoryGap(RuntimeError):
    """The bounded vLLM replay buffer cannot reconstruct authoritative state."""


class VllmTransport(Protocol):
    async def next_sequences(self, expected_sequence: int) -> list[tuple[int, bytes]]: ...

    async def close(self) -> None: ...


@dataclass(frozen=True)
class _CommitToken:
    cursor: int
    blocks: dict[tuple[int, str], frozenset[int]]
    full_snapshot: bool


class VllmCacheSource:
    """Transactional consumer for vLLM's sequenced MsgPack event stream."""

    def __init__(
        self,
        pub_endpoint: str,
        replay_endpoint: str,
        *,
        topic: str = "",
        replay_timeout_s: float = 1.0,
        full_refresh_interval_s: float = 300.0,
        expected_block_size: int | None = None,
        cache_group_count: int = 1,
        transport: VllmTransport | None = None,
        decoder: Callable[[bytes], Any] | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if expected_block_size is not None and expected_block_size <= 0:
            raise ValueError("expected_block_size must be positive")
        if cache_group_count <= 0:
            raise ValueError("cache_group_count must be positive")
        self._transport = transport or _ZmqVllmTransport(
            pub_endpoint, replay_endpoint, topic, replay_timeout_s
        )
        self._decoder = decoder or _MsgspecDecoder()
        self._clock = clock
        self._full_refresh_interval_s = full_refresh_interval_s
        self._expected_block_size = expected_block_size
        self._cache_group_count = cache_group_count
        self._cursor = -1
        self._blocks: dict[tuple[int, str], frozenset[int]] = {}
        self._authoritative = False
        self._next_full_refresh = 0.0
        self._pending: _CommitToken | None = None

    async def prepare(self) -> EngineUpdate:
        if self._pending is not None:
            raise RuntimeError("previous vLLM update has not been committed or aborted")
        messages = await self._transport.next_sequences(self._cursor + 1)
        if not messages:
            if not self._authoritative:
                raise SourceNotReady("vLLM has not provided an authoritative baseline")
            full = self._clock() >= self._next_full_refresh
            token = _CommitToken(self._cursor, self._blocks, full)
            self._pending = token
            if full:
                return EngineUpdate(True, blocks=_records(self._blocks), commit_token=token)
            return EngineUpdate(False, commit_token=token)
        candidate = {key: set(groups) for key, groups in self._blocks.items()}
        cursor = self._cursor
        authoritative = self._authoritative
        clear_seen = False
        for sequence, payload in messages:
            batch = self._decoder(payload)
            has_clear = any(
                getattr(event, "event_type", type(event).__name__)
                == "AllBlocksCleared"
                for event in batch.events
            )
            if sequence == 0 and cursor >= 0:
                # vLLM sequence numbers are process-local. Seeing zero after a
                # committed positive cursor is an explicit publisher epoch
                # change, so the old process state must not survive.
                candidate.clear()
                authoritative = True
                clear_seen = True
                cursor = -1
            if sequence != cursor + 1:
                if not has_clear:
                    raise VllmHistoryGap(
                        f"vLLM replay gap: expected={cursor + 1}, got={sequence}"
                    )
                # AllBlocksCleared is an authoritative reset marker. It makes
                # events before this sequence irrelevant, so recovery does not
                # require the pruned replay prefix.
                candidate.clear()
                authoritative = True
                cursor = sequence - 1
            for event in batch.events:
                event_name = getattr(event, "event_type", type(event).__name__)
                if event_name == "AllBlocksCleared":
                    candidate.clear()
                    authoritative = True
                    clear_seen = True
                elif event_name == "BlockStored":
                    block_size = getattr(event, "block_size", None)
                    if (
                        self._expected_block_size is not None
                        and block_size is not None
                        and int(block_size) != self._expected_block_size
                    ):
                        raise RuntimeError(
                            "vLLM cache block size does not match KVCM registration: "
                            f"configured={self._expected_block_size}, observed={block_size}"
                        )
                    medium = _medium(getattr(event, "medium", None))
                    group = _group(
                        getattr(event, "group_idx", None), self._cache_group_count
                    )
                    for block_hash in event.block_hashes:
                        candidate.setdefault((to_signed_i64(block_hash), medium), set()).add(group)
                elif event_name == "BlockRemoved":
                    medium = _medium(getattr(event, "medium", None))
                    group = _group(
                        getattr(event, "group_idx", None), self._cache_group_count
                    )
                    counts = getattr(event, "remaining_copy_counts", None)
                    if counts is not None and len(counts) != len(event.block_hashes):
                        raise RuntimeError(
                            "vLLM BlockRemoved remaining_copy_counts length does not "
                            "match block_hashes"
                        )
                    for index, block_hash in enumerate(event.block_hashes):
                        if counts is not None and index < len(counts) and counts[index] > 0:
                            continue
                        key = (to_signed_i64(block_hash), medium)
                        groups = candidate.get(key)
                        if groups is not None:
                            groups.discard(group)
                            if not groups:
                                candidate.pop(key, None)
                else:
                    raise RuntimeError(f"unknown vLLM cache event: {event_name}")
            cursor = sequence

        # A sequence-zero replay is the only cold-start proof available from
        # vLLM unless the stream explicitly resets all blocks.
        if self._cursor == -1 and messages[0][0] == 0:
            authoritative = True
        if not authoritative:
            raise VllmHistoryGap(
                "vLLM replay history was pruned before an authoritative baseline"
            )

        frozen = {key: frozenset(groups) for key, groups in candidate.items()}
        full = (
            not self._authoritative
            or clear_seen
            or self._clock() >= self._next_full_refresh
        )
        token = _CommitToken(cursor, frozen, full)
        self._pending = token
        if full:
            return EngineUpdate(
                True,
                blocks=_records(frozen),
                commit_token=token,
            )

        changed = {key for key, groups in frozen.items() if self._blocks.get(key) != groups}
        removed = set(self._blocks) - set(frozen)
        return EngineUpdate(
            False,
            upserts=_records({key: frozen[key] for key in changed}),
            removals=_records({key: self._blocks[key] for key in removed}),
            commit_token=token,
        )

    def commit(self, update: EngineUpdate) -> None:
        if update.commit_token is not self._pending:
            raise RuntimeError("vLLM update is stale or belongs to another source")
        token = self._pending
        assert token is not None
        self._cursor = token.cursor
        self._blocks = token.blocks
        self._authoritative = True
        if token.full_snapshot:
            self._next_full_refresh = self._clock() + self._full_refresh_interval_s
        self._pending = None

    def abort(self, update: EngineUpdate) -> None:
        if update.commit_token is not self._pending:
            raise RuntimeError("vLLM update is stale or belongs to another source")
        self._pending = None

    async def close(self) -> None:
        await self._transport.close()


def _records(blocks: dict[tuple[int, str], frozenset[int]]) -> tuple[BlockRecord, ...]:
    return tuple(
        BlockRecord(key, medium, tuple(sorted(groups)))
        for (key, medium), groups in sorted(blocks.items())
    )


def _medium(value: str | None) -> str:
    if value == "CPU":
        return "mem"
    if value in (None, "GPU"):
        return "hbm"
    raise RuntimeError(f"unsupported vLLM cache medium: {value!r}")


def _group(value: int | None, cache_group_count: int) -> int:
    group = 0 if value is None else int(value)
    if group < 0 or group >= cache_group_count:
        raise RuntimeError(f"unregistered vLLM cache group id: {group}")
    return group


class _MsgspecDecoder:
    def __init__(self) -> None:
        import msgspec

        self._decoder = msgspec.msgpack.Decoder()

    def __call__(self, payload: bytes) -> Any:
        raw = self._decoder.decode(payload)
        if not isinstance(raw, list) or len(raw) < 2 or not isinstance(raw[1], list):
            raise ValueError("invalid vLLM KVEventBatch payload")
        return _DecodedBatch([_DecodedEvent(event) for event in raw[1]])


class _DecodedBatch:
    def __init__(self, events: list["_DecodedEvent"]) -> None:
        self.events = events


class _DecodedEvent:
    def __init__(self, data: Any) -> None:
        if not isinstance(data, dict):
            raise ValueError("invalid vLLM cache event payload")
        event_type = data.get("type")
        if not isinstance(event_type, str):
            raise ValueError("vLLM cache event has no type tag")
        self.event_type = event_type
        for key, value in data.items():
            if key != "type" and isinstance(key, str):
                setattr(self, key, value)


class _ZmqVllmTransport:
    _END = (-1).to_bytes(8, "big", signed=True)

    def __init__(
        self, pub_endpoint: str, replay_endpoint: str, topic: str, timeout_s: float
    ) -> None:
        import zmq
        import zmq.asyncio

        self._zmq = zmq
        self._timeout_s = timeout_s
        context = zmq.asyncio.Context.instance()
        self._sub = context.socket(zmq.SUB)
        self._sub.connect(pub_endpoint)
        self._sub.setsockopt_string(zmq.SUBSCRIBE, topic)
        self._dealer = context.socket(zmq.DEALER)
        self._dealer.connect(replay_endpoint)

    async def next_sequences(self, expected_sequence: int) -> list[tuple[int, bytes]]:
        while True:
            try:
                raw_frames = await asyncio.wait_for(
                    self._sub.recv_multipart(), timeout=self._timeout_s
                )
            except asyncio.TimeoutError:
                # A late subscriber may start after the publisher became
                # idle. Ask the replay socket directly instead of waiting for
                # a future live message to reveal the gap.
                return await self._replay(expected_sequence, None)
            frames = _payload_frames(raw_frames)
            if len(frames) != 3 or len(frames[1]) != 8:
                continue
            live_sequence = int.from_bytes(frames[1], "big")
            if live_sequence < expected_sequence:
                if live_sequence == 0:
                    return [(live_sequence, frames[2])]
                continue
            live = (live_sequence, frames[2])
            if live_sequence == expected_sequence:
                return [live]
            replay = await self._replay(expected_sequence, live_sequence)
            if not replay or replay[0][0] != expected_sequence:
                # Let the source inspect the live batch: an AllBlocksCleared
                # marker is sufficient to recover without the missing prefix.
                return [live]
            return replay + [live]

    async def _replay(self, start: int, stop: int | None) -> list[tuple[int, bytes]]:
        await self._dealer.send_multipart([b"", start.to_bytes(8, "big")])
        result: list[tuple[int, bytes]] = []
        while True:
            frames = _payload_frames(
                await asyncio.wait_for(
                    self._dealer.recv_multipart(), timeout=self._timeout_s
                )
            )
            if len(frames) != 3:
                raise RuntimeError("malformed vLLM replay response")
            if frames[1] == self._END:
                return result
            sequence = int.from_bytes(frames[1], "big")
            if stop is not None and sequence >= stop:
                continue
            result.append((sequence, frames[2]))

    async def close(self) -> None:
        self._sub.close(linger=0)
        self._dealer.close(linger=0)


def _payload_frames(frames: list[bytes]) -> list[bytes]:
    # DEALER sockets may retain the empty delimiter sent by vLLM's ROUTER.
    return frames[1:] if len(frames) == 4 and frames[0] == b"" else frames
