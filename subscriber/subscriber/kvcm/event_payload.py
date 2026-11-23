from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import cast

from subscriber.kvcm.enum import KvcmReportEventType
from subscriber.types import (
    AllBlocksCleared,
    BlockRemoved,
    BlockSnapshot,
    BlockStored,
    KVEventBatch,
)


@dataclass(frozen=True)
class ReportEventSourceCounts:
    """Number of subscriber batches and events that produced a request."""

    batch_count: int
    event_count: int


def expand_report_events(
    batches: list[KVEventBatch],
    *,
    medium_mapper: Callable[[str | None], str],
    block_specs: Callable[[str, int | None], list[dict[str, str]]],
    block_spec_names: Callable[[int | None], list[str]],
) -> list[dict[str, object]]:
    """Expand subscriber batches into ordered KVCM report events."""

    events: list[dict[str, object]] = []
    for batch in batches:
        for event in batch.events:
            if isinstance(event, BlockStored):
                medium = medium_mapper(event.medium)
                component_id = (
                    event.component_id
                    if event.component_id is not None
                    else event.group_idx
                )
                specs = block_specs(medium, component_id)
                for block_hash in event.block_hashes:
                    events.append(
                        {
                            "event_type": KvcmReportEventType.BLOCK_ADD,
                            "block_add": {
                                "block_key": str(block_hash),
                                "medium": medium,
                                "specs": specs,
                            },
                        }
                    )
            elif isinstance(event, BlockRemoved):
                medium = medium_mapper(event.medium)
                component_id = (
                    event.component_id
                    if event.component_id is not None
                    else event.group_idx
                )
                spec_names = block_spec_names(component_id)
                for block_hash in event.block_hashes:
                    events.append(
                        {
                            "event_type": KvcmReportEventType.BLOCK_DELETE,
                            "block_delete": {
                                "block_key": str(block_hash),
                                "medium": medium,
                                "spec_names": spec_names,
                            },
                        }
                    )
            elif isinstance(event, AllBlocksCleared):
                events.append(
                    {"event_type": KvcmReportEventType.HOST_DOWN, "host_down": {}}
                )
    return events


def split_report_event_requests(
    events: list[dict[str, object]],
) -> list[list[dict[str, object]]]:
    """Build KVCM-valid requests while preserving the latest reset boundary.

    KVCM requires ``EVENT_HOST_DOWN`` to be the only event in a ReportEvent
    request. Events before the latest host-down are stale by definition, so
    they are discarded; later block events are sent in a second request.
    """

    last_host_down_index: int | None = None
    for index, event in enumerate(events):
        if event.get("event_type") == KvcmReportEventType.HOST_DOWN:
            last_host_down_index = index
    if last_host_down_index is None:
        return [events] if events else []

    requests = [[events[last_host_down_index]]]
    current_events = events[last_host_down_index + 1 :]
    if current_events:
        requests.append(current_events)
    return requests


def build_merged_snapshot_blocks(
    batches: list[KVEventBatch],
    *,
    medium_mapper: Callable[[str | None], str],
    block_specs: Callable[[str, int | None], list[dict[str, str]]],
) -> list[dict[str, object]]:
    """Single-pass fused collect + merge for snapshot blocks.

    Flattens and merges blocks sharing ``(block_key, medium)`` inline, then
    deduplicates specs by ``group_idx`` — the same ``(block_hash, group_idx)``
    pair appearing multiple times (high-concurrency duplicate) produces only one
    spec entry. Order of first occurrence is preserved.
    """

    if not batches:
        return []

    index: dict[tuple[str, str], int] = {}
    seen_groups: list[int | set[int]] = []
    merged: list[dict[str, object]] = []

    for batch in batches:
        for event in batch.events:
            if not isinstance(event, BlockSnapshot):
                continue
            medium = medium_mapper(event.medium)
            for item in event.items:
                block_key = str(item.block_hash)
                key = (block_key, medium)
                pos = index.get(key)
                if pos is None:
                    idx = len(merged)
                    index[key] = idx
                    seen_groups.append(item.group_idx)
                    merged.append(
                        {
                            "block_key": block_key,
                            "medium": medium,
                            "specs": list(block_specs(medium, item.group_idx)),
                        }
                    )
                    continue
                groups = seen_groups[pos]
                if isinstance(groups, int):
                    if item.group_idx == groups:
                        continue
                    seen_groups[pos] = {groups, item.group_idx}
                elif item.group_idx in groups:
                    continue
                else:
                    groups.add(item.group_idx)
                specs = cast(list[dict[str, str]], merged[pos]["specs"])
                specs.extend(block_specs(medium, item.group_idx))

    return merged


def report_event_count(batches: list[KVEventBatch]) -> int:
    """Count KVCM report events without expanding payloads or serializing JSON."""

    count = 0
    for batch in batches:
        for event in batch.events:
            if isinstance(event, (BlockStored, BlockRemoved)):
                count += len(event.block_hashes)
            elif isinstance(event, AllBlocksCleared):
                count += 1
    return count


def source_counts(batches: list[KVEventBatch]) -> ReportEventSourceCounts:
    """Count source batches and unexpanded subscriber events."""

    return ReportEventSourceCounts(
        batch_count=len(batches),
        event_count=sum(len(batch.events) for batch in batches),
    )
