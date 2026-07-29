from __future__ import annotations

from subscriber.kvcm.event_payload import (
    ReportEventSourceCounts,
    expand_report_events,
    report_event_count,
    source_counts,
    split_report_event_requests,
)
from subscriber.types import (
    AllBlocksCleared,
    BlockRemoved,
    BlockStored,
    KVEventBatch,
)


def _medium_mapper(medium: str | None) -> str:
    return {"GPU": "hbm", "CPU": "mem"}.get(medium, "")


def _block_specs(medium: str, group_idx: int | None) -> list[dict[str, str]]:
    return [
        {
            "name": f"spec-{group_idx}",
            "uri": f"vllm://127.0.0.1:9000/{medium}",
        }
    ]


def _block_spec_names(group_idx: int | None) -> list[str]:
    return [f"spec-{group_idx}"]


def test_event_payload_expands_ordered_events_and_counts_report_events() -> None:
    batches = [
        KVEventBatch(
            ts=1.0,
            events=[
                BlockStored(
                    block_hashes=[11, 12],
                    parent_block_hash=None,
                    token_ids=[1, 2],
                    block_size=2,
                    lora_id=None,
                    medium="GPU",
                    lora_name=None,
                    group_idx=0,
                ),
                BlockRemoved(block_hashes=[13], medium="CPU", group_idx=1),
            ],
        ),
        KVEventBatch(ts=2.0, events=[AllBlocksCleared()]),
    ]

    events = expand_report_events(
        batches,
        medium_mapper=_medium_mapper,
        block_specs=_block_specs,
        block_spec_names=_block_spec_names,
    )
    assert events == [
        {
            "event_type": "EVENT_BLOCK_ADD",
            "block_add": {
                "block_key": "11",
                "medium": "hbm",
                "specs": [
                    {
                        "name": "spec-0",
                        "uri": "vllm://127.0.0.1:9000/hbm",
                    }
                ],
            },
        },
        {
            "event_type": "EVENT_BLOCK_ADD",
            "block_add": {
                "block_key": "12",
                "medium": "hbm",
                "specs": [
                    {
                        "name": "spec-0",
                        "uri": "vllm://127.0.0.1:9000/hbm",
                    }
                ],
            },
        },
        {
            "event_type": "EVENT_BLOCK_DELETE",
            "block_delete": {
                "block_key": "13",
                "medium": "mem",
                "spec_names": ["spec-1"],
            },
        },
        {"event_type": "EVENT_HOST_DOWN", "host_down": {}},
    ]
    assert source_counts(batches) == ReportEventSourceCounts(
        batch_count=2, event_count=3
    )
    assert report_event_count(batches) == 4


def test_split_report_event_requests_keeps_host_down_exclusive() -> None:
    stale_add = {"event_type": "EVENT_BLOCK_ADD", "block_add": {"block_key": "1"}}
    host_down = {"event_type": "EVENT_HOST_DOWN", "host_down": {}}
    current_add = {
        "event_type": "EVENT_BLOCK_ADD",
        "block_add": {"block_key": "2"},
    }

    assert split_report_event_requests([stale_add, host_down, current_add]) == [
        [host_down],
        [current_add],
    ]
