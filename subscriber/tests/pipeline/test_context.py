"""Tests for PipelineContext — the per-batch mutable pipeline context."""

from __future__ import annotations

from unittest.mock import MagicMock

from subscriber.engine.base import EngineEventBatch
from subscriber.metrics import BatchTelemetry
from subscriber.pipeline.context import PipelineContext
from subscriber.trace import generate_trace_id
from subscriber.types import BlockStored, KVEventBatch


def _make_event_batch() -> EngineEventBatch:
    batch = KVEventBatch(
        ts=1.0,
        events=[
            BlockStored(
                block_hashes=["hash1"],
                parent_block_hash=None,
                token_ids=[],
                block_size=16,
                lora_id=None,
                medium="gpu",
                lora_name=None,
            )
        ],
        data_parallel_rank=None,
    )
    telemetry = BatchTelemetry(pipeline="incremental")
    trace_id = generate_trace_id()
    return EngineEventBatch(batches=[batch], telemetry=telemetry, trace_id=trace_id)


def _ctx(**kwargs: object) -> PipelineContext:
    defaults: dict = {
        "event": _make_event_batch(),
        "epoch_snapshot": 1,
        "reporter": MagicMock(),
    }
    defaults.update(kwargs)
    return PipelineContext(**defaults)


class TestPipelineContext:
    """PipelineContext delegates to EngineEventBatch and tracks pipeline state."""

    def test_property_delegation(self) -> None:
        event = _make_event_batch()
        ctx = PipelineContext(event=event, epoch_snapshot=1, reporter=MagicMock())
        assert ctx.trace_id == event.trace_id
        assert ctx.batches is event.batches
        assert ctx.telemetry is event.telemetry

    def test_initial_state(self) -> None:
        ctx = _ctx()
        assert ctx.epoch_snapshot == 1
        assert ctx.batch_trace_id is None
        assert ctx.drop_reason is None

    def test_epoch_snapshot_set_by_producer(self) -> None:
        ctx = _ctx(epoch_snapshot=3)
        assert ctx.epoch_snapshot == 3

    def test_batch_trace_id_set_by_sender(self) -> None:
        ctx = _ctx()
        ctx.batch_trace_id = "some_first_item_trace"
        assert ctx.batch_trace_id == "some_first_item_trace"

    def test_mark_dropped_sets_both_fields(self) -> None:
        ctx = _ctx()
        ctx.mark_dropped("epoch_changed")
        assert ctx.drop_reason == "epoch_changed"
        assert ctx.telemetry.drop_reason == "epoch_changed"

    def test_trace_id_is_30_chars(self) -> None:
        ctx = _ctx()
        assert len(ctx.trace_id) == 30

    def test_submit_calls_reporter(self) -> None:
        reporter = MagicMock()
        ctx = _ctx(reporter=reporter)
        ctx.submit()
        reporter.submit.assert_called_once_with(ctx.telemetry)

    def test_submit_dropped_marks_and_submits(self) -> None:
        reporter = MagicMock()
        ctx = _ctx(reporter=reporter)
        ctx.submit_dropped("send_failed")
        assert ctx.drop_reason == "send_failed"
        assert ctx.telemetry.drop_reason == "send_failed"
        reporter.submit.assert_called_once_with(ctx.telemetry)
