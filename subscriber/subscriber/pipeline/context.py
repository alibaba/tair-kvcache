"""Per-batch mutable context flowing through the forwarding pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from subscriber.engine.base import EngineEventBatch
from subscriber.metrics import BatchTelemetry
from subscriber.types import KVEventBatch

if TYPE_CHECKING:
    from subscriber.metrics import MetricsReporter


@dataclass
class PipelineContext:
    """Mutable per-batch context carried from producer through sender to kvcm.

    Replaces ``QueuedKVEventBatch`` as the queue item type.  Holds the
    adapter's :class:`EngineEventBatch` and enriches it with epoch gating
    and merge correlation metadata.  Owns the telemetry lifecycle: mark
    stages, record drops, and submit to the reporter.
    """

    event: EngineEventBatch
    epoch_snapshot: int
    reporter: MetricsReporter
    batch_trace_id: str | None = None
    drop_reason: str | None = None

    @property
    def trace_id(self) -> str:
        return self.event.trace_id

    @property
    def batches(self) -> list[KVEventBatch]:
        return self.event.batches

    @property
    def telemetry(self) -> BatchTelemetry:
        return self.event.telemetry

    def mark(self, stage: str) -> None:
        self.telemetry.mark(stage)

    def mark_dropped(self, reason: str) -> None:
        self.drop_reason = reason
        self.telemetry.mark_dropped(reason)

    def submit(self) -> None:
        self.reporter.submit(self.telemetry)

    def submit_dropped(self, reason: str) -> None:
        self.mark_dropped(reason)
        self.submit()
