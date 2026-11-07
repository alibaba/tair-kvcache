"""Main-path telemetry container and the single async metrics output.

This module owns the two objects the forwarding and snapshot pipelines use to
capture and flush per-batch metrics:

- :class:`BatchTelemetry`: an event-scoped record that the adapter and
  forwarding code populate incrementally with :meth:`mark`, :meth:`count`, and
  :meth:`mark_dropped`. It carries the timing origin, per-stage spans, aggregate
  counters, a drop reason, and the pipeline tag.
- :class:`MetricsReporter`: the single asynchronous output for main-path
  metrics. Callers submit finalized :class:`BatchTelemetry` objects; the
  reporter's background task flushes them to dashlog and emits a warning log
  when a *successful* batch exceeds a latency threshold. Submitting a
  dropped-batch telemetry emits a drop counter plus any span data captured
  before the drop, tagged with ``drop_reason``, and skips the slow-batch
  warning.

The reporter is best-effort: :meth:`submit` never blocks or raises, and
dashlog failures inside the flush task are swallowed. This preserves the
project invariant that metrics reporting must never fail the main path.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field

from subscriber import logger
from subscriber.metrics._base import _dashlog_counter, _dashlog_gauge


@dataclass(frozen=True)
class Span:
    """A single named pipeline stage with its measured duration in seconds."""

    name: str
    duration_s: float


@dataclass(frozen=True)
class _CounterObservation:
    """One counter observation; ``tags`` may be empty for an unlabelled value."""

    name: str
    delta: int
    tags: Mapping[str, str]


@dataclass(frozen=True)
class _GaugeObservation:
    """One gauge observation; ``tags`` may be empty for an unlabelled value."""

    name: str
    value: float
    tags: Mapping[str, str]


class BatchTelemetry:
    """Per-event telemetry container for the main forwarding path.

    An adapter constructs one instance per yielded event and marks its own
    internal stages (``decode``, ``replay_fetch``, ``snapshot_fetch``, ...).
    The subscriber core then continues marking downstream stages
    (``queue_wait``, ``engine_gate_wait``, ``block_filter``, ``kvcm_send``) on
    the same instance and, in the terminal step, calls :meth:`count` for
    aggregate counters or :meth:`mark_dropped` for a lossy path. The instance
    is finally submitted to :class:`MetricsReporter` for asynchronous flush.

    The container replaces the earlier ``StageTimer`` + ad-hoc ``MetricSample``
    pair: instead of building an anonymous sample tuple at the terminal step,
    the same object accumulates every observation as it happens.
    """

    def __init__(
        self,
        *,
        pipeline: str,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._pipeline = pipeline
        self._clock = clock
        self._origin = self._clock_now(0.0)
        self._marks: list[tuple[str, float]] = []
        self._checkpoints: dict[str, float] = {}
        self._counter_observations: list[_CounterObservation] = []
        self._gauge_observations: list[_GaugeObservation] = []
        self._drop_reason: str | None = None

    def _clock_now(self, fallback_s: float) -> float:
        """Read the monotonic clock without letting telemetry break forwarding."""

        try:
            return float(self._clock())
        except Exception:
            return fallback_s

    @property
    def pipeline(self) -> str:
        return self._pipeline

    @property
    def drop_reason(self) -> str | None:
        return self._drop_reason

    @property
    def spans(self) -> list[Span]:
        """Derive per-stage spans from the recorded marks."""

        spans: list[Span] = []
        previous = self._origin
        for name, at in self._marks:
            spans.append(Span(name, at - previous))
            previous = at
        return spans

    @property
    def counters(self) -> Mapping[str, int]:
        """Aggregate unlabelled counters recorded on this telemetry."""

        counters: dict[str, int] = {}
        for observation in self._counter_observations:
            if not observation.tags:
                counters[observation.name] = (
                    counters.get(observation.name, 0) + observation.delta
                )
        return counters

    @property
    def gauges(self) -> Mapping[str, float]:
        """Per-batch gauge values recorded on this telemetry.

        Unlike counters (which accumulate deltas), gauges record the *last*
        value written under each name. Use gauges for per-event measurements
        like a fetched payload size or a fetch latency where each observation
        is a standalone data point rather than a rate.
        """

        gauges: dict[str, float] = {}
        for observation in self._gauge_observations:
            if not observation.tags:
                gauges[observation.name] = observation.value
        return gauges

    def mark(self, name: str) -> float:
        """Close and label the current stage, returning its duration in seconds."""

        previous_s = self._marks[-1][1] if self._marks else self._origin
        now_s = self._clock_now(previous_s)
        self._marks.append((name, now_s))
        return now_s - previous_s

    def checkpoint(self, name: str) -> None:
        """Record a named timestamp without creating a stage span."""

        fallback_s = self._marks[-1][1] if self._marks else self._origin
        self._checkpoints[name] = self._clock_now(fallback_s)

    def elapsed_since_checkpoint(
        self, name: str, *, at_s: float | None = None
    ) -> float | None:
        """Return elapsed seconds since ``name`` was checkpointed, or ``None``."""

        checkpoint_at_s = self._checkpoints.get(name)
        if checkpoint_at_s is None:
            return None
        now_s = self._clock_now(checkpoint_at_s) if at_s is None else at_s
        try:
            return now_s - checkpoint_at_s
        except Exception:
            return None

    def count(
        self,
        name: str,
        delta: int,
        *,
        tags: Mapping[str, str] | None = None,
    ) -> None:
        """Record a counter observation, optionally with request-specific labels.

        Omitting ``tags`` (or passing an empty mapping) retains the historical
        per-batch accumulation behavior. Non-empty labels identify an
        independent observation and are preserved until the asynchronous flush.
        """

        try:
            self._counter_observations.append(
                _CounterObservation(name, int(delta), dict(tags or {}))
            )
        except Exception:
            pass

    def gauge(
        self,
        name: str,
        value: float,
        *,
        tags: Mapping[str, str] | None = None,
    ) -> None:
        """Record a gauge observation, optionally with request-specific labels.

        Omitting ``tags`` (or passing an empty mapping) retains the historical
        last-value behavior. Non-empty labels identify an independent
        observation and are preserved until the asynchronous flush.
        """

        try:
            self._gauge_observations.append(
                _GaugeObservation(name, float(value), dict(tags or {}))
            )
        except Exception:
            pass

    def mark_dropped(self, reason: str) -> None:
        """Record that this batch was dropped mid-pipeline with the given reason.

        Preserves any stage marks captured before the drop so the reporter can
        report which stage the pipeline reached before dropping.
        """

        self._drop_reason = reason


@dataclass(frozen=True)
class _FinalizedTelemetry:
    """Immutable snapshot of a :class:`BatchTelemetry` for the flush task."""

    pipeline: str
    spans: Sequence[Span]
    counter_observations: Sequence[_CounterObservation]
    gauge_observations: Sequence[_GaugeObservation]
    drop_reason: str | None = field(default=None)


def _snapshot(telemetry: BatchTelemetry) -> _FinalizedTelemetry:
    return _FinalizedTelemetry(
        pipeline=telemetry.pipeline,
        spans=telemetry.spans,
        counter_observations=tuple(telemetry._counter_observations),
        gauge_observations=tuple(telemetry._gauge_observations),
        drop_reason=telemetry.drop_reason,
    )


class MetricsReporter:
    """Best-effort asynchronous flush for main-path :class:`BatchTelemetry`.

    A single background task drains submissions and writes them to dashlog.
    Successful batches emit per-stage histograms, an end-to-end latency
    histogram, and any accumulated counters, all tagged with the batch's
    ``pipeline``. Dropped batches emit a ``kv_batch_drop_count`` counter and
    stage histograms tagged with ``drop_reason`` in addition to ``pipeline``;
    the slow-batch warning is skipped for dropped telemetry because a partial
    pipeline is expected.

    :meth:`submit` is non-blocking and never raises; if the internal queue
    fails or the flush task has not been started, the sample is silently
    discarded. Dashlog failures inside the flush task are swallowed. Callers
    should not depend on any specific delivery guarantee.
    """

    def __init__(
        self,
        *,
        task_name: str = "kv-event-metrics",
        warning_threshold_s: float = 0.05,
    ) -> None:
        self._task_name = task_name
        self._warning_threshold_s = warning_threshold_s
        self._queue: asyncio.Queue[_FinalizedTelemetry] = asyncio.Queue(maxsize=1024)
        self._task: asyncio.Task[None] | None = None

    async def start(self) -> None:
        try:
            if self._task is not None and not self._task.done():
                return
            self._task = asyncio.create_task(self._run(), name=self._task_name)
        except Exception:
            pass

    async def stop(self) -> None:
        try:
            task = self._task
            if task is None:
                return
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
            finally:
                if self._task is task:
                    self._task = None
        except Exception:
            pass

    def submit(self, telemetry: BatchTelemetry) -> None:
        """Queue a finalized telemetry for async flush. Never blocks or raises."""

        try:
            self._queue.put_nowait(_snapshot(telemetry))
        except Exception:
            pass

    async def _run(self) -> None:
        try:
            while True:
                try:
                    sample = await self._queue.get()
                    if sample.drop_reason is not None:
                        self._flush_dropped(sample)
                    else:
                        self._flush_success(sample)
                        self._log_warning_if_slow(sample)
                except asyncio.CancelledError:
                    raise
                except Exception:
                    pass
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.warning(
                "metrics reporter flush task terminated unexpectedly; "
                "subsequent submissions will be dropped",
                step="kv_event_metrics",
                exc_info=True,
            )

    # ------------------------------------------------------------------ flush

    def _flush_success(self, sample: _FinalizedTelemetry) -> None:
        try:
            tags = {"pipeline": sample.pipeline}
            for span in sample.spans:
                _dashlog_gauge(
                    f"kv_stage_{span.name}_ms",
                    span.duration_s * 1000,
                    tags=tags,
                )
            if sample.spans:
                total_s = sum(span.duration_s for span in sample.spans)
                _dashlog_gauge(
                    "kv_event_e2e_latency_ms",
                    total_s * 1000,
                    tags=tags,
                )
            self._flush_observations(sample, tags)
        except Exception:
            pass

    def _flush_dropped(self, sample: _FinalizedTelemetry) -> None:
        try:
            reason = sample.drop_reason or "unknown"
            _dashlog_counter(
                "kv_batch_drop_count",
                1,
                tags={"pipeline": sample.pipeline, "reason": reason},
            )
            drop_tags = {"pipeline": sample.pipeline, "drop_reason": reason}
            for span in sample.spans:
                _dashlog_gauge(
                    f"kv_stage_{span.name}_ms",
                    span.duration_s * 1000,
                    tags=drop_tags,
                )
            # Counters and gauges recorded before the drop are still emitted
            # so partial-batch data isn't lost; the drop_reason tag lets
            # dashboards separate them from successful measurements.
            self._flush_observations(sample, drop_tags)
        except Exception:
            pass

    def _flush_observations(
        self,
        sample: _FinalizedTelemetry,
        base_tags: Mapping[str, str],
    ) -> None:
        """Flush observations while preserving label and legacy aggregation rules.

        Unlabelled counters are accumulated and unlabelled gauges keep their
        final value, matching the original ``BatchTelemetry`` interface.
        Labelled observations stay independent even if their metric names
        match. ``base_tags`` (pipeline and optional drop reason) intentionally
        override a caller-provided tag of the same name.
        """

        counters: dict[str, int] = {}
        gauges: dict[str, float] = {}
        labelled_counters: list[_CounterObservation] = []
        labelled_gauges: list[_GaugeObservation] = []
        for observation in sample.counter_observations:
            if not observation.tags:
                counters[observation.name] = (
                    counters.get(observation.name, 0) + observation.delta
                )
                continue
            labelled_counters.append(observation)
        for name, count in counters.items():
            _dashlog_counter(name, count, tags=base_tags)
        for observation in labelled_counters:
            tags = dict(observation.tags)
            tags.update(base_tags)
            _dashlog_counter(observation.name, observation.delta, tags=tags)
        for gauge_observation in sample.gauge_observations:
            if not gauge_observation.tags:
                gauges[gauge_observation.name] = gauge_observation.value
                continue
            labelled_gauges.append(gauge_observation)
        for name, value in gauges.items():
            _dashlog_gauge(name, value, tags=base_tags)
        for gauge_observation in labelled_gauges:
            gauge_tags = dict(gauge_observation.tags)
            gauge_tags.update(base_tags)
            _dashlog_gauge(
                gauge_observation.name,
                gauge_observation.value,
                tags=gauge_tags,
            )

    def _log_warning_if_slow(self, sample: _FinalizedTelemetry) -> None:
        if not sample.spans:
            return
        total_s = sum(span.duration_s for span in sample.spans)
        if total_s <= self._warning_threshold_s:
            return
        tags: dict[str, object] = {"pipeline": sample.pipeline}
        tags.update(
            {
                f"{span.name}_ms": round(span.duration_s * 1000, 3)
                for span in sample.spans
            }
        )
        tags["total_ms"] = round(total_s * 1000, 3)
        tags["threshold_ms"] = round(self._warning_threshold_s * 1000, 3)
        logger.warning(
            "kv event forwarding latency exceeded threshold",
            step="kv_event_metrics",
            tags=tags,
        )
