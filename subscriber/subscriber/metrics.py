from __future__ import annotations

import asyncio
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass

from subscriber import logger


@dataclass(frozen=True)
class Span:
    """A single named pipeline stage with its measured duration in seconds."""

    name: str
    duration_s: float


@dataclass(frozen=True)
class MetricSample:
    """One forwarded batch's timing spans and aggregate counters."""

    spans: Sequence[Span]
    counters: Mapping[str, int]


class StageTimer:
    """Records monotonic marks along a pipeline and derives per-stage spans.

    The timer captures an origin at construction. Each ``mark(name)`` closes the
    stage that began at the previous mark (or the origin) and labels it. Adding a
    new stage only requires one extra ``mark`` call at the right point.
    """

    def __init__(self, *, clock: Callable[[], float] = time.monotonic) -> None:
        self._clock = clock
        self._origin = clock()
        self._marks: list[tuple[str, float]] = []

    def mark(self, name: str) -> None:
        self._marks.append((name, self._clock()))

    def spans(self) -> list[Span]:
        spans: list[Span] = []
        previous = self._origin
        for name, at in self._marks:
            spans.append(Span(name, at - previous))
            previous = at
        return spans


class SpanMetricsReporter:
    """Best-effort reporting for forwarding latency and KVCM block hash counts."""

    def __init__(
        self,
        *,
        warning_threshold_s: float = 0.05,
        summary_interval_s: float = 60.0,
    ) -> None:
        self._warning_threshold_s = warning_threshold_s
        self._summary_interval_s = summary_interval_s
        self._queue: asyncio.Queue[MetricSample] = asyncio.Queue(maxsize=1024)
        self._task: asyncio.Task[None] | None = None

    async def start(self) -> None:
        try:
            if self._task is not None and not self._task.done():
                return
            self._task = asyncio.create_task(
                self._run(),
                name="kv-event-metrics",
            )
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

    def report(self, sample: MetricSample) -> None:
        """Queue a forwarding metric sample. Never blocks and never raises."""

        try:
            self._queue.put_nowait(sample)
        except Exception:
            pass

    async def _run(self) -> None:
        stage_totals: dict[str, float] = {}
        stage_counts: dict[str, int] = {}
        total_sum = 0.0
        sample_count = 0
        counter_totals: dict[str, int] = {}
        next_summary_at = time.monotonic() + self._summary_interval_s
        try:
            while True:
                try:
                    timeout = max(0.0, next_summary_at - time.monotonic())
                    sample = await asyncio.wait_for(
                        self._queue.get(),
                        timeout=timeout,
                    )
                    spans = sample.spans
                    trace_total = 0.0
                    for span in spans:
                        stage_totals[span.name] = (
                            stage_totals.get(span.name, 0.0) + span.duration_s
                        )
                        stage_counts[span.name] = stage_counts.get(span.name, 0) + 1
                        trace_total += span.duration_s
                    for name, count in sample.counters.items():
                        counter_totals[name] = counter_totals.get(name, 0) + count
                    total_sum += trace_total
                    sample_count += 1
                    await self._log_warning_if_slow(spans, trace_total)
                except TimeoutError:
                    pass
                except asyncio.CancelledError:
                    raise
                except Exception:
                    pass
                if time.monotonic() >= next_summary_at:
                    try:
                        await self._log_summary(
                            stage_totals,
                            stage_counts,
                            total_sum,
                            sample_count,
                            counter_totals,
                        )
                    except Exception:
                        pass
                    stage_totals = {}
                    stage_counts = {}
                    total_sum = 0.0
                    sample_count = 0
                    counter_totals = {}
                    next_summary_at = time.monotonic() + self._summary_interval_s
        except asyncio.CancelledError:
            raise
        except Exception:
            pass

    async def _log_warning_if_slow(self, spans: Sequence[Span], total_s: float) -> None:
        if total_s <= self._warning_threshold_s:
            return
        tags: dict[str, object] = {
            f"{span.name}_ms": round(span.duration_s * 1000, 3) for span in spans
        }
        tags["total_ms"] = round(total_s * 1000, 3)
        tags["threshold_ms"] = round(self._warning_threshold_s * 1000, 3)
        await asyncio.to_thread(
            logger.warning,
            "kv event forwarding latency exceeded threshold",
            step="kv_event_metrics",
            tags=tags,
        )

    async def _log_summary(
        self,
        stage_totals: dict[str, float],
        stage_counts: dict[str, int],
        total_sum: float,
        sample_count: int,
        counter_totals: Mapping[str, int],
    ) -> None:
        if sample_count == 0:
            return
        tags: dict[str, object] = {
            f"{name}_avg_ms": round(total / stage_counts[name] * 1000, 3)
            for name, total in stage_totals.items()
        }
        tags["total_avg_ms"] = round(total_sum / sample_count * 1000, 3)
        tags["sample_count"] = sample_count
        tags.update(counter_totals)
        await asyncio.to_thread(
            logger.info,
            "kv event forwarding latency average",
            step="kv_event_metrics",
            tags=tags,
        )


class ZmqQueueMetricsReporter:
    """Best-effort periodic summary of observable vLLM SUB queue signals.

    Stable libzmq exposes only ``POLLIN`` readiness, not a numeric inbound
    queue depth. The reporter records how often a message remains ready after
    each receive, which proves at least one message was queued behind it.
    """

    def __init__(
        self,
        *,
        state_reader: Callable[[], Mapping[str, object]],
        summary_interval_s: float = 60.0,
    ) -> None:
        self._state_reader = state_reader
        self._summary_interval_s = summary_interval_s
        self._received_message_count = 0
        self._received_message_bytes = 0
        self._queue_nonempty_observation_count = 0
        self._sequence_gap_count = 0
        self._missed_message_count = 0
        self._task: asyncio.Task[None] | None = None

    async def start(self) -> None:
        """Start the periodic logger without affecting event forwarding."""

        try:
            if self._task is not None and not self._task.done():
                return
            self._task = asyncio.create_task(
                self._run(),
                name="zmq-queue-metrics",
            )
        except Exception:
            pass

    async def stop(self) -> None:
        """Stop periodic logging without propagating reporter failures."""

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

    def record_message(
        self,
        *,
        message_bytes: int,
        queue_nonempty_after_receive: bool,
    ) -> None:
        """Record one live ZMQ message. Never blocks or raises."""

        try:
            self._received_message_count += 1
            self._received_message_bytes += message_bytes
            if queue_nonempty_after_receive:
                self._queue_nonempty_observation_count += 1
        except Exception:
            pass

    def record_sequence_gap(self, *, missed_message_count: int) -> None:
        """Record a live-sequence gap. Never blocks or raises."""

        try:
            if missed_message_count > 0:
                self._sequence_gap_count += 1
                self._missed_message_count += missed_message_count
        except Exception:
            pass

    async def _run(self) -> None:
        try:
            while True:
                await asyncio.sleep(self._summary_interval_s)
                try:
                    await self._log_summary()
                except asyncio.CancelledError:
                    raise
                except Exception:
                    pass
        except asyncio.CancelledError:
            raise
        except Exception:
            pass

    async def _log_summary(self) -> None:
        tags: dict[str, object] = {}
        try:
            tags.update(self._state_reader())
        except Exception as exc:
            tags["zmq_queue_state_error"] = type(exc).__name__
        tags.update(
            {
                "zmq_received_message_count": self._received_message_count,
                "zmq_received_message_bytes": self._received_message_bytes,
                "zmq_queue_nonempty_observation_count": (
                    self._queue_nonempty_observation_count
                ),
                "zmq_sequence_gap_count": self._sequence_gap_count,
                "zmq_missed_message_count": self._missed_message_count,
            }
        )
        self._reset_window()
        await asyncio.to_thread(
            logger.info,
            "vLLM ZMQ queue metrics",
            step="zmq_queue_metrics",
            tags=tags,
        )

    def _reset_window(self) -> None:
        self._received_message_count = 0
        self._received_message_bytes = 0
        self._queue_nonempty_observation_count = 0
        self._sequence_gap_count = 0
        self._missed_message_count = 0
