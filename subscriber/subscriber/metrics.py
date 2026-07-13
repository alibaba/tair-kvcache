from __future__ import annotations

import asyncio
import time

from subscriber import logger


class LatencyReporter:
    """Best-effort, isolated reporting for KV event forwarding latency."""

    def __init__(
        self,
        *,
        warning_threshold_s: float = 0.05,
        summary_interval_s: float = 60.0,
    ) -> None:
        self._warning_threshold_s = warning_threshold_s
        self._summary_interval_s = summary_interval_s
        self._queue: asyncio.Queue[float] = asyncio.Queue(maxsize=1024)
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

    def report(self, latency_s: float) -> None:
        """Queue a latency sample without blocking the event loop."""

        try:
            self._queue.put_nowait(latency_s)
        except Exception:
            pass

    async def _run(self) -> None:
        samples: list[float] = []
        next_summary_at = time.monotonic() + self._summary_interval_s
        try:
            while True:
                try:
                    timeout = max(0.0, next_summary_at - time.monotonic())
                    latency_s = await asyncio.wait_for(
                        self._queue.get(),
                        timeout=timeout,
                    )
                    samples.append(latency_s)
                    await self._log_warning_if_slow(latency_s)
                except TimeoutError:
                    pass
                except asyncio.CancelledError:
                    raise
                except Exception:
                    pass
                if time.monotonic() >= next_summary_at:
                    try:
                        await self._log_summary(samples)
                    except Exception:
                        pass
                    samples.clear()
                    next_summary_at = time.monotonic() + self._summary_interval_s
        except asyncio.CancelledError:
            raise
        except Exception:
            pass

    async def _log_warning_if_slow(self, latency_s: float) -> None:
        if latency_s <= self._warning_threshold_s:
            return
        await asyncio.to_thread(
            logger.warning,
            "kv event forwarding latency exceeded threshold",
            step="kv_event_metrics",
            tags={
                "latency_ms": round(latency_s * 1000, 3),
                "threshold_ms": round(self._warning_threshold_s * 1000, 3),
            },
        )

    async def _log_summary(self, samples: list[float]) -> None:
        if not samples:
            return
        await asyncio.to_thread(
            logger.info,
            "kv event forwarding latency average",
            step="kv_event_metrics",
            tags={
                "average_latency_ms": round(sum(samples) / len(samples) * 1000, 3),
                "sample_count": len(samples),
            },
        )
