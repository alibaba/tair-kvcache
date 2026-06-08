"""Concurrent benchmark runner with QPS rate limiting."""

import time
import logging
import threading
from concurrent.futures import ThreadPoolExecutor

from .client import OptimizerClient
from .config import BenchmarkConfig
from .stats import StatsCollector
from .workload import WorkloadGenerator

logger = logging.getLogger(__name__)


class TokenBucket(object):
    """Token bucket rate limiter (thread-safe) with Condition-based waiting."""

    def __init__(self, rate: float, capacity: float):
        self._rate = rate          # tokens per second
        self._capacity = capacity  # max burst
        self._tokens = capacity
        self._last_refill = time.monotonic()
        self._condition = threading.Condition()

    def acquire(self):
        """Block until a token is available."""
        with self._condition:
            while True:
                now = time.monotonic()
                elapsed = now - self._last_refill
                self._tokens = min(self._capacity, self._tokens + elapsed * self._rate)
                self._last_refill = now
                if self._tokens >= 1.0:
                    self._tokens -= 1.0
                    return
                # Calculate wait time until next token arrives
                wait_time = (1.0 - self._tokens) / self._rate
                self._condition.wait(timeout=wait_time)


class BenchmarkRunner:
    """Drives concurrent TraceQuery requests at a target QPS."""

    def __init__(self, config: BenchmarkConfig, client: OptimizerClient,
                 stats: StatsCollector):
        self._config = config
        self._client = client
        self._stats = stats
        self._stop_event = threading.Event()

    def run(self):
        config = self._config
        logger.info("Starting benchmark: qps=%d, threads=%d, duration=%ds, warmup=%ds",
                     config.qps, config.thread_count, config.duration_seconds,
                     config.warmup_seconds)

        if config.warmup_seconds > 0:
            logger.info("Warming up for %d seconds...", config.warmup_seconds)
            self._run_phase(config.warmup_seconds, is_warmup=True)
            logger.info("Warmup complete.")

        self._stats.start()
        self._run_phase(config.duration_seconds, is_warmup=False)
        self._stats.report_final()

    def _run_phase(self, duration: int, is_warmup: bool):
        """Run requests for a given duration with QPS rate limiting."""
        config = self._config
        self._stop_event.clear()

        # Token bucket: allow burst up to thread_count to keep all threads busy
        bucket = TokenBucket(rate=config.qps, capacity=config.thread_count)

        # Each thread gets its own workload generator (thread-local RNG)
        generators = [
            WorkloadGenerator(config, seed=i)
            for i in range(config.thread_count)
        ]

        def worker(thread_id: int):
            generator = generators[thread_id]
            while not self._stop_event.is_set():
                bucket.acquire()
                if self._stop_event.is_set():
                    break

                block_keys = generator.generate()
                start_ns = time.monotonic_ns()
                try:
                    response = self._client.trace_query_raw(
                        config.instance_id, block_keys)
                    latency_ms = (time.monotonic_ns() - start_ns) / 1_000_000

                    if response.status_code == 200 and not is_warmup:
                        body = response.json()
                        hit_count = body.get("cache_hit_count", 0)
                        total_blocks = body.get("total_blocks", 0)
                        self._stats.record_success(latency_ms, hit_count, total_blocks)
                    elif not is_warmup:
                        self._stats.record_error(latency_ms)
                except Exception:
                    latency_ms = (time.monotonic_ns() - start_ns) / 1_000_000
                    if not is_warmup:
                        self._stats.record_error(latency_ms)

        with ThreadPoolExecutor(max_workers=config.thread_count) as pool:
            futures = [pool.submit(worker, i) for i in range(config.thread_count)]

            deadline = time.monotonic() + duration
            while time.monotonic() < deadline:
                time.sleep(config.report_interval if not is_warmup else 1.0)
                if not is_warmup:
                    self._stats.maybe_report_interval()

            self._stop_event.set()
            for future in futures:
                future.result()
