"""Statistics collector and reporter for benchmark results."""

import logging
import threading
import time

logger = logging.getLogger(__name__)


class LatencyStats(object):
    """Accumulated latency statistics."""

    def __init__(self):
        self.total_requests = 0
        self.success_count = 0
        self.error_count = 0
        self.total_latency_ms = 0.0
        self.min_latency_ms = float("inf")
        self.max_latency_ms = 0.0
        self.total_query_blocks = 0
        self.hit_blocks_per_capacity = {}
        self.theoretical_hit_blocks = 0
        self.has_theoretical = False
        self.latencies = []


class StatsCollector:
    """Thread-safe statistics collector with periodic reporting."""

    def __init__(self, report_interval: int):
        self._report_interval = report_interval
        self._lock = threading.Lock()
        self._global_stats = LatencyStats()
        self._interval_stats = LatencyStats()
        self._start_time = 0.0
        self._interval_start = 0.0

    def start(self):
        self._start_time = time.monotonic()
        self._interval_start = self._start_time

    def record_success(self, latency_ms: float, total_blocks: int,
                       per_capacity=None, theoretical_hits=None):
        """Record a successful TraceQuery."""
        per_capacity = per_capacity or []
        with self._lock:
            for stats in (self._global_stats, self._interval_stats):
                stats.total_requests += 1
                stats.success_count += 1
                stats.total_latency_ms += latency_ms
                stats.min_latency_ms = min(stats.min_latency_ms, latency_ms)
                stats.max_latency_ms = max(stats.max_latency_ms, latency_ms)
                stats.total_query_blocks += total_blocks
                for cap_gb, hits in per_capacity:
                    stats.hit_blocks_per_capacity[cap_gb] = \
                        stats.hit_blocks_per_capacity.get(cap_gb, 0) + hits
                if theoretical_hits is not None:
                    stats.has_theoretical = True
                    stats.theoretical_hit_blocks += theoretical_hits
                stats.latencies.append(latency_ms)

    def record_error(self, latency_ms: float):
        with self._lock:
            for stats in (self._global_stats, self._interval_stats):
                stats.total_requests += 1
                stats.error_count += 1
                stats.total_latency_ms += latency_ms
                stats.min_latency_ms = min(stats.min_latency_ms, latency_ms)
                stats.max_latency_ms = max(stats.max_latency_ms, latency_ms)

    def maybe_report_interval(self):
        """Check and print interval report if enough time has passed."""
        now = time.monotonic()
        if now - self._interval_start >= self._report_interval:
            self._print_interval_report(now)

    @staticmethod
    def _format_hit_rates(stats) -> str:
        total = stats.total_query_blocks
        lines = []
        for cap_gb in sorted(stats.hit_blocks_per_capacity.keys()):
            hits = stats.hit_blocks_per_capacity[cap_gb]
            rate = (hits / total * 100) if total > 0 else 0.0
            lines.append("    cap=%-8s hit_rate=%6.2f%%  (hits=%d)" % (
                ("%gGB" % cap_gb), rate, hits))
        if stats.has_theoretical:
            rate = (stats.theoretical_hit_blocks / total * 100) if total > 0 else 0.0
            lines.append("    theoretical  hit_rate=%6.2f%%  (hits=%d)" % (
                rate, stats.theoretical_hit_blocks))
        return "\n".join(lines) if lines else "    (no capacity results)"

    def _print_interval_report(self, now: float):
        with self._lock:
            stats = self._interval_stats
            elapsed = now - self._interval_start
            if stats.total_requests == 0:
                self._interval_start = now
                return

            actual_qps = stats.total_requests / elapsed if elapsed > 0 else 0
            avg_latency = stats.total_latency_ms / stats.total_requests
            hit_rate_lines = self._format_hit_rates(stats)

            logger.info(
                "[Interval] qps=%.1f  avg=%.2fms  min=%.2fms  max=%.2fms  "
                "ok=%d  err=%d\n%s",
                actual_qps, avg_latency, stats.min_latency_ms, stats.max_latency_ms,
                stats.success_count, stats.error_count, hit_rate_lines,
            )

            self._interval_stats = LatencyStats()
            self._interval_start = now

    def report_final(self):
        """Print final summary report."""
        with self._lock:
            stats = self._global_stats
            elapsed = time.monotonic() - self._start_time

        if stats.total_requests == 0:
            logger.info("No requests completed.")
            return

        actual_qps = stats.total_requests / elapsed if elapsed > 0 else 0
        avg_latency = stats.total_latency_ms / stats.total_requests
        hit_rate_lines = self._format_hit_rates(stats)

        sorted_latencies = sorted(stats.latencies)
        total = len(sorted_latencies)

        def percentile(p: float) -> float:
            if total == 0:
                return 0.0
            index = int(total * p / 100)
            return sorted_latencies[min(index, total - 1)]

        logger.info(
            "\n"
            "============ Benchmark Summary ============\n"
            "  Duration:       %.1f s\n"
            "  Total requests: %d\n"
            "  Success:        %d\n"
            "  Errors:         %d\n"
            "  Actual QPS:     %.1f\n"
            "  Avg latency:    %.2f ms\n"
            "  Min latency:    %.2f ms\n"
            "  Max latency:    %.2f ms\n"
            "  P50 latency:    %.2f ms\n"
            "  P90 latency:    %.2f ms\n"
            "  P99 latency:    %.2f ms\n"
            "  Cache hit rate (per capacity):\n%s\n"
            "===========================================",
            elapsed, stats.total_requests, stats.success_count,
            stats.error_count, actual_qps, avg_latency,
            stats.min_latency_ms, stats.max_latency_ms,
            percentile(50), percentile(90), percentile(99),
            hit_rate_lines,
        )
