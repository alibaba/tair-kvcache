"""Trace replay runner: replay offline trace data as online TraceQuery requests."""

import glob
import json
import logging
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Iterator, List

from ..core.config import BenchmarkConfig
from ..core.stats import StatsCollector

logger = logging.getLogger(__name__)

_INT64_MAX = (1 << 63) - 1
_UINT64_MOD = 1 << 64


def _hex_to_int64(hex_str):
    """Convert hex hash string to signed int64."""
    value = int(hex_str, 16)
    if value > _INT64_MAX:
        value -= _UINT64_MOD
    return value


class TraceRecord(object):
    """A single trace record parsed from JSONL."""

    __slots__ = ("timestamp", "block_keys", "request_id")

    def __init__(self, timestamp, block_keys, request_id=""):
        self.timestamp = timestamp
        self.block_keys = block_keys
        self.request_id = request_id


class TraceDataSummary(object):
    """Summary statistics of loaded trace data."""

    def __init__(self, total_files=0):
        self.total_files = total_files
        self.total_records = 0
        self.first_timestamp = 0.0
        self.last_timestamp = 0.0
        self.total_block_keys = 0
        self.min_block_keys_per_request = 0
        self.max_block_keys_per_request = 0

    @property
    def time_span_seconds(self):
        return self.last_timestamp - self.first_timestamp if self.total_records > 0 else 0.0

    @property
    def average_qps(self):
        span = self.time_span_seconds
        return self.total_records / span if span > 0 else 0.0

    @property
    def average_block_keys(self):
        return self.total_block_keys / self.total_records if self.total_records > 0 else 0.0

    def estimated_replay_seconds(self, speed_factor):
        return self.time_span_seconds / speed_factor if speed_factor > 0 else float("inf")


class TraceDataLoader:
    """Loads and iterates trace records from JSONL files in timestamp order."""

    def __init__(self, data_dir: str, loop: bool = False):
        self._data_dir = data_dir
        self._files = sorted(glob.glob(os.path.join(data_dir, "*.jsonl")))
        self._loop = loop
        if not self._files:
            raise FileNotFoundError(f"No .jsonl files found in {data_dir}")

    def scan_summary(self) -> TraceDataSummary:
        """Scan all files to collect summary statistics without loading them into memory."""
        summary = TraceDataSummary(total_files=len(self._files))
        min_bk = float("inf")
        max_bk = 0
        global_first_ts = None
        global_last_ts = None

        for filepath in self._files:
            with open(filepath) as fh:
                for line in fh:
                    record = json.loads(line)
                    timestamp = record["timestamp"]
                    block_key_count = len(record.get("input_block_hash_ids", []))

                    summary.total_records += 1
                    summary.total_block_keys += block_key_count
                    min_bk = min(min_bk, block_key_count)
                    max_bk = max(max_bk, block_key_count)

                    if global_first_ts is None or timestamp < global_first_ts:
                        global_first_ts = timestamp
                    if global_last_ts is None or timestamp > global_last_ts:
                        global_last_ts = timestamp

        if summary.total_records > 0:
            summary.first_timestamp = global_first_ts
            summary.last_timestamp = global_last_ts
            summary.min_block_keys_per_request = int(min_bk)
            summary.max_block_keys_per_request = max_bk

        return summary

    def __iter__(self) -> Iterator[TraceRecord]:
        """Yield TraceRecords in file/line order."""
        first_pass = True
        while True:
            if not first_pass and self._loop:
                yield TraceRecord(timestamp=-1.0, block_keys=[], request_id="__loop_boundary__")
            for filepath in self._files:
                with open(filepath) as fh:
                    for line in fh:
                        raw = json.loads(line)
                        yield TraceRecord(
                            timestamp=raw["timestamp"],
                            block_keys=[_hex_to_int64(h) for h in raw.get("input_block_hash_ids", [])],
                            request_id=raw.get("request_id", ""),
                        )
            first_pass = False
            if not self._loop:
                return


class TraceReplayRunner:
    """Replays trace data against an optimizer instance, preserving real timing."""

    def __init__(self, config: BenchmarkConfig, client: Any,
                 stats: StatsCollector):
        self._config = config
        self._client = client
        self._stats = stats
        self._stop_event = threading.Event()

    def run(self):
        config = self._config
        loader = TraceDataLoader(config.trace_data_dir, loop=config.trace_loop)
        summary = loader.scan_summary()
        self._print_trace_summary(summary)
        if summary.total_records == 0:
            raise ValueError(f"No trace records found in {config.trace_data_dir}")

        speed_factor = config.trace_speed_factor
        max_requests = config.trace_max_requests
        loop_count_limit = config.trace_loop_count
        buffer_window_seconds = 10
        effective_qps = summary.average_qps * speed_factor
        max_pending = int(max(config.thread_count, effective_qps) * buffer_window_seconds)

        self._stats.start()
        executor = ThreadPoolExecutor(max_workers=config.thread_count)
        self._pending_semaphore = threading.Semaphore(max_pending)

        try:
            trace_iter = iter(loader)
            first_record = next(trace_iter)
            base_trace_time = first_record.timestamp
            base_real_time = time.monotonic()
            loop_count = 0

            self._submit_request(executor, first_record)
            sent_count = 1

            for record in trace_iter:
                if self._stop_event.is_set():
                    break
                if 0 < max_requests <= sent_count:
                    logger.info("Reached max_requests limit (%d). Stopping.", max_requests)
                    break

                if record.request_id == "__loop_boundary__":
                    loop_count += 1
                    if 0 < loop_count_limit <= loop_count:
                        logger.info("Reached loop count limit (%d). Stopping.", loop_count_limit)
                        break
                    logger.info("Loop pass %d completed, starting pass %d.", loop_count, loop_count + 1)
                    continue

                if record.timestamp < base_trace_time:
                    base_trace_time = record.timestamp
                    base_real_time = time.monotonic()

                trace_elapsed = record.timestamp - base_trace_time
                target_real_time = base_real_time + trace_elapsed / speed_factor
                sleep_duration = target_real_time - time.monotonic()

                if sleep_duration > 0:
                    self._stop_event.wait(timeout=sleep_duration)
                    if self._stop_event.is_set():
                        break

                self._submit_request(executor, record)
                sent_count += 1
                self._stats.maybe_report_interval()

            logger.info("Trace replay finished. Total sent: %d", sent_count)
        finally:
            executor.shutdown(wait=True)

        self._stats.report_final()

    def _submit_request(self, executor, record):
        self._pending_semaphore.acquire()
        executor.submit(self._do_request_with_release, record)

    def _do_request_with_release(self, record):
        try:
            self._do_request(record)
        finally:
            self._pending_semaphore.release()

    def _do_request(self, record):
        start = time.monotonic()
        try:
            latency_ms, total_blocks, per_capacity, theoretical_hits = \
                self._client.trace_query_for_stats(
                    self._config.instance_id, record.block_keys)
            self._stats.record_success(
                latency_ms, total_blocks, per_capacity, theoretical_hits)
        except Exception:
            latency_ms = (time.monotonic() - start) * 1000
            self._stats.record_error(latency_ms)

    def _print_trace_summary(self, summary):
        speed_factor = self._config.trace_speed_factor
        max_requests = self._config.trace_max_requests
        effective_qps = summary.average_qps * speed_factor

        if max_requests > 0 and max_requests < summary.total_records:
            replay_records = max_requests
            replay_ratio = replay_records / summary.total_records if summary.total_records > 0 else 0
            estimated_seconds = summary.time_span_seconds * replay_ratio / speed_factor
        else:
            replay_records = summary.total_records
            estimated_seconds = summary.estimated_replay_seconds(speed_factor)

        estimated_minutes = estimated_seconds / 60

        logger.info(
            "\n"
            "============ Trace Data Summary ============\n"
            "  Data directory:     %s\n"
            "  Total files:        %d\n"
            "  Total records:      %d\n"
            "  Time span:          %.1f s (%.1f min)\n"
            "  Average QPS:        %.1f\n"
            "  Block keys/request: min=%d, max=%d, avg=%.1f\n"
            "  ---- Replay Plan ----\n"
            "  Speed factor:       %.2fx\n"
            "  Replay records:     %d\n"
            "  Estimated duration: %.1f s (%.1f min)\n"
            "  Effective QPS:      %.1f\n"
            "  Loop:               %s\n"
            "  Loop count:         %s\n"
            "  Max requests:       %s\n"
            "=============================================",
            self._config.trace_data_dir,
            summary.total_files,
            summary.total_records,
            summary.time_span_seconds, summary.time_span_seconds / 60,
            summary.average_qps,
            summary.min_block_keys_per_request,
            summary.max_block_keys_per_request,
            summary.average_block_keys,
            speed_factor,
            replay_records,
            estimated_seconds, estimated_minutes,
            effective_qps,
            self._config.trace_loop,
            self._config.trace_loop_count if self._config.trace_loop_count > 0 else "unlimited",
            max_requests if max_requests > 0 else "unlimited",
        )
