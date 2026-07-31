#!/usr/bin/env python3
"""
ReportEvent / GetHostCacheState correctness and load generator.

The tool registers a fresh instance and a set of reporters, establishes the
required empty snapshot baseline, and then mixes:

* EVENT_BLOCK_ADD
* EVENT_BLOCK_DELETE
* EVENT_BLOCK_SNAPSHOT
* EVENT_HEARTBEAT
* standalone GetHostCacheState reads

Every ReportEvent call is followed by a GetHostCacheState query while holding
that reporter's local operation lock.  The response is compared with an
in-memory authoritative state, so a successful HTTP response alone is never
counted as a successful mutation.

Example:

    python3 tools/scripts/report_event_load.py \
        --base-url http://127.0.0.1:6382 \
        --instance-group vllm_kvcm_test_2 \
        --instance-id report_event_load_001 \
        --host-count 3 \
        --duration-sec 60 \
        --add-qps 30 \
        --delete-qps 30 \
        --get-qps 30 \
        --snapshot-interval-sec 35 \
        --add-batch-size 10 \
        --delete-batch-size 10 \
        --workers 64

Use a new instance id for each run.  The snapshot interval must not be shorter
than the EventReport backend's per-reporter snapshot rate limit.
"""

import argparse
import json
import os
import queue
import random
import statistics
import sys
import threading
import time
from typing import Dict, Iterable, List, Optional, Set, Tuple

try:
    import requests
except ImportError:
    print("请先安装 requests: pip install requests", file=sys.stderr)
    sys.exit(1)


OK_CODES = {"OK", "1", 1}
STAT_KINDS = ("add", "delete", "snapshot", "get", "heartbeat")
EVENT_REPORT_STORAGE_TYPES = (
    "ST_EVENT_REPORT_L2",
    "ST_EVENT_REPORT_L1P5",
)


def now_ms() -> float:
    return time.monotonic() * 1000.0


def percentile(sorted_values: List[float], p: float) -> float:
    if not sorted_values:
        return 0.0
    k = (len(sorted_values) - 1) * (p / 100.0)
    lower = int(k)
    upper = lower + 1
    if upper >= len(sorted_values):
        return sorted_values[-1]
    return sorted_values[lower] + (k - lower) * (
        sorted_values[upper] - sorted_values[lower]
    )


def is_ok_response(body: Dict) -> bool:
    code = body.get("header", {}).get("status", {}).get("code")
    return code in OK_CODES


def response_summary(body: Dict) -> str:
    try:
        return json.dumps(body, ensure_ascii=False)[:1000]
    except Exception:
        return str(body)[:1000]


def parse_cpu_cores(value: str) -> Set[int]:
    cores: Set[int] = set()
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        try:
            if "-" in part:
                start_str, end_str = part.split("-", 1)
                start = int(start_str)
                end = int(end_str)
                if start > end:
                    raise ValueError
                cores.update(range(start, end + 1))
            else:
                cores.add(int(part))
        except ValueError as exc:
            raise ValueError(
                f"--cpu-cores 格式错误: {value}，示例: 0-3 或 0,2,4"
            ) from exc
    if not cores or any(core < 0 for core in cores):
        raise ValueError(f"invalid cpu core list: {value}")
    return cores


def apply_cpu_affinity(cpu_cores: str) -> None:
    if not cpu_cores:
        return
    if not hasattr(os, "sched_setaffinity"):
        raise RuntimeError(
            "当前 Python/系统不支持 os.sched_setaffinity，无法设置 CPU affinity"
        )
    cores = parse_cpu_cores(cpu_cores)
    os.sched_setaffinity(0, cores)
    print(
        f"cpu affinity: {','.join(str(core) for core in sorted(cores))}",
        flush=True,
    )


class OpStats:
    def __init__(
        self,
        sent: int = 0,
        success: int = 0,
        failed: int = 0,
        skipped: int = 0,
        latencies_ms: Optional[List[float]] = None,
    ) -> None:
        self.sent = sent
        self.success = success
        self.failed = failed
        self.skipped = skipped
        self.latencies_ms = latencies_ms if latencies_ms is not None else []


class Stats:
    def __init__(self, max_errors: int) -> None:
        self._lock = threading.Lock()
        self._stats: Dict[str, OpStats] = {
            kind: OpStats() for kind in STAT_KINDS
        }
        self._errors: List[str] = []
        self._max_errors = max_errors

    def record_success(self, kind: str, latency_ms: float) -> None:
        with self._lock:
            stat = self._stats[kind]
            stat.sent += 1
            stat.success += 1
            stat.latencies_ms.append(latency_ms)

    def record_failure(self, kind: str, latency_ms: float, error: str) -> None:
        with self._lock:
            stat = self._stats[kind]
            stat.sent += 1
            stat.failed += 1
            stat.latencies_ms.append(latency_ms)
            if len(self._errors) < self._max_errors:
                self._errors.append(f"{kind}: {error}")

    def record_skipped(self, kind: str) -> None:
        with self._lock:
            self._stats[kind].skipped += 1

    def snapshot(self) -> Tuple[Dict[str, OpStats], List[str]]:
        with self._lock:
            stats_copy = {
                kind: OpStats(
                    sent=stat.sent,
                    success=stat.success,
                    failed=stat.failed,
                    skipped=stat.skipped,
                    latencies_ms=list(stat.latencies_ms),
                )
                for kind, stat in self._stats.items()
            }
            return stats_copy, list(self._errors)


class HostShadow:
    def __init__(self) -> None:
        self.lock = threading.RLock()
        self.keys: Set[int] = set()
        self.committed_snapshot_version = ""
        self.snapshot_not_before = 0.0


class ShadowState:
    def __init__(self, hosts: Iterable[str]) -> None:
        self._hosts = {host: HostShadow() for host in hosts}

    def host(self, host: str) -> HostShadow:
        return self._hosts[host]

    def pick_existing(self, host: str, count: int) -> Tuple[int, ...]:
        state = self.host(host)
        with state.lock:
            if not state.keys:
                return ()
            sample_size = min(count, len(state.keys))
            return tuple(random.sample(tuple(state.keys), sample_size))


class Task:
    def __init__(
        self, kind: str, host: str, block_keys: Tuple[int, ...] = ()
    ) -> None:
        self.kind = kind
        self.host = host
        self.block_keys = block_keys


class KVCMRequestError(RuntimeError):
    def __init__(self, body: Dict) -> None:
        self.body = body
        super().__init__(response_summary(body))

    @property
    def code(self):
        return self.body.get("header", {}).get("status", {}).get("code")

    @property
    def message(self) -> str:
        return self.body.get("header", {}).get("status", {}).get(
            "message", ""
        )

    @property
    def retry_after_ms(self) -> int:
        try:
            return int(self.body.get("retry_after_ms", 0))
        except (TypeError, ValueError):
            return 0


class KVCMHttpClient:
    def __init__(self, base_url: str, timeout: float) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers.update(
            {
                "Content-Type": "application/json",
                "Accept": "application/json",
            }
        )

    def post(self, api: str, payload: Dict) -> Dict:
        response = self.session.post(
            f"{self.base_url}{api}", json=payload, timeout=self.timeout
        )
        response.raise_for_status()
        body = response.json()
        if not is_ok_response(body):
            raise KVCMRequestError(body)
        return body

    def close(self) -> None:
        self.session.close()


def build_register_instance_payload(args: argparse.Namespace) -> Dict:
    return {
        "trace_id": "report_event_load/register_instance",
        "instance_group": args.instance_group,
        "instance_id": args.instance_id,
        "model_deployment": {
            "model_name": args.model_name,
            "dtype": args.dtype,
            "use_mla": args.use_mla,
            "tp_size": args.tp_size,
            "dp_size": args.dp_size,
            "pp_size": args.pp_size,
            "extra": args.extra,
            "user_data": args.user_data,
        },
        "block_size": args.block_size,
        "location_spec_infos": [
            {"name": args.spec_name, "size": args.spec_size}
        ],
        "query_type": "QT_PREFIX_MATCH",
    }


def build_report_request(
    args: argparse.Namespace,
    host: str,
    events: List[Dict],
    trace_id: str,
) -> Dict:
    return {
        "trace_id": trace_id,
        "instance_id": args.instance_id,
        "host_ip_port": host,
        "storage_type": args.storage_type,
        "events": events,
    }


def build_node_register_payload(
    args: argparse.Namespace, host: str
) -> Dict:
    return build_report_request(
        args,
        host,
        [
            {
                "event_type": "EVENT_NODE_REGISTER",
                "node_register": {"mediums": [args.medium]},
            }
        ],
        f"report_event_load/register_node/{host}",
    )


def build_heartbeat_payload(
    args: argparse.Namespace, host: str, seq: int
) -> Dict:
    return build_report_request(
        args,
        host,
        [
            {
                "event_type": "EVENT_HEARTBEAT",
                "heartbeat": {"system_status": {}},
            }
        ],
        f"report_event_load/heartbeat/{host}/{seq}",
    )


def event_report_uri(host: str, medium: str, block_key: int) -> str:
    return f"event_report://{host}/{medium}?block={block_key}"


def build_specs(
    args: argparse.Namespace, host: str, block_key: int
) -> List[Dict]:
    return [
        {
            "name": args.spec_name,
            "uri": event_report_uri(host, args.medium, block_key),
        }
    ]


def build_add_payload(
    args: argparse.Namespace,
    host: str,
    keys: Tuple[int, ...],
    seq: int,
) -> Dict:
    events = [
        {
            "event_type": "EVENT_BLOCK_ADD",
            "block_add": {
                "block_key": str(key),
                "medium": args.medium,
                "specs": build_specs(args, host, key),
            },
        }
        for key in keys
    ]
    return build_report_request(
        args, host, events, f"report_event_load/add/{seq}"
    )


def build_delete_payload(
    args: argparse.Namespace,
    host: str,
    keys: Tuple[int, ...],
    seq: int,
) -> Dict:
    events = [
        {
            "event_type": "EVENT_BLOCK_DELETE",
            "block_delete": {
                "block_key": str(key),
                "medium": args.medium,
                "spec_names": [args.spec_name],
            },
        }
        for key in keys
    ]
    return build_report_request(
        args, host, events, f"report_event_load/delete/{seq}"
    )


def build_snapshot_payload(
    args: argparse.Namespace,
    host: str,
    keys: Set[int],
    seq: int,
) -> Dict:
    blocks = [
        {
            "block_key": str(key),
            "medium": args.medium,
            "specs": build_specs(args, host, key),
        }
        for key in sorted(keys)
    ]
    return build_report_request(
        args,
        host,
        [
            {
                "event_type": "EVENT_BLOCK_SNAPSHOT",
                "block_snapshot": {"blocks": blocks},
            }
        ],
        f"report_event_load/snapshot/{seq}",
    )


def build_get_payload(
    args: argparse.Namespace,
    keys: Tuple[int, ...],
    seq: int,
    reason: str,
) -> Dict:
    return {
        "trace_id": f"report_event_load/get/{reason}/{seq}",
        "instance_id": args.instance_id,
        "query_type": "QT_PREFIX_MATCH",
        "block_cache_keys": list(keys),
        "medium": [args.medium],
    }


def parse_host_prefixes(response: Dict) -> Dict[str, int]:
    return {
        item["host_ip_port"]: int(item["prefix_match_blocks"])
        for item in response.get("hosts", [])
    }


def query_and_verify_prefix(
    client: KVCMHttpClient,
    args: argparse.Namespace,
    host: str,
    keys: Tuple[int, ...],
    expected_prefix: int,
    seq: int,
    reason: str,
) -> None:
    if not keys:
        raise AssertionError("GetHostCacheState verification requires keys")
    response = client.post(
        "/api/getHostCacheState",
        build_get_payload(args, keys, seq, reason),
    )
    actual_prefix = parse_host_prefixes(response).get(host, 0)
    if actual_prefix != expected_prefix:
        raise AssertionError(
            f"{reason}: host={host}, keys={keys[:20]}, "
            f"expected_prefix={expected_prefix}, actual_prefix={actual_prefix}, "
            f"response={response_summary(response)}"
        )


def sample_keys(keys: Set[int], limit: int) -> Tuple[int, ...]:
    if not keys:
        return ()
    if len(keys) <= limit:
        return tuple(sorted(keys))
    return tuple(sorted(random.sample(tuple(keys), limit)))


def verify_current_host_state(
    client: KVCMHttpClient,
    args: argparse.Namespace,
    host: str,
    state: HostShadow,
    seq: int,
    reason: str,
) -> None:
    present = sample_keys(state.keys, args.verify_keys_per_event)
    if present:
        query_and_verify_prefix(
            client,
            args,
            host,
            present,
            len(present),
            seq,
            f"{reason}/present",
        )
    else:
        query_and_verify_prefix(
            client,
            args,
            host,
            (args.key_base - 1,),
            0,
            seq,
            f"{reason}/empty",
        )


def verify_absent_keys(
    client: KVCMHttpClient,
    args: argparse.Namespace,
    host: str,
    keys: Set[int],
    seq: int,
    reason: str,
) -> None:
    for key in sample_keys(keys, args.verify_keys_per_event):
        query_and_verify_prefix(
            client, args, host, (key,), 0, seq, f"{reason}/absent"
        )


def validate_committed_version(
    response: Dict,
    expected: str,
    operation: str,
) -> None:
    actual = response.get("committed_snapshot_version", "")
    if actual != expected:
        raise AssertionError(
            f"{operation}: expected committed_snapshot_version={expected!r}, "
            f"actual={actual!r}, response={response_summary(response)}"
        )
    if response.get("snapshot_required"):
        raise AssertionError(
            f"{operation}: snapshot_required=true after baseline was committed"
        )


def validate_new_snapshot_version(
    response: Dict,
    previous: str,
    operation: str,
) -> str:
    actual = response.get("committed_snapshot_version", "")
    if (
        len(actual) != 32
        or any(char not in "0123456789abcdef" for char in actual)
        or actual == previous
    ):
        raise AssertionError(
            f"{operation}: invalid/new token expected, previous={previous!r}, "
            f"actual={actual!r}, response={response_summary(response)}"
        )
    if response.get("snapshot_required"):
        raise AssertionError(
            f"{operation}: snapshot_required=true after successful snapshot"
        )
    return actual


def expected_prefix(keys: Tuple[int, ...], existing: Set[int]) -> int:
    prefix = 0
    for key in keys:
        if key not in existing:
            break
        prefix += 1
    return prefix


def validate_args(args: argparse.Namespace) -> None:
    nonnegative = [
        ("--add-qps", args.add_qps),
        ("--delete-qps", args.delete_qps),
        ("--get-qps", args.get_qps),
        ("--snapshot-interval-sec", args.snapshot_interval_sec),
        ("--heartbeat-interval-sec", args.heartbeat_interval_sec),
    ]
    for name, value in nonnegative:
        if value < 0:
            raise ValueError(f"{name} 不能小于 0")

    positive_ints = [
        ("--duration-sec", args.duration_sec),
        ("--workers", args.workers),
        ("--queue-size", args.queue_size),
        ("--host-count", args.host_count),
        ("--key-space", args.key_space),
        ("--query-blocks", args.query_blocks),
        ("--add-batch-size", args.add_batch_size),
        ("--delete-batch-size", args.delete_batch_size),
        ("--verify-keys-per-event", args.verify_keys_per_event),
        ("--block-size", args.block_size),
        ("--spec-size", args.spec_size),
    ]
    for name, value in positive_ints:
        if value <= 0:
            raise ValueError(f"{name} 必须为正数")

    if not 0.0 <= args.snapshot_drop_ratio <= 1.0:
        raise ValueError("--snapshot-drop-ratio 必须在 [0, 1] 范围内")
    if (
        args.add_qps == 0
        and args.delete_qps == 0
        and args.get_qps == 0
        and args.snapshot_interval_sec == 0
    ):
        raise ValueError("至少需要一个增量、查询或周期 snapshot 负载")
    if args.storage_type not in ("auto",) + EVENT_REPORT_STORAGE_TYPES:
        raise ValueError(
            "--storage-type 必须为 auto、ST_EVENT_REPORT_L1P5 或 "
            "ST_EVENT_REPORT_L2"
        )
    if (
        args.add_qps == 0
        and args.delete_qps == 0
        and args.get_qps == 0
        and args.snapshot_interval_sec >= args.duration_sec
    ):
        raise ValueError(
            "ADD/DELETE/GET QPS 都是 0，且本次 duration 内不会触发周期 "
            "snapshot；请检查命令换行符或增大 --duration-sec"
        )


def make_hosts(count: int) -> List[str]:
    return [
        f"10.250.{index // 250}.{index % 250 + 1}:8080"
        for index in range(count)
    ]


def enqueue_with_rate(
    kind: str,
    qps: float,
    args: argparse.Namespace,
    hosts: List[str],
    task_queue: "queue.Queue[Task]",
    stop_event: threading.Event,
    stats: Stats,
    shadow: ShadowState,
) -> None:
    if qps <= 0:
        return
    interval = 1.0 / qps
    next_tick = time.monotonic()
    seq = 0
    while not stop_event.is_set():
        now = time.monotonic()
        if now < next_tick:
            stop_event.wait(min(next_tick - now, 0.05))
            continue
        next_tick += interval
        seq += 1
        host = random.choice(hosts)

        if kind == "add":
            keys = tuple(
                sorted(
                    {
                        args.key_base + random.randrange(args.key_space)
                        for _ in range(args.add_batch_size)
                    }
                )
            )
        elif kind == "delete":
            keys = shadow.pick_existing(host, args.delete_batch_size)
            if not keys:
                stats.record_skipped("delete")
                continue
        else:
            start = args.key_base + random.randrange(
                max(1, args.key_space - args.query_blocks + 1)
            )
            keys = tuple(start + offset for offset in range(args.query_blocks))

        try:
            task_queue.put(
                Task(kind=kind, host=host, block_keys=keys), timeout=0.1
            )
        except queue.Full:
            stats.record_skipped(kind)


def enqueue_snapshots(
    args: argparse.Namespace,
    hosts: List[str],
    task_queue: "queue.Queue[Task]",
    stop_event: threading.Event,
    stats: Stats,
    shadow: ShadowState,
) -> None:
    interval = args.snapshot_interval_sec
    if interval <= 0:
        return
    while not stop_event.wait(interval):
        for host in hosts:
            state = shadow.host(host)
            with state.lock:
                if time.monotonic() < state.snapshot_not_before:
                    stats.record_skipped("snapshot")
                    continue
            try:
                task_queue.put(
                    Task(kind="snapshot", host=host), timeout=0.1
                )
            except queue.Full:
                stats.record_skipped("snapshot")


def snapshot_keys_for_report(
    state: HostShadow, drop_ratio: float
) -> Tuple[Set[int], Set[int]]:
    new_keys = set(state.keys)
    if not new_keys or drop_ratio <= 0:
        return new_keys, set()
    drop_count = max(1, int(len(new_keys) * drop_ratio))
    drop_count = min(drop_count, len(new_keys))
    dropped = set(random.sample(tuple(new_keys), drop_count))
    new_keys.difference_update(dropped)
    return new_keys, dropped


def worker_loop(
    worker_id: int,
    args: argparse.Namespace,
    task_queue: "queue.Queue[Task]",
    stats: Stats,
    shadow: ShadowState,
) -> None:
    client = KVCMHttpClient(args.base_url, args.timeout)
    seq = worker_id * 1000000000
    try:
        while True:
            task = task_queue.get()
            if task.kind == "stop":
                task_queue.task_done()
                break

            seq += 1
            start_ms = now_ms()
            state = shadow.host(task.host)
            try:
                with state.lock:
                    if task.kind == "add":
                        response = client.post(
                            "/api/reportEvent",
                            build_add_payload(
                                args,
                                task.host,
                                task.block_keys,
                                seq,
                            ),
                        )
                        validate_committed_version(
                            response,
                            state.committed_snapshot_version,
                            "BLOCK_ADD",
                        )
                        state.keys.update(task.block_keys)
                        query_and_verify_prefix(
                            client,
                            args,
                            task.host,
                            task.block_keys,
                            len(task.block_keys),
                            seq,
                            "after_add",
                        )
                    elif task.kind == "delete":
                        response = client.post(
                            "/api/reportEvent",
                            build_delete_payload(
                                args,
                                task.host,
                                task.block_keys,
                                seq,
                            ),
                        )
                        validate_committed_version(
                            response,
                            state.committed_snapshot_version,
                            "BLOCK_DELETE",
                        )
                        state.keys.difference_update(task.block_keys)
                        verify_absent_keys(
                            client,
                            args,
                            task.host,
                            set(task.block_keys),
                            seq,
                            "after_delete",
                        )
                    elif task.kind == "snapshot":
                        snapshot_keys, dropped = snapshot_keys_for_report(
                            state, args.snapshot_drop_ratio
                        )
                        try:
                            response = client.post(
                                "/api/reportEvent",
                                build_snapshot_payload(
                                    args, task.host, snapshot_keys, seq
                                ),
                            )
                        except KVCMRequestError as error:
                            if error.code != "SNAPSHOT_RATE_LIMITED":
                                raise
                            retry_seconds = max(
                                error.retry_after_ms / 1000.0, 0.001
                            )
                            state.snapshot_not_before = max(
                                state.snapshot_not_before,
                                time.monotonic() + retry_seconds,
                            )
                            raise RuntimeError(
                                "SNAPSHOT_RATE_LIMITED: configured "
                                f"interval={args.snapshot_interval_sec}s is "
                                "shorter than this backend allows; "
                                f"retry_after_ms={error.retry_after_ms}. "
                                "Increase --snapshot-interval-sec and use a "
                                "longer --duration-sec."
                            ) from error
                        new_version = validate_new_snapshot_version(
                            response,
                            state.committed_snapshot_version,
                            "BLOCK_SNAPSHOT",
                        )
                        state.keys = snapshot_keys
                        state.committed_snapshot_version = new_version
                        state.snapshot_not_before = 0.0
                        verify_current_host_state(
                            client,
                            args,
                            task.host,
                            state,
                            seq,
                            "after_snapshot",
                        )
                        verify_absent_keys(
                            client,
                            args,
                            task.host,
                            dropped,
                            seq,
                            "after_snapshot_reconcile",
                        )
                    else:
                        expected = expected_prefix(
                            task.block_keys, state.keys
                        )
                        query_and_verify_prefix(
                            client,
                            args,
                            task.host,
                            task.block_keys,
                            expected,
                            seq,
                            "standalone_get",
                        )
                stats.record_success(task.kind, now_ms() - start_ms)
            except Exception as exc:
                stats.record_failure(
                    task.kind,
                    now_ms() - start_ms,
                    f"host={task.host}: {exc}",
                )
            finally:
                task_queue.task_done()
    finally:
        client.close()


def print_window_stats(
    start_time: float,
    previous: Dict[str, OpStats],
    current: Dict[str, OpStats],
) -> None:
    elapsed = time.monotonic() - start_time
    parts = [f"[{elapsed:7.1f}s]"]
    for kind in STAT_KINDS:
        success = current[kind].success - previous[kind].success
        failed = current[kind].failed - previous[kind].failed
        parts.append(
            f"{kind}: ok={success}/s fail={failed}/s "
            f"total={current[kind].sent}"
        )
    print("  ".join(parts), flush=True)


def print_final_stats(
    args: argparse.Namespace, stats: Stats, elapsed: float
) -> None:
    snapshot, errors = stats.snapshot()
    print("\n=== report_event_load summary ===")
    print(f"instance_id: {args.instance_id}")
    print(f"elapsed_sec: {elapsed:.2f}")
    for kind in STAT_KINDS:
        stat = snapshot[kind]
        latencies = sorted(stat.latencies_ms)
        qps = stat.success / elapsed if elapsed > 0 else 0.0
        average = statistics.mean(latencies) if latencies else 0.0
        print(
            f"{kind:9s} sent={stat.sent} success={stat.success} "
            f"failed={stat.failed} skipped={stat.skipped} "
            f"actual_qps={qps:.2f} "
            f"lat_ms(avg={average:.2f}, "
            f"p50={percentile(latencies, 50):.2f}, "
            f"p95={percentile(latencies, 95):.2f}, "
            f"p99={percentile(latencies, 99):.2f}, "
            f"max={(latencies[-1] if latencies else 0.0):.2f})"
        )
    if errors:
        print("\nfirst_errors:")
        for error in errors:
            print(f"  - {error}")


def event_report_backend_missing(error: KVCMRequestError) -> bool:
    return (
        error.code == "INSTANCE_NOT_EXIST"
        and "EventReportBackend not found" in error.message
    )


def register_first_reporter(
    client: KVCMHttpClient,
    args: argparse.Namespace,
    host: str,
) -> Dict:
    if args.storage_type != "auto":
        try:
            return client.post(
                "/api/reportEvent",
                build_node_register_payload(args, host),
            )
        except KVCMRequestError as error:
            if not event_report_backend_missing(error):
                raise
            raise RuntimeError(
                f"registerInstance succeeded, but instance group "
                f"{args.instance_group!r} has no {args.storage_type} "
                "EventReport backend. Use --storage-type auto or configure "
                "event_report_storage_candidates for this instance group. "
                f"Server response: {response_summary(error.body)}"
            ) from error

    missing_responses = []
    for storage_type in EVENT_REPORT_STORAGE_TYPES:
        args.storage_type = storage_type
        try:
            response = client.post(
                "/api/reportEvent",
                build_node_register_payload(args, host),
            )
            print(
                f"selected EventReport storage_type: {storage_type}",
                flush=True,
            )
            return response
        except KVCMRequestError as error:
            if not event_report_backend_missing(error):
                raise
            missing_responses.append(
                f"{storage_type}: {error.message or error.code}"
            )

    args.storage_type = "auto"
    raise RuntimeError(
        f"registerInstance succeeded, but instance group "
        f"{args.instance_group!r} exposes neither an L2 nor an L1P5 "
        "EventReport backend. Configure event_report_storage_candidates "
        "for the group or choose a group that already has EventReport "
        f"storage. Probe results: {'; '.join(missing_responses)}"
    )


def bootstrap(
    args: argparse.Namespace,
    hosts: List[str],
    shadow: ShadowState,
) -> None:
    client = KVCMHttpClient(args.base_url, args.timeout)
    try:
        client.post(
            "/api/registerInstance", build_register_instance_payload(args)
        )
        print(f"instance registered: {args.instance_id}", flush=True)
        for index, host in enumerate(hosts):
            state = shadow.host(host)
            with state.lock:
                if index == 0:
                    register_response = register_first_reporter(
                        client, args, host
                    )
                else:
                    register_response = client.post(
                        "/api/reportEvent",
                        build_node_register_payload(args, host),
                    )
                if not register_response.get("snapshot_required"):
                    if not args.allow_reuse_instance:
                        raise RuntimeError(
                            f"host {host} already has a committed snapshot. "
                            "Use a fresh --instance-id; pass "
                            "--allow-reuse-instance only when intentionally "
                            "replacing all reporter state."
                        )
                    print(
                        f"[WARN] reusing committed reporter state for {host}; "
                        "the initial empty snapshot will replace it",
                        file=sys.stderr,
                    )
                verify_current_host_state(
                    client,
                    args,
                    host,
                    state,
                    index,
                    "after_node_register",
                )

                snapshot_response = client.post(
                    "/api/reportEvent",
                    build_snapshot_payload(args, host, set(), index),
                )
                state.committed_snapshot_version = (
                    validate_new_snapshot_version(
                        snapshot_response,
                        "",
                        "INITIAL_BLOCK_SNAPSHOT",
                    )
                )
                verify_current_host_state(
                    client,
                    args,
                    host,
                    state,
                    index,
                    "after_initial_snapshot",
                )
    finally:
        client.close()


def heartbeat_loop(
    args: argparse.Namespace,
    hosts: List[str],
    shadow: ShadowState,
    stop_event: threading.Event,
    stats: Stats,
) -> None:
    interval = args.heartbeat_interval_sec
    if interval <= 0:
        return
    client = KVCMHttpClient(args.base_url, args.timeout)
    seq = 0
    try:
        while not stop_event.is_set():
            seq += 1
            for host in hosts:
                start_ms = now_ms()
                state = shadow.host(host)
                try:
                    with state.lock:
                        response = client.post(
                            "/api/reportEvent",
                            build_heartbeat_payload(args, host, seq),
                        )
                        validate_committed_version(
                            response,
                            state.committed_snapshot_version,
                            "HEARTBEAT",
                        )
                        verify_current_host_state(
                            client,
                            args,
                            host,
                            state,
                            seq,
                            "after_heartbeat",
                        )
                    stats.record_success(
                        "heartbeat", now_ms() - start_ms
                    )
                except Exception as exc:
                    stats.record_failure(
                        "heartbeat",
                        now_ms() - start_ms,
                        f"host={host}: {exc}",
                    )
            stop_event.wait(interval)
    finally:
        client.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "混合压测 ReportEvent ADD/DELETE/SNAPSHOT，并在每次事件后通过 "
            "GetHostCacheState 校验数据"
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--base-url",
        default="http://127.0.0.1:6382",
        help="KVCM MetaService HTTP 地址",
    )
    parser.add_argument(
        "--instance-group", required=True, help="已存在的 instance group"
    )
    parser.add_argument(
        "--instance-id", required=True, help="本次测试使用的全新 instance id"
    )
    parser.add_argument(
        "--allow-reuse-instance",
        action="store_true",
        help=(
            "允许复用已有 committed snapshot 的 instance；启动空 snapshot "
            "会覆盖 reporter 的全部旧状态"
        ),
    )
    parser.add_argument(
        "--storage-type",
        default="auto",
        help=(
            "ReportEvent storage_type；auto 会先探测 L2，再探测 L1P5，"
            "并明确报告 instance group 未配置 EventReport storage"
        ),
    )
    parser.add_argument(
        "--duration-sec", type=int, default=60, help="压测持续时间"
    )
    parser.add_argument(
        "--workers", type=int, default=32, help="HTTP worker 数"
    )
    parser.add_argument(
        "--queue-size",
        type=int,
        default=10000,
        help="待发送任务队列长度",
    )
    parser.add_argument(
        "--timeout", type=float, default=10.0, help="单请求超时时间，秒"
    )
    parser.add_argument(
        "--max-errors", type=int, default=20, help="最多打印多少条错误样例"
    )
    parser.add_argument(
        "--cpu-cores",
        default="",
        help="限制进程使用的 CPU 核，例如 0-3 或 0,2,4；默认不限制",
    )

    parser.add_argument(
        "--add-qps", type=float, default=0.0, help="BLOCK_ADD 请求 QPS"
    )
    parser.add_argument(
        "--delete-qps",
        type=float,
        default=0.0,
        help="BLOCK_DELETE 请求 QPS",
    )
    parser.add_argument(
        "--get-qps",
        type=float,
        default=0.0,
        help="独立 GetHostCacheState 请求 QPS",
    )
    parser.add_argument(
        "--snapshot-interval-sec",
        type=float,
        default=35.0,
        help="每个 host 周期 snapshot 间隔；0 表示只发启动 snapshot",
    )
    parser.add_argument(
        "--snapshot-drop-ratio",
        type=float,
        default=0.0,
        help=(
            "周期 snapshot 主动遗漏现有 key 的比例，用于验证全量对账删除；"
            "影子状态会同步更新"
        ),
    )
    parser.add_argument(
        "--add-batch-size",
        type=int,
        default=1,
        help="每个 BLOCK_ADD 请求包含多少个 block",
    )
    parser.add_argument(
        "--delete-batch-size",
        type=int,
        default=1,
        help="每个 BLOCK_DELETE 请求包含多少个 block",
    )
    parser.add_argument(
        "--verify-keys-per-event",
        type=int,
        default=16,
        help="每次事件后最多抽样校验多少个 block",
    )

    parser.add_argument(
        "--host-count", type=int, default=16, help="模拟 host 数"
    )
    parser.add_argument(
        "--key-base",
        type=int,
        default=1000000000,
        help="压测 key 起始值",
    )
    parser.add_argument(
        "--key-space",
        type=int,
        default=1000000,
        help="随机 key 空间大小",
    )
    parser.add_argument(
        "--query-blocks",
        type=int,
        default=32,
        help="每次独立 GetHostCacheState 查询的连续 block 数",
    )
    parser.add_argument(
        "--medium", default="mem", help="ReportEvent block medium"
    )
    parser.add_argument(
        "--spec-name", default="tp0", help="LocationSpec name"
    )
    parser.add_argument(
        "--spec-size",
        type=int,
        default=1024,
        help="registerInstance LocationSpecInfo size",
    )

    parser.add_argument(
        "--model-name",
        default="report_event_load",
        help="registerInstance model_name",
    )
    parser.add_argument(
        "--dtype", default="FP8", help="registerInstance dtype"
    )
    parser.add_argument(
        "--use-mla", action="store_true", help="registerInstance use_mla"
    )
    parser.add_argument(
        "--tp-size",
        type=int,
        default=1,
        help="registerInstance tp_size",
    )
    parser.add_argument(
        "--dp-size",
        type=int,
        default=1,
        help="registerInstance dp_size",
    )
    parser.add_argument(
        "--pp-size",
        type=int,
        default=1,
        help="registerInstance pp_size",
    )
    parser.add_argument(
        "--extra", default="", help="registerInstance model extra"
    )
    parser.add_argument(
        "--user-data",
        default="",
        help="registerInstance model user_data",
    )
    parser.add_argument(
        "--block-size",
        type=int,
        default=128,
        help="registerInstance block_size",
    )
    parser.add_argument(
        "--heartbeat-interval-sec",
        type=float,
        default=10.0,
        help="心跳发送间隔，秒；0 表示不发心跳",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        validate_args(args)
        apply_cpu_affinity(args.cpu_cores)
    except (ValueError, RuntimeError) as exc:
        print(f"参数或环境错误: {exc}", file=sys.stderr)
        return 2

    hosts = make_hosts(args.host_count)
    shadow = ShadowState(hosts)
    print(
        f"bootstrap: base_url={args.base_url} "
        f"instance_group={args.instance_group} "
        f"instance_id={args.instance_id} "
        f"storage_type={args.storage_type} hosts={len(hosts)}",
        flush=True,
    )
    try:
        bootstrap(args, hosts, shadow)
    except Exception as exc:
        print(f"bootstrap failed: {exc}", file=sys.stderr)
        return 1

    stats = Stats(args.max_errors)
    task_queue: "queue.Queue[Task]" = queue.Queue(
        maxsize=args.queue_size
    )
    stop_event = threading.Event()

    workers = [
        threading.Thread(
            target=worker_loop,
            args=(
                index,
                args,
                task_queue,
                stats,
                shadow,
            ),
            name=f"worker-{index}",
            daemon=True,
        )
        for index in range(args.workers)
    ]
    schedulers = [
        threading.Thread(
            target=enqueue_with_rate,
            args=(
                kind,
                qps,
                args,
                hosts,
                task_queue,
                stop_event,
                stats,
                shadow,
            ),
            name=f"scheduler-{kind}",
            daemon=True,
        )
        for kind, qps in (
            ("add", args.add_qps),
            ("delete", args.delete_qps),
            ("get", args.get_qps),
        )
    ]
    snapshot_scheduler = threading.Thread(
        target=enqueue_snapshots,
        args=(args, hosts, task_queue, stop_event, stats, shadow),
        name="scheduler-snapshot",
        daemon=True,
    )
    heartbeat_thread = threading.Thread(
        target=heartbeat_loop,
        args=(args, hosts, shadow, stop_event, stats),
        name="heartbeat",
        daemon=True,
    )

    start = time.monotonic()
    previous, _ = stats.snapshot()
    for thread in (
        workers + schedulers + [snapshot_scheduler, heartbeat_thread]
    ):
        thread.start()

    try:
        while time.monotonic() - start < args.duration_sec:
            time.sleep(1.0)
            current, _ = stats.snapshot()
            print_window_stats(start, previous, current)
            previous = current
    except KeyboardInterrupt:
        print("received interrupt, stopping...", file=sys.stderr)
    finally:
        stop_event.set()
        for thread in schedulers + [snapshot_scheduler, heartbeat_thread]:
            thread.join()
        task_queue.join()
        for _ in workers:
            task_queue.put(Task(kind="stop", host=hosts[0]))
        for thread in workers:
            thread.join()

    elapsed = time.monotonic() - start
    print_final_stats(args, stats, elapsed)
    snapshot, _ = stats.snapshot()
    return (
        1
        if any(snapshot[kind].failed > 0 for kind in STAT_KINDS)
        else 0
    )


if __name__ == "__main__":
    sys.exit(main())
