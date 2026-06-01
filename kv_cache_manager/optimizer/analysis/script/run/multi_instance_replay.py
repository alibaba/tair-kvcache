#!/usr/bin/env python3
"""
Replay per-instance optimizer traces with one optimizer process per instance.

Each input JSONL must contain traces for one optimizer instance.
The script writes one generated optimizer config per instance, runs each instance in a
separate process, and aggregates the emitted *_hit_rates.csv files with the
standard token hit-rate semantics.
"""

import argparse
import gc
import hashlib
import json
import os
import re
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import List, Optional

def _parse_tier_flow_config_arg(parser, raw_value):
    if not raw_value:
        return []
    config_text = Path(raw_value).read_text() if os.path.exists(raw_value) else raw_value
    try:
        tier_flows = json.loads(config_text)
    except json.JSONDecodeError as exc:
        parser.error(f"--tier-flow-config must be a JSON array or a path to one: {exc}")
    if not isinstance(tier_flows, list):
        parser.error("--tier-flow-config must be a JSON array")
    valid_write_modes = {"write_through", "cascading", "write_through_selective"}
    for idx, flow in enumerate(tier_flows):
        if not isinstance(flow, dict):
            parser.error(f"--tier-flow-config[{idx}] must be an object")
        if not isinstance(flow.get("from_tier"), str) or not flow["from_tier"]:
            parser.error(f"--tier-flow-config[{idx}] must contain non-empty from_tier")
        if not isinstance(flow.get("to_tier"), str) or not flow["to_tier"]:
            parser.error(f"--tier-flow-config[{idx}] must contain non-empty to_tier")
        if "write_mode" in flow and flow["write_mode"] not in valid_write_modes:
            parser.error(f"--tier-flow-config[{idx}].write_mode must be one of {sorted(valid_write_modes)}")
        for bool_key in ("access_propagation_enabled", "promote_enabled"):
            if bool_key in flow and type(flow[bool_key]) is not bool:
                parser.error(f"--tier-flow-config[{idx}].{bool_key} must be a boolean")
        if "selective_write_threshold" in flow:
            threshold = flow["selective_write_threshold"]
            if type(threshold) is not int or threshold <= 0:
                parser.error(f"--tier-flow-config[{idx}].selective_write_threshold must be a positive integer")
    return tier_flows


def _configure_bazel_run_output():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(line_buffering=True)
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(line_buffering=True)
    os.environ.setdefault("KVCM_LOG_TO_CONSOLE", "1")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Replay per-instance optimizer traces with one process per instance"
    )
    inputs = parser.add_mutually_exclusive_group()
    inputs.add_argument("--trace-dir", help="Directory containing per-instance JSONL trace files")
    inputs.add_argument("--trace-files", nargs="+", help="Explicit per-instance JSONL trace files")
    parser.add_argument("--trace-glob", default="*.jsonl", help="Glob used with --trace-dir")
    parser.add_argument("--recursive", action="store_true", help="Use recursive glob under --trace-dir")
    parser.add_argument("--output-dir", required=True, help="Output directory")
    parser.add_argument("--bucket-name", default="", help="Bucket name recorded in aggregate CSVs")
    parser.add_argument("--max-workers", type=int, default=0,
                        help="Maximum optimizer processes. Default: min(cpu_count, instance_count)")
    parser.add_argument("--skip-existing", action="store_true",
                        help="Skip an instance if its hit_rates CSV already exists")

    parser.add_argument("--l1-capacity", type=float, default=50.0,
                        help="Tier 0 capacity per instance in GB")
    parser.add_argument("--l2-capacity", type=float, default=128.0,
                        help="Tier 1 capacity per instance in GB. Use 0 to disable")
    parser.add_argument("--block-size", type=int, default=16, help="Block size in tokens")
    parser.add_argument("--bytes-per-token", type=int, default=512, help="Bytes per token")
    parser.add_argument("--eviction-policy", default="lru",
                        choices=["lru", "random_lru", "leaf_aware_lru", "ttl"],
                        help="Eviction policy per instance")
    parser.add_argument("--eviction-policy-params", default="",
                        help="JSON object overriding eviction_policy_params")
    parser.add_argument("--eviction-mode", type=int, default=3,
                        help="1=group rough, 2=instance rough, 3=instance precise")
    parser.add_argument("--eviction-batch-size", type=int, default=100,
                        help="Eviction batch size per instance")
    parser.add_argument("--default-tier-write-mode", dest="default_tier_write_mode", default="write_through",
                        choices=[
                            "write_through",
                            "cascading",
                            "write_through_selective",
                        ],
                        help="Default tier_strategy.write_mode for adjacent tier edges")
    parser.add_argument("--selective-write-threshold", type=int, default=2,
                        help="Access-count threshold for write_through_selective tier writes")
    parser.add_argument("--tier-flow-config", default=None,
                        help="JSON array or file path for tier_strategy.tier_flows edge overrides")
    access_group = parser.add_mutually_exclusive_group()
    access_group.add_argument("--enable-tier-access-propagation",
                              dest="tier_access_propagation_enabled",
                              action="store_true",
                              default=True,
                              help="Refresh lower-tier access metadata when an upper-tier copy is hit")
    access_group.add_argument("--disable-tier-access-propagation",
                              dest="tier_access_propagation_enabled",
                              action="store_false",
                              help="Refresh only the hit tier when a block has copies in multiple tiers")
    promote_group = parser.add_mutually_exclusive_group()
    promote_group.add_argument("--enable-promote", dest="enable_promote", action="store_true", default=True)
    promote_group.add_argument("--disable-promote", dest="enable_promote", action="store_false")
    parser.add_argument("--disable-hierarchical-eviction", action="store_true")
    parser.add_argument("--used-percentage", type=float, default=1.0)
    parser.add_argument("--default-block-ttl-seconds", type=int, default=0)
    ttl_group = parser.add_mutually_exclusive_group()
    ttl_group.add_argument("--ttl-refresh-on-read", dest="ttl_refresh_on_read", action="store_true", default=True)
    ttl_group.add_argument("--no-ttl-refresh-on-read", dest="ttl_refresh_on_read", action="store_false")
    parser.add_argument("--write-delay-ns", type=int, default=1,
                        help="Delay from request read timestamp to generated write timestamp for type=request traces")

    parser.add_argument("--aggregate-only", action="store_true",
                        help="Skip replay and aggregate existing hit_rates CSVs")
    parser.add_argument("--start-ns", type=int, default=None)
    parser.add_argument("--end-ns", type=int, default=None)
    parser.add_argument("--window-ns", type=int, default=None)
    parser.add_argument("--window-seconds", type=float, default=None)
    parser.add_argument("--include-instance-windows", action="store_true")
    parser.add_argument("--aggregation-chunksize", type=int, default=1_000_000)
    parser.add_argument("--log-level", type=int, default=4)
    args = parser.parse_args()
    if not args.aggregate_only and not args.trace_dir and not args.trace_files:
        parser.error("one of --trace-dir or --trace-files is required")
    if args.selective_write_threshold <= 0:
        parser.error("--selective-write-threshold must be positive")
    if args.write_delay_ns <= 0:
        parser.error("--write-delay-ns must be positive")
    args.tier_flow_config = _parse_tier_flow_config_arg(parser, args.tier_flow_config)
    return args


def main():
    from utils.window_aggregator import aggregate_and_write, collect_hit_rate_csvs

    _configure_bazel_run_output()
    args = parse_args()
    total_start = time.time()

    output_dir = os.path.abspath(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)
    config_dir = os.path.join(output_dir, "configs")
    os.makedirs(config_dir, exist_ok=True)

    trace_files = [] if args.aggregate_only else _resolve_trace_files(args)
    bucket_name = args.bucket_name or _default_bucket_name(args)

    print("\n" + "=" * 80)
    print("  Multi-Instance Replay")
    print("=" * 80)
    print(f"  Output:       {output_dir}")
    print(f"  Bucket:       {bucket_name}")
    print(f"  Mode:         {'aggregate-only' if args.aggregate_only else 'replay'}")
    if not args.aggregate_only:
        print(f"  Instance traces: {len(trace_files)}")

    if not args.aggregate_only:
        tasks, csv_files = _prepare_tasks(args, trace_files, config_dir, output_dir)
        if tasks:
            max_workers = args.max_workers if args.max_workers > 0 else min(os.cpu_count() or 1, len(tasks))
            max_workers = max(1, max_workers)
            print(f"  Workers:      {max_workers}")
            print("\n[1/2] Running instance optimizer processes...")
            _run_tasks(tasks, max_workers)
        else:
            print("No multi-instance replay tasks to run.")
    else:
        print("\n[1/2] Skipping replay.")
        csv_files = collect_hit_rate_csvs(output_dir)

    print("\n[2/2] Aggregating instance hit-rate CSVs...")
    if not csv_files:
        print(f"No *_hit_rates.csv files found in {output_dir}")
        sys.exit(1)
    _validate_hit_rate_csvs(csv_files)

    result = aggregate_and_write(
        csv_files=csv_files,
        output_dir=output_dir,
        bucket_name=bucket_name,
        start_ns=args.start_ns,
        end_ns=args.end_ns,
        window_ns=_resolve_window_ns(args),
        chunksize=args.aggregation_chunksize,
        include_instance_windows=args.include_instance_windows,
    )

    global_row = result["global"]
    print(f"  Instance aggregate:    {result['instance_summary_path']}")
    print(f"  Global aggregate: {result['global_summary_path']}")
    if "global_window_path" in result:
        print(f"  Window aggregate: {result['global_window_path']}")
    if "instance_window_path" in result:
        print(f"  Instance windows:      {result['instance_window_path']}")
    print("")
    print(f"  Instances:         {result['instance_count']}")
    print(f"  Requests:     {global_row['RequestCount']:,}")
    print(f"  Read blocks:  {global_row['ReadBlocks']:,}")
    print(f"  Hit blocks:   {global_row['HitBlocks']:,}")
    print(f"  Input tokens: {global_row['InputTokens']:,}")
    print(f"  Hit tokens:   {global_row['HitTokens']:,}")
    print(f"  Hit rate:     {global_row['HitRate']:.6f}")
    print(f"\nTotal time: {time.time() - total_start:.2f}s")


def _resolve_trace_files(args) -> List[str]:
    if args.trace_files:
        paths = [os.path.abspath(p) for p in args.trace_files]
    else:
        root = Path(args.trace_dir)
        pattern = f"**/{args.trace_glob}" if args.recursive else args.trace_glob
        paths = [str(p.resolve()) for p in root.glob(pattern) if p.is_file()]
    paths = sorted(paths)
    if not paths:
        raise SystemExit("No trace files found")
    return paths


def _default_bucket_name(args) -> str:
    if args.trace_dir:
        return os.path.basename(os.path.abspath(args.trace_dir))
    if args.aggregate_only:
        return os.path.basename(os.path.abspath(args.output_dir))
    return ""


def _prepare_tasks(args, trace_files: List[str], config_dir: str, output_dir: str) -> tuple:
    tasks = []
    csv_files = {}
    seen_instance_ids = {}
    policy_params = _resolve_policy_params(args.eviction_policy, args.eviction_policy_params)

    for trace_file in trace_files:
        instance_id = _inspect_optimizer_trace(trace_file)
        if "/" in instance_id or "\\" in instance_id:
            raise SystemExit(f"instance_id {instance_id!r} contains a path separator")
        if instance_id in seen_instance_ids:
            raise SystemExit(
                "Duplicate instance_id '{}' inferred from both '{}' and '{}'"
                .format(instance_id, seen_instance_ids[instance_id], trace_file)
            )
        seen_instance_ids[instance_id] = trace_file

        csv_path = os.path.join(output_dir, f"{instance_id}_hit_rates.csv")
        csv_files[instance_id] = csv_path
        if args.skip_existing and os.path.isfile(csv_path):
            print(f"  skip existing: {instance_id} -> {csv_path}")
            continue
        if os.path.exists(csv_path):
            if not os.path.isfile(csv_path):
                raise SystemExit(f"Expected hit-rate CSV path is not a file: {csv_path}")
            os.remove(csv_path)

        config = _make_single_instance_config(args, trace_file, output_dir, instance_id, policy_params)
        config_path = os.path.join(config_dir, f"{_config_file_name(instance_id)}.json")
        with open(config_path, "w") as f:
            json.dump(config, f, indent=2)

        tasks.append({
            "instance_id": instance_id,
            "trace_file": trace_file,
            "config_path": config_path,
            "csv_path": csv_path,
            "log_level": args.log_level,
        })

    task_csv = os.path.join(output_dir, "multi_instance_replay_tasks.csv")
    with open(task_csv, "w") as f:
        f.write("InstanceId,TraceFile,ConfigPath,CsvPath\n")
        for task in tasks:
            f.write("{},{},{},{}\n".format(
                task["instance_id"],
                task["trace_file"],
                task["config_path"],
                task["csv_path"],
            ))
    print(f"  Task manifest: {task_csv}")
    return tasks, csv_files


def _run_tasks(tasks: List[dict], max_workers: int) -> None:
    summary_path = os.path.join(os.path.dirname(os.path.dirname(tasks[0]["config_path"])), "multi_instance_replay_summary.csv")
    results = []
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(_run_instance_worker, task): task for task in tasks}
        for idx, future in enumerate(as_completed(futures), start=1):
            task = futures[future]
            try:
                result = future.result()
            except Exception as exc:
                result = {
                    "success": False,
                    "instance_id": task["instance_id"],
                    "trace_file": task["trace_file"],
                    "config_path": task["config_path"],
                    "csv_path": task["csv_path"],
                    "elapsed_seconds": 0.0,
                    "error": repr(exc),
                }
            results.append(result)
            status = "OK" if result["success"] else "FAIL"
            print("[{}/{}] {} {} ({:.2f}s)".format(
                idx, len(tasks), status, result["instance_id"], result["elapsed_seconds"]
            ))
            if result.get("error"):
                print(f"      {result['error']}")

    with open(summary_path, "w") as f:
        f.write("InstanceId,Success,ElapsedSeconds,TraceFile,ConfigPath,CsvPath,Error\n")
        for result in sorted(results, key=lambda r: r["instance_id"]):
            f.write("{},{},{:.6f},{},{},{},{}\n".format(
                result["instance_id"],
                int(result["success"]),
                result["elapsed_seconds"],
                result["trace_file"],
                result["config_path"],
                result["csv_path"],
                str(result.get("error", "")).replace("\n", "\\n").replace(",", ";"),
            ))
    print(f"  Replay summary: {summary_path}")

    failed = [r for r in results if not r["success"]]
    if failed:
        raise SystemExit(f"{len(failed)}/{len(results)} multi-instance replay tasks failed")


def _run_instance_worker(task: dict) -> dict:
    start = time.time()
    try:
        _configure_bazel_run_output()
        from kv_cache_manager.optimizer.pybind import kvcm_py_optimizer

        kvcm_py_optimizer.LoggerBroker.InitLogger("", False)
        kvcm_py_optimizer.LoggerBroker.SetLogLevel(int(task["log_level"]))

        config_loader = kvcm_py_optimizer.OptimizerConfigLoader()
        if not config_loader.load(task["config_path"]):
            raise RuntimeError(f"Failed to load config: {task['config_path']}")
        config = config_loader.config()

        manager = kvcm_py_optimizer.OptimizerManager(config)
        if not manager.Init():
            raise RuntimeError(f"OptimizerManager.Init failed: {task['config_path']}")
        manager.DirectRun()
        manager.AnalyzeResults()
        if not os.path.isfile(task["csv_path"]):
            raise RuntimeError(f"Expected hit-rate CSV was not produced: {task['csv_path']}")
        manager.ClearAllCachesAndResetStats()
        del manager
        gc.collect()

        return {
            "success": True,
            "instance_id": task["instance_id"],
            "trace_file": task["trace_file"],
            "config_path": task["config_path"],
            "csv_path": task["csv_path"],
            "elapsed_seconds": time.time() - start,
            "error": "",
        }
    except Exception as exc:
        return {
            "success": False,
            "instance_id": task["instance_id"],
            "trace_file": task["trace_file"],
            "config_path": task["config_path"],
            "csv_path": task["csv_path"],
            "elapsed_seconds": time.time() - start,
            "error": "{}\n{}".format(repr(exc), traceback.format_exc()),
        }


def _validate_hit_rate_csvs(csv_files: dict) -> None:
    missing = [
        f"{instance_id}: {csv_path}"
        for instance_id, csv_path in sorted(csv_files.items())
        if not os.path.isfile(csv_path)
    ]
    if missing:
        raise SystemExit("Missing expected hit-rate CSV(s):\n  " + "\n  ".join(missing))


def _inspect_optimizer_trace(trace_file: str) -> str:
    instance_id = None
    row_count = 0
    with open(trace_file, "r") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                raise SystemExit(f"{trace_file}:{line_no} invalid JSON: {exc}") from exc

            row_count += 1
            trace_type = obj.get("type")
            if trace_type not in {"get", "write", "request"}:
                raise SystemExit(f"{trace_file}:{line_no} has invalid type={trace_type!r}")
            current_instance_id = obj.get("instance_id")
            if not isinstance(current_instance_id, str) or not current_instance_id:
                raise SystemExit(f"{trace_file}:{line_no} has no non-empty instance_id")
            if instance_id is None:
                instance_id = current_instance_id
            elif current_instance_id != instance_id:
                raise SystemExit(
                    f"{trace_file}:{line_no} has instance_id={current_instance_id!r}, "
                    f"expected {instance_id!r}"
                )
            if "timestamp_ns" not in obj or type(obj["timestamp_ns"]) is not int:
                raise SystemExit(f"{trace_file}:{line_no} must contain integer timestamp_ns")
            if trace_type in {"get", "request"}:
                if "input_len" not in obj or type(obj["input_len"]) is not int or obj["input_len"] <= 0:
                    raise SystemExit(
                        f"{trace_file}:{line_no} {trace_type} trace must contain positive integer input_len")
            keys = obj.get("keys")
            if not isinstance(keys, list):
                raise SystemExit(f"{trace_file}:{line_no} must contain keys array")
    if row_count == 0 or instance_id is None:
        raise SystemExit(f"{trace_file} has no valid optimizer trace rows")
    return instance_id


def _make_single_instance_config(args, trace_file: str, output_dir: str, instance_id: str, policy_params: dict) -> dict:
    storages = [
        {
            "unique_name": "hbm",
            "storage_type": "pace",
            "band_width_mbps": 20000,
            "priority": 0,
            "capacity": args.l1_capacity,
        }
    ]
    if args.l2_capacity > 0:
        storages.append({
            "unique_name": "dram",
            "storage_type": "hf3fs",
            "band_width_mbps": 20000,
            "priority": 1,
            "capacity": args.l2_capacity,
        })

    tier_strategy = {
        "hierarchical_eviction_enabled": not args.disable_hierarchical_eviction,
        "write_mode": args.default_tier_write_mode,
        "access_propagation_enabled": args.tier_access_propagation_enabled,
        "promote_enabled": args.enable_promote,
        "selective_write_threshold": args.selective_write_threshold,
    }
    if args.tier_flow_config:
        tier_strategy["tier_flows"] = args.tier_flow_config

    return {
        "trace_file_path": trace_file,
        "output_result_path": output_dir,
        "eviction_params": {
            "eviction_mode": args.eviction_mode,
            "eviction_batch_size_per_instance": args.eviction_batch_size,
        },
        "trace_replay": {
            "write_delay_ns": args.write_delay_ns,
        },
        "instance_groups": [
            {
                "group_name": instance_id,
                "quota_capacity": args.l1_capacity + max(args.l2_capacity, 0.0),
                "used_percentage": args.used_percentage,
                "tier_strategy": tier_strategy,
                "default_block_ttl_seconds": args.default_block_ttl_seconds,
                "ttl_refresh_on_read": args.ttl_refresh_on_read,
                "storages": storages,
                "instances": [
                    {
                        "instance_id": instance_id,
                        "block_size": args.block_size,
                        "bytes_per_token": args.bytes_per_token,
                        "eviction_policy_type": args.eviction_policy,
                        "eviction_policy_params": policy_params,
                    }
                ],
            }
        ],
    }


def _resolve_policy_params(policy: str, override_json: str) -> dict:
    if override_json:
        value = json.loads(override_json)
        if not isinstance(value, dict):
            raise SystemExit("--eviction-policy-params must be a JSON object")
        return value
    if policy == "random_lru":
        return {"sample_rate": 1.0}
    if policy == "ttl":
        return {"fallback_on_pressure": True}
    return {
        "sample_rate": 1.0,
        "shard_count": 1,
        "sample_times": 32,
        "eviction_amplification_factor": 1.0,
    }


def _resolve_window_ns(args) -> Optional[int]:
    if args.window_ns is not None and args.window_seconds is not None:
        raise SystemExit("Use only one of --window-ns or --window-seconds")
    if args.window_ns is not None:
        return args.window_ns
    if args.window_seconds is not None:
        if args.window_seconds <= 0:
            raise SystemExit("--window-seconds must be positive")
        return int(args.window_seconds * 1_000_000_000)
    return None


def _safe_name(value: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", value)
    return safe or "instance"


def _config_file_name(instance_id: str) -> str:
    digest = hashlib.sha256(instance_id.encode("utf-8")).hexdigest()[:12]
    return f"{_safe_name(instance_id)}-{digest}"


if __name__ == "__main__":
    main()
