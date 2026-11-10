#!/usr/bin/env python3
"""Standalone process smoke: python3 service_dispatch_smoke.py --binary /path/to/kvcm.

Uses a temporary cwd, memory coordination and a local registry without persistence.
The only storage root is inside that temporary cwd. Bazel supplies the gRPC stubs.
"""

import argparse
from concurrent.futures import ThreadPoolExecutor
import http.client
import json
import os
from pathlib import Path
import re
import shutil
import signal
import socket
import subprocess
import tempfile
import time

import grpc
from kv_cache_manager.protocol.protobuf import admin_service_pb2, admin_service_pb2_grpc
from kv_cache_manager.protocol.protobuf import meta_service_pb2, meta_service_pb2_grpc


def check(condition, message):
    if not condition:
        raise AssertionError(message)


def http_json(port, path, payload=None):
    connection = http.client.HTTPConnection("127.0.0.1", port, timeout=1)
    try:
        body = None if payload is None else json.dumps(payload)
        connection.request("GET" if body is None else "POST", path, body,
                           {"Content-Type": "application/json"})
        response = connection.getresponse()
        return response.status, json.loads(response.read())
    finally:
        connection.close()


def check_endpoints(process, ports, timeout):
    deadline = time.monotonic() + timeout
    while True:
        check(process.poll() is None, "server exited before HTTP readiness")
        try:
            status, body = http_json(ports[3], "/api/healthy")
            if status == 200 and body == {"status": "OK"}:
                status, cluster = http_json(ports[1], "/api/getClusterInfo",
                                            {"trace_id": "dispatch-smoke", "instance_id": "missing"})
                if status == 200:
                    break
        except (OSError, http.client.HTTPException):
            pass
        check(time.monotonic() < deadline, "HTTP readiness timed out")
        time.sleep(0.05)
    check(cluster["header"]["status"]["code"] == "OK", cluster)
    check(cluster["header"]["request_id"] and cluster["self_node_id"], cluster)
    status, missing = http_json(ports[1], "/api/getCacheLocation",
                               {"trace_id": "dispatch-smoke-missing", "instance_id": "missing"})
    check(status == 200 and missing["header"]["status"]["code"] == "INSTANCE_NOT_EXIST", missing)
    status, health = http_json(ports[3], "/api/checkHealth", {"trace_id": "dispatch-smoke-admin"})
    check(status == 200 and health["header"]["status"]["code"] == "OK" and health["is_health"], health)
    with grpc.insecure_channel(f"127.0.0.1:{ports[0]}") as channel:
        meta = meta_service_pb2_grpc.MetaServiceStub(channel)
        cluster = meta.GetClusterInfo(meta_service_pb2.GetClusterInfoRequest(trace_id="smoke-grpc"), timeout=2)
        check(cluster.header.status.code == meta_service_pb2.OK and cluster.self_node_id, cluster)
        try:
            channel.unary_unary("/kv_cache_manager.proto.meta.MetaService/UnknownMethod")(b"", timeout=2)
            raise AssertionError("unknown method unexpectedly succeeded")
        except grpc.RpcError as error:
            check(error.code() == grpc.StatusCode.UNIMPLEMENTED, error)
    with grpc.insecure_channel(f"127.0.0.1:{ports[2]}") as channel:
        admin = admin_service_pb2_grpc.AdminServiceStub(channel)
        health = admin.CheckHealth(admin_service_pb2.CheckHealthRequest(trace_id="smoke-admin-grpc"), timeout=2)
        check(health.header.status.code == admin_service_pb2.OK and health.is_health, health)


def check_all_meta_methods(ports):
    # Verify every generated unary method is actually posted on the async CQ,
    # and every corresponding HTTP route still produces a business response.
    methods = meta_service_pb2.DESCRIPTOR.services_by_name["MetaService"].methods
    with grpc.insecure_channel(f"127.0.0.1:{ports[0]}") as channel:
        stub = meta_service_pb2_grpc.MetaServiceStub(channel)
        for method in methods:
            request_type = getattr(meta_service_pb2, method.input_type.name)
            request = request_type(trace_id=f"smoke-all-{method.name}")
            response = getattr(stub, method.name)(request, timeout=3)
            check(response.HasField("header") and response.header.HasField("status"), response)
            check(response.header.status.code != meta_service_pb2.SERVICE_NOT_READY, response)
            route = method.name[0].lower() + method.name[1:]
            status, body = http_json(ports[1], f"/api/{route}", {"trace_id": request.trace_id})
            check(status == 200 and "status" in body.get("header", {}), (method.name, body))
            check(body["header"]["status"]["code"] != "SERVICE_NOT_READY", body)
    return len(methods) * 2


def check_mixed_concurrency(ports, bounded):
    with grpc.insecure_channel(f"127.0.0.1:{ports[0]}") as channel:
        stub = meta_service_pb2_grpc.MetaServiceStub(channel)

        def invoke(index):
            if index % 2:
                response = stub.GetClusterInfo(
                    meta_service_pb2.GetClusterInfoRequest(trace_id=f"smoke-mixed-{index}"), timeout=5)
                code = meta_service_pb2.ErrorCode.Name(response.header.status.code)
            else:
                status, response = http_json(ports[1], "/api/getClusterInfo", {"trace_id": f"smoke-mixed-{index}"})
                check(status == 200, response)
                code = response["header"]["status"]["code"]
            check(code == "OK" or (bounded and code == "SERVICE_NOT_READY"), code)
            return code

        with ThreadPoolExecutor(max_workers=16) as workers:
            outcomes = list(workers.map(invoke, range(64)))
        # The bounded configuration may reject at admission; no request may hang.
        check("OK" in outcomes, outcomes)
        return outcomes.count("OK")


def check_logging_metrics(workdir, capacity, minimum_completed, timeout):
    deadline = time.monotonic() + timeout
    required = ["queue_length", "queue_length_max", "queue_length_avg", "queue_capacity", "running",
                "wait_time_us_max", "wait_time_us_avg", "execution_time_us_max", "execution_time_us_avg",
                "completed", "rejected", "discarded"]
    while time.monotonic() < deadline:
        # run_case explicitly routes every logger to the console, captured by
        # process.log. Also support the default file logger for manual runs.
        paths = [workdir / "process.log", workdir / "logs" / "metrics.log"]
        content = "\n".join(path.read_text(errors="replace") for path in paths if path.exists())
        samples = []
        for line in content.splitlines():
            if "[Report_Request_Executor_Metrics_By_Print]" in line:
                values = {name: float(value) for name, value in
                          re.findall(r"\[service\.request_([a-z_]+)\]:([0-9.eE+-]+)\|", line)}
                if all(name in values for name in required):
                    samples.append(values)
        if samples and samples[-1]["completed"] >= minimum_completed:
            check(all(sample["queue_capacity"] == capacity for sample in samples), samples)
            check(any(sample["wait_time_us_max"] > 0 for sample in samples), "no queue-wait sample")
            check(any(sample["execution_time_us_max"] > 0 for sample in samples), "no execution sample")
            for sample in samples:
                for prefix in ("wait_time_us", "execution_time_us"):
                    check(0 <= sample[prefix + "_avg"] <= sample[prefix + "_max"] + 1, sample)
                check(0 <= sample["queue_length_avg"] <= sample["queue_length_max"] + 1e-5, sample)
                if capacity:
                    check(sample["queue_length_max"] <= capacity, sample)
            print("Logging metrics:", json.dumps(samples[-1], sort_keys=True), flush=True)
            return
        time.sleep(0.05)
    raise AssertionError("executor Logging metrics were missing or incomplete")


def run_case(binary, name, timeout, mode=None, conflict=None, shared_rpc=False, capacity=None, evidence=None):
    reservations = []
    process = None
    with tempfile.TemporaryDirectory(prefix="kvcm-dispatch-") as temporary:
        workdir = Path(temporary)
        log_path = workdir / "process.log"
        try:
            # Hold all four reservations until their distinct port numbers are known.
            for _ in range(4):
                listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                reservations.append(listener)
                listener.bind(("127.0.0.1", 0))
                listener.listen(1)
            ports = [listener.getsockname()[1] for listener in reservations]
            if shared_rpc:
                ports[2] = ports[0]
            # An empty startup_config would install the built-in /tmp/nfs root.
            storage_root = workdir / "storage"
            storage_root.mkdir()
            startup_path = workdir / "startup.json"
            startup_path.write_text(json.dumps({
                "storage_config": {"type": "file", "global_unique_name": "smoke-file",
                                   "storage_spec": {"root_path": str(storage_root), "key_count_per_file": 8}},
                "instance_group": {
                    "name": "smoke", "storage_candidates": ["smoke-file"], "global_quota_group_name": "smoke",
                    "max_instance_count": 1, "version": 1, "user_data": "",
                    "quota": {"capacity": 1048576, "quota_config": [{"storage_type": "file", "capacity": 1048576}]},
                    "cache_config": {
                        "reclaim_strategy": {"reclaim_policy": 1, "trigger_strategy": {"used_percentage": 0.8},
                                             "delay_before_delete_ms": 1000}, "cache_prefer_strategy": 2,
                        "meta_indexer_config": {"max_key_count": 1000, "mutex_shard_num": 16, "batch_key_size": 16,
                                                "meta_storage_backend_config": {"storage_type": "local", "storage_uri": ""},
                                                "meta_cache_policy_config": {"type": "LRU", "capacity": 1000}}}}}))
            config = {
                "kvcm.registry_storage.uri": "local://",
                "kvcm.coordination.uri": "memory://",
                "kvcm.startup_config": startup_path,
                "kvcm.service.advertised_host": "127.0.0.1",
                "kvcm.service.rpc_port": ports[0],
                "kvcm.service.http_port": ports[1],
                "kvcm.service.admin_rpc_port": ports[2],
                "kvcm.service.admin_http_port": ports[3],
                "kvcm.service.io_thread_num": 1,
                "kvcm.service.worker_thread_num": 2,
                "kvcm.service.enable_debug_service": "false",
                "kvcm.metrics.reporter_type": "logging",
                "kvcm.metrics.report_interval_ms": 100,
            }
            if capacity is not None:
                config["kvcm.service.worker_queue_capacity"] = capacity
            if mode is not None:
                config["kvcm.service.worker_shutdown_mode"] = mode
            config_path = workdir / "server.conf"
            config_path.write_text("".join(f"{key}={value}\n" for key, value in config.items()))
            # ServerConfig allows environment overrides; exclude inherited settings.
            env = {key: value for key, value in os.environ.items()
                   if not key.lower().startswith(("kvcm.", "kvcm_"))}
            env.update(KVCM_LOG_TO_CONSOLE="1", KVCM_LOG_LEVEL="INFO")
            for index, listener in enumerate(reservations):
                if index != conflict:
                    listener.close()
            with log_path.open("wb") as output:
                process = subprocess.Popen([str(binary), "-c", str(config_path)], cwd=workdir,
                                           env=env, stdout=output, stderr=subprocess.STDOUT)
                if conflict is None:
                    check_endpoints(process, ports, timeout)
                    minimum = check_all_meta_methods(ports)
                    minimum += check_mixed_concurrency(ports, bounded=capacity is not None)
                    check_logging_metrics(workdir, 2048 if capacity is None else capacity, minimum, timeout)
                    process.send_signal(signal.SIGTERM)
                    code = process.wait(timeout=timeout)
                    check(code == 0, f"SIGTERM exit code={code}, expected 0")
                else:
                    code = process.wait(timeout=timeout)
                    check(code == 255, f"bind failure exit code={code}, expected main return -1 (255)")
            output = log_path.read_text(errors="replace")
            if conflict is None:
                check(f"received signal {signal.SIGTERM}" in output, "termination signal was not logged")
                check("service request executor stopped:" in output, "missing final executor metrics summary")
            if conflict is not None:
                failure = "Failed to start rpc server" if conflict == 0 else "Failed to start admin rpc server"
                check(failure in output, f"missing expected startup error: {failure}")
                if conflict == 2:
                    check(f"Server listening on 0.0.0.0:{ports[0]} success" in output,
                          "Meta RPC did not start before the independent Admin RPC failure")
            print(f"PASS {name}", flush=True)
        except BaseException:
            if process is not None and process.poll() is None:
                process.kill()  # Only the process created by this case, never a name/port match.
                process.wait(timeout=5)
            if log_path.exists():
                print(f"--- {name}: process log tail ---\n{log_path.read_text(errors='replace')[-12000:]}",
                      flush=True)
            raise
        finally:
            for listener in reservations:
                listener.close()
            if evidence:
                destination = evidence / re.sub(r"[^a-zA-Z0-9_-]+", "_", name)
                shutil.copytree(workdir, destination, dirs_exist_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", required=True, type=Path)
    parser.add_argument("--timeout", type=float, default=20, help="startup/shutdown deadline in seconds")
    parser.add_argument("--evidence-dir", type=Path, help="retain per-case config, stdout and Logging metrics")
    args = parser.parse_args()
    binary = args.binary.resolve()
    check(binary.is_file() and os.access(binary, os.X_OK), f"not an executable: {binary}")
    check(args.timeout > 0, "--timeout must be positive")
    common = {"evidence": args.evidence_dir.resolve() if args.evidence_dir else None}
    run_case(binary, "separate ports / default discard_pending / HTTP+gRPC / SIGTERM", args.timeout, **common)
    run_case(binary, "separate ports / drain / HTTP+gRPC / SIGTERM", args.timeout, mode="drain", **common)
    run_case(binary, "shared RPC port / discard_pending / HTTP+gRPC / SIGTERM", args.timeout, shared_rpc=True, **common)
    run_case(binary, "shared RPC port / drain / HTTP+gRPC / SIGTERM", args.timeout, mode="drain", shared_rpc=True, **common)
    run_case(binary, "configured capacity 7 / mixed load", args.timeout, capacity=7, **common)
    run_case(binary, "Meta RPC bind failure", args.timeout, conflict=0, **common)
    run_case(binary, "Admin RPC bind failure after Meta started", args.timeout, conflict=2, **common)


if __name__ == "__main__":
    main()
