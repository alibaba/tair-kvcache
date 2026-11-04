#!/usr/bin/env python3
"""Standalone process smoke: python3 service_dispatch_smoke.py --binary /path/to/kvcm.

Uses only the standard library, a temporary cwd, memory coordination and a local
registry without persistence. The only storage root is inside that temporary cwd.
"""

import argparse
import http.client
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import tempfile
import time


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


def run_case(binary, name, timeout, mode=None, conflict=None):
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
                "kvcm.service.worker_queue_capacity": 0,
                "kvcm.service.enable_debug_service": "false",
            }
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
                    process.send_signal(signal.SIGTERM)
                    code = process.wait(timeout=timeout)
                    check(code == 0, f"SIGTERM exit code={code}, expected 0")
                else:
                    code = process.wait(timeout=timeout)
                    check(code == 255, f"bind failure exit code={code}, expected main return -1 (255)")
            output = log_path.read_text(errors="replace")
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", required=True, type=Path)
    parser.add_argument("--timeout", type=float, default=20, help="startup/shutdown deadline in seconds")
    args = parser.parse_args()
    binary = args.binary.resolve()
    check(binary.is_file() and os.access(binary, os.X_OK), f"not an executable: {binary}")
    check(args.timeout > 0, "--timeout must be positive")
    run_case(binary, "default discard_pending / HTTP / SIGTERM", args.timeout)
    run_case(binary, "drain / HTTP / SIGTERM", args.timeout, mode="drain")
    run_case(binary, "Meta RPC bind failure", args.timeout, conflict=0)
    run_case(binary, "Admin RPC bind failure after Meta started", args.timeout, conflict=2)


if __name__ == "__main__":
    main()
