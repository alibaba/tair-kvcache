"""Exercise two proxy processes and two isolated KVCM partitions over loopback."""
import contextlib
from http.client import HTTPConnection, HTTPException
import json
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
import time

import grpc
from kv_cache_manager.protocol.protobuf import meta_service_pb2 as meta
from kv_cache_manager.protocol.protobuf import meta_service_pb2_grpc as rpc


def http(port, method, data):
    connection = HTTPConnection("127.0.0.1", port, timeout=3)
    try:
        connection.request("POST", "/api/" + method, json.dumps(data), {"Content-Type": "application/json"})
        response = connection.getresponse()
        assert response.status == 200, (method, response.status)
        return json.loads(response.read())
    finally:
        connection.close()


def code(response):
    return response["header"]["status"]["code"]


def eventually(check, processes):
    deadline = time.monotonic() + 45
    while time.monotonic() < deadline:
        assert all(p.poll() is None for p in processes), [p.poll() for p in processes]
        try:
            if check():
                return
        except (OSError, HTTPException):
            pass
        time.sleep(0.1)
    raise AssertionError("processes did not reach expected routing state")


def start(binary, directory, options, stack):
    directory.mkdir()
    logfile = stack.enter_context((directory / "output").open("w"))
    args = [binary]
    for key, value in options.items():
        args.extend(["-e", f"{key}={value}"])
    process = subprocess.Popen(args, cwd=directory, stdout=logfile, stderr=subprocess.STDOUT)

    def stop():
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=45)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
                raise
        if process.returncode != 0:
            service_log = directory / "logs/kv_cache_manager.log"
            if service_log.exists():
                print(directory.name, service_log.read_text(errors="replace")[-3000:], file=sys.stderr)
            raise AssertionError((directory, process.returncode))
    stack.callback(stop)
    return process


def run(binary, root, stack):
    with contextlib.ExitStack() as reservations:
        ports = []
        for _ in range(18):
            sock = reservations.enter_context(socket.socket())
            sock.bind(("127.0.0.1", 0))
            ports.append(sock.getsockname()[1])
    proxies = [ports[:3], ports[3:6]]
    backends = [ports[6:10], ports[10:14]]
    common = {"kvcm.service.advertised_host": "127.0.0.1", "kvcm.service.io_thread_num": 1,
              "kvcm.service.worker_thread_num": 2, "kvcm.metrics.reporter_type": "logging"}
    processes = []
    for i, (rpc_port, http_port, control_port) in enumerate(proxies):
        options = dict(common, **{"kvcm.service.worker_type": "proxy", "kvcm.proxy.cluster": "test",
                                  "kvcm.proxy.route_version": 7, "kvcm.service.rpc_port": rpc_port,
                                  "kvcm.service.http_port": http_port,
                                  "kvcm.proxy.control_address": f"127.0.0.1:{control_port}"})
        processes.append(start(binary, root / f"proxy-{i}", options, stack))
    discovery = "static://" + ",".join(f"127.0.0.1:{ports[2]}" for ports in proxies)
    for i, (rpc_port, http_port, admin_rpc, admin_http) in enumerate(backends):
        target = root / f"target-{i}.json"
        target.write_text(json.dumps({"cluster": "test", "partition": i, "version": "7", "bucketOwners": [0, 1]}))
        storage_root = root / f"storage-{i}"
        storage_root.mkdir()
        startup = root / f"startup-{i}.json"
        startup.write_text(json.dumps({
            "storage_config": {"type": "file", "global_unique_name": "local", "storage_spec": {
                "root_path": str(storage_root), "key_count_per_file": 8}},
            "instance_group": {"name": "group", "storage_candidates": ["local"], "global_quota_group_name": "group",
                "max_instance_count": 64, "version": 1,
                "quota": {"capacity": 1048576, "quota_config": [{"storage_type": "file", "capacity": 1048576}]},
                "cache_config": {"reclaim_strategy": {"reclaim_policy": 1, "trigger_strategy": {"used_percentage": 0.8},
                                                      "delay_before_delete_ms": 1000}, "cache_prefer_strategy": 2,
                    "meta_indexer_config": {"max_key_count": 1000, "mutex_shard_num": 16, "batch_key_size": 16,
                        "meta_storage_backend_config": {"storage_type": "local", "storage_uri": ""}}}}}))
        options = dict(common, **{"kvcm.registry_storage.uri": f"local://{root}/registry-{i}.json",
                                  "kvcm.coordination.uri": f"file://{root}/coordination",
                                  "kvcm.startup_config": startup, "kvcm.partition.target_file": target,
                                  "kvcm.proxy.discovery_uri": discovery, "kvcm.service.rpc_port": rpc_port,
                                  "kvcm.service.http_port": http_port, "kvcm.service.admin_rpc_port": admin_rpc,
                                  "kvcm.service.admin_http_port": admin_http})
        processes.append(start(binary, root / f"manager-{i}", options, stack))
        if i == 1:
            standby_config = dict(options)

    owners = {}
    for i in range(32):
        instance = f"instance-{i}"
        register = {"trace_id": "register", "instance_id": instance, "instance_group": "group", "block_size": 16,
                    "model_deployment": {"model_name": "test", "dtype": "FP16", "tp_size": 1, "dp_size": 1, "pp_size": 1},
                    "location_spec_infos": [{"name": "tp0", "size": 32}]}
        eventually(lambda: code(http(proxies[0][1], "registerInstance", register)) == "OK", processes)
        present = [code(http(p[1], "getInstanceInfo", {"instance_id": instance})) == "OK" for p in backends]
        assert sum(present) == 1, present
        owners[present.index(True)] = instance
        if len(owners) == 2:
            break
    assert len(owners) == 2, owners
    key = {"instance_id": owners[0], "trace_id": "cross-proxy"}
    direct = stack.enter_context(grpc.insecure_channel(f"127.0.0.1:{backends[0][0]}"))
    stale = rpc.MetaServiceStub(direct).GetInstanceInfo(
        meta.GetInstanceInfoRequest(**key), timeout=5, metadata=(("kvcm-route-version", "6"), ("kvcm-partition", "0")))
    assert stale.header.status.code == meta.SERVICE_NOT_READY and stale.header.request_id.startswith("cross-proxy_"), stale
    eventually(lambda: code(http(proxies[1][1], "getInstanceInfo", key)) == "OK", processes)
    for rpc_port, http_port, _ in proxies:
        reply = http(http_port, "getClusterInfo", {"trace_id": "discovery"})
        assert reply["header"]["request_id"].startswith("discovery_"), reply
        info = reply["leader_endpoint"]
        assert info["meta_http_port"] == http_port and info["meta_rpc_port"] == rpc_port, info
    started = http(proxies[0][1], "startWriteCache", dict(key, block_keys=[123], write_timeout_seconds=30))
    assert code(started) == "OK" and started["write_session_id"] and started["locations"], started
    processes[0].terminate()
    assert processes[0].wait(timeout=45) == 0
    channel = stack.enter_context(grpc.insecure_channel(f"127.0.0.1:{proxies[1][0]}"))
    stub = rpc.MetaServiceStub(channel)
    finished = stub.FinishWriteCache(meta.FinishWriteCacheRequest(
        **key, write_session_id=started["write_session_id"], success_blocks={"bool_masks": {"values": [True]}}), timeout=5)
    assert finished.header.status.code == meta.OK, finished
    matched = http(proxies[1][1], "getCacheLocationLen", dict(key, query_type="QT_PREFIX_MATCH", block_keys=[123]))
    assert code(matched) == "OK" and int(matched["cache_location_len"]) == 1, matched
    # Partition failure cannot redirect to an unrelated healthy KVCM.
    processes[2].terminate()
    assert processes[2].wait(timeout=45) == 0
    eventually(lambda: code(http(proxies[1][1], "getInstanceInfo", key)) == "SERVICE_NOT_READY",
               [processes[1], processes[3]])
    assert code(http(proxies[1][1], "getInstanceInfo", {"instance_id": owners[1]})) == "OK"

    # A second replica gets the same target, registry and coordination namespace.
    # Wait until it observes the original leader before triggering the election.
    for setting, port in zip(("rpc_port", "http_port", "admin_rpc_port", "admin_http_port"), ports[14:]):
        standby_config["kvcm.service." + setting] = port
    standby = start(binary, root / "standby", standby_config, stack)
    lookup = {"instance_id": owners[1]}
    eventually(lambda: http(ports[15], "getClusterInfo", lookup).get("leader_endpoint", {}).get("meta_rpc_port")
               == backends[1][0], [processes[1], processes[3], standby])
    processes[3].terminate()
    assert processes[3].wait(timeout=45) == 0
    eventually(lambda: code(http(proxies[1][1], "getInstanceInfo", lookup)) == "OK", [processes[1], standby])


if __name__ == "__main__":
    with tempfile.TemporaryDirectory() as tmp, contextlib.ExitStack() as stack:
        root = Path(tmp)
        try:
            run(str(Path(sys.argv[1]).resolve()), root, stack)
        except BaseException:
            for logfile in root.glob("*/output"):
                print(logfile.parent.name, logfile.read_text(errors="replace")[-1500:], file=sys.stderr)
            raise
