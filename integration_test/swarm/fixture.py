"""KVCM Swarm CI fixture: isolated test resources only.

The fixture is the only component allowed to create and destroy deployment
resources (event-report storage, instance group, quota). It also renders the
effective C++ run configuration by injecting the dynamically allocated
endpoints and the actual quota, and it tears the environment down afterwards.

It never proxies an RPC, never enters the request hot path and never keeps
business state on behalf of KVCM.
"""
import json
import os
import time
import uuid

import requests


# The event-report reporter liveness window must be comfortably larger than the
# generator heartbeat interval, otherwise the server would legitimately evict
# reporters mid-run and the availability contract would be meaningless.
HEARTBEAT_TIMEOUT_MS = 60000
CLEANUP_GRACE_MS = 10000
LIVENESS_CHECK_INTERVAL_MS = 500

# The default startup configuration already registers this cold storage.
COLD_STORAGE_NAME = "nfs_01"
COLD_STORAGE_TYPE_VALUE = 4  # DataStorageType::DATA_STORAGE_TYPE_NFS


class AdminClient(object):
    def __init__(self, admin_url, meta_url):
        self.admin_url = admin_url
        self.meta_url = meta_url
        self.session = requests.Session()
        self.session.headers.update({"Content-Type": "application/json", "Accept": "application/json"})

    def _post(self, base, path, payload, accept=("OK",), timeout=10):
        response = self.session.post(base + path, json=payload, timeout=timeout)
        response.raise_for_status()
        body = response.json()
        code = body.get("header", {}).get("status", {}).get("code")
        if accept is not None and code not in accept:
            raise AssertionError("%s failed: %s" % (path, json.dumps(body, ensure_ascii=False)))
        return body

    def add_storage(self, storage):
        return self._post(self.admin_url, "/api/addStorage", {"trace_id": "swarm-fixture", "storage": storage},
                          accept=("OK", "DUPLICATE_ENTITY"))

    def update_storage(self, storage):
        return self._post(self.admin_url, "/api/updateStorage",
                          {"trace_id": "swarm-fixture", "storage": storage, "force_update": True})

    def remove_storage(self, name):
        return self._post(self.admin_url, "/api/removeStorage",
                          {"trace_id": "swarm-fixture", "storage_unique_name": name}, accept=None)

    def create_instance_group(self, group):
        return self._post(self.admin_url, "/api/createInstanceGroup",
                          {"trace_id": "swarm-fixture", "instance_group": group})

    def remove_instance_group(self, name):
        return self._post(self.admin_url, "/api/removeInstanceGroup",
                          {"trace_id": "swarm-fixture", "name": name}, accept=None)

    def list_instance_info(self, group=None):
        payload = {"trace_id": "swarm-fixture"}
        if group:
            payload["instance_group_name"] = group
        return self._post(self.admin_url, "/api/listInstanceInfo", payload, accept=None)

    def remove_instance(self, instance_id):
        return self._post(self.admin_url, "/api/removeInstance",
                          {"trace_id": "swarm-fixture", "instance_id": instance_id}, accept=None)

    def healthy(self):
        response = self.session.get(self.admin_url + "/api/healthy", timeout=5)
        return response.status_code == 200

    def close(self):
        self.session.close()


class SwarmFixture(object):
    """Creates an isolated instance group + event-report storage for one test."""

    def __init__(self, meta_http, meta_grpc, admin_http, workdir,
                 quota_bytes=2 * 1024 * 1024 * 1024, name_hint="swarm"):
        self.meta_http = meta_http
        self.meta_grpc = meta_grpc
        self.admin_http = admin_http
        self.workdir = workdir
        self.quota_bytes = quota_bytes
        unique = uuid.uuid4().hex[:10]
        self.instance_group = "%s-ig-%s" % (name_hint, unique)
        self.event_storage = "%s-er-l2-%s" % (name_hint, unique)
        self.instance_id_prefix = "%s-%s" % (name_hint, unique)
        self.client = AdminClient(admin_http, meta_http)
        self._created_group = False
        self._created_storage = False
        os.makedirs(self.workdir, exist_ok=True)

    # ---- environment ----
    def wait_ready(self, timeout_seconds=30):
        deadline = time.time() + timeout_seconds
        last_error = None
        while time.time() < deadline:
            try:
                if self.client.healthy():
                    return
            except Exception as error:  # noqa: BLE001 - retried below
                last_error = error
            time.sleep(0.2)
        raise AssertionError("KVCM did not become healthy in time: %s" % last_error)

    def setup(self):
        self.wait_ready()
        storage = {
            "global_unique_name": self.event_storage,
            "storage_type": "ST_EVENT_REPORT_L2",
            "event_report": {
                "heartbeat_timeout_ms": HEARTBEAT_TIMEOUT_MS,
                "cleanup_grace_ms": CLEANUP_GRACE_MS,
                "liveness_check_interval_ms": LIVENESS_CHECK_INTERVAL_MS,
            },
            "check_storage_available_when_open": False,
        }
        self.client.add_storage(storage)
        self.client.update_storage(storage)
        self._created_storage = True
        self.client.create_instance_group({
            "name": self.instance_group,
            "storage_candidates": [COLD_STORAGE_NAME],
            "global_quota_group_name": "default_quota_group",
            "max_instance_count": 100,
            "quota": {
                "capacity": self.quota_bytes,
                "quota_config": [{"storage_type": COLD_STORAGE_TYPE_VALUE, "capacity": self.quota_bytes}],
            },
            "cache_config": {
                "reclaim_strategy": {
                    "storage_unique_name": COLD_STORAGE_NAME,
                    "reclaim_policy": 1,
                    "trigger_strategy": {"used_percentage": 0.9},
                    "trigger_period_seconds": 3600,
                    "reclaim_step_percentage": 10,
                },
                "data_storage_strategy": 2,
                "meta_indexer_config": {
                    "max_key_count": 4000000,
                    "mutex_shard_num": 16,
                    "batch_key_size": 16,
                    "meta_storage_backend_config": {"storage_type": "local", "storage_uri": ""},
                    "meta_cache_policy_config": {"type": "LRU", "capacity": 200000},
                },
            },
            "event_report_storage_candidates": [self.event_storage],
            "version": 1,
        })
        self._created_group = True
        return self

    def teardown(self):
        """Removes everything the fixture created, including run residue."""
        notes = []
        try:
            listed = self.client.list_instance_info(self.instance_group)
            for instance in listed.get("instance_info", []) or []:
                instance_id = instance.get("instance_id")
                if instance_id:
                    self.client.remove_instance(instance_id)
                    notes.append("removed instance %s" % instance_id)
        except Exception as error:  # noqa: BLE001 - teardown is best effort
            notes.append("instance cleanup failed: %s" % error)
        if self._created_group:
            try:
                self.client.remove_instance_group(self.instance_group)
                notes.append("removed instance group %s" % self.instance_group)
            except Exception as error:  # noqa: BLE001
                notes.append("instance group cleanup failed: %s" % error)
        if self._created_storage:
            try:
                self.client.remove_storage(self.event_storage)
                notes.append("removed storage %s" % self.event_storage)
            except Exception as error:  # noqa: BLE001
                notes.append("storage cleanup failed: %s" % error)
        self.client.close()
        return notes

    # ---- effective run configuration ----
    def render_config(self, scenario_path, out_name="effective_run_config.json", overrides=None,
                      transport_override=None):
        with open(scenario_path) as handle:
            config = json.load(handle)
        if transport_override is not None:
            if transport_override not in ("http", "grpc"):
                raise ValueError("unsupported transport override: %s" % transport_override)
            for behavior in config.get("behaviors", []):
                behavior["transport"] = transport_override
        config["target"]["endpoints"] = {
            "meta_http": self.meta_http,
            "meta_grpc": self.meta_grpc,
            "admin_http": self.admin_http,
        }
        config["target"]["instance_groups"] = {self.instance_group: {"quota_bytes": self.quota_bytes}}
        for index, behavior in enumerate(config.get("behaviors", [])):
            if behavior.get("type") != "v6d_deployment":
                continue
            behavior["config"]["instance_group"] = self.instance_group
            # A unique instance_id per run keeps replica-threshold behavior
            # reproducible: the generator is deterministic, so reusing an
            # instance would inherit the previous run's locations.
            behavior["config"]["instance_id"] = "%s-d%d" % (self.instance_id_prefix, index)
        if overrides:
            _deep_update(config, overrides)
        config["evidence"] = {
            "output_json": os.path.join(self.workdir, "report.json"),
            "violations_jsonl": os.path.join(self.workdir, "violations.jsonl"),
            "markdown_summary": os.path.join(self.workdir, "summary.md"),
        }
        out_path = os.path.join(self.workdir, out_name)
        with open(out_path, "w") as handle:
            json.dump(config, handle, indent=2, sort_keys=True)
        return out_path


def _deep_update(target, updates):
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(target.get(key), dict):
            _deep_update(target[key], value)
        else:
            target[key] = value
    return target
