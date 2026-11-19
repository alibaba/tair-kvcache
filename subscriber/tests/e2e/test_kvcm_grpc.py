"""End-to-end KVCM gRPC transport test against a real KVCM deployment.

The suite is explicit opt-in: it skips unless ``KVCM_REAL_GRPC_TARGET`` names
a reachable KVCM MetaService endpoint. ``KVCM_REAL_ADMIN_HTTP_URL`` is
optional; when set, the test creates an isolated event-report instance group
through the real admin API before issuing gRPC requests.
"""

from __future__ import annotations

import os
import uuid

import httpx
import pytest

from subscriber.kvcm.errors import KvcmResponseRejectedError
from subscriber.kvcm.grpc_manager_client import GrpcKvCacheManagerClient

_KVCM_REAL_GRPC_TARGET = os.environ.get("KVCM_REAL_GRPC_TARGET", "").strip()
_KVCM_REAL_ADMIN_HTTP_URL = os.environ.get("KVCM_REAL_ADMIN_HTTP_URL", "").strip()

pytestmark = [
    pytest.mark.e2e,
    pytest.mark.skipif(
        not _KVCM_REAL_GRPC_TARGET,
        reason="e2e opt-in: set KVCM_REAL_GRPC_TARGET to a real KVCM MetaService",
    ),
]


async def _setup_real_event_report_instance_group(
    admin_url: str,
    *,
    instance_group: str,
    storage_name: str,
) -> None:
    async with httpx.AsyncClient(
        base_url=admin_url.rstrip("/"),
        headers={"Content-Type": "application/json", "Accept": "application/json"},
        timeout=2.0,
    ) as http_client:
        storage = {
            "global_unique_name": storage_name,
            "storage_type": "ST_EVENT_REPORT_L1P5",
            "event_report": {
                "heartbeat_timeout_ms": 1000,
                "cleanup_grace_ms": 2000,
                "liveness_check_interval_ms": 200,
            },
            "check_storage_available_when_open": False,
        }
        add_response = await http_client.post(
            "/api/addStorage",
            json={"trace_id": "real-grpc-add-storage", "storage": storage},
        )
        add_response.raise_for_status()
        add_body = add_response.json()
        add_code = add_body.get("header", {}).get("status", {}).get("code")
        assert add_code in {"OK", "DUPLICATE_ENTITY"}, add_body

        group_response = await http_client.post(
            "/api/createInstanceGroup",
            json={
                "trace_id": "real-grpc-create-instance-group",
                "instance_group": {
                    "name": instance_group,
                    "storage_candidates": ["nfs_01"],
                    "global_quota_group_name": "default_quota_group",
                    "max_instance_count": 100,
                    "quota": {
                        "capacity": 10737418240,
                        "quota_config": [{"storage_type": 4, "capacity": 10737418240}],
                    },
                    "cache_config": {
                        "reclaim_strategy": {
                            "storage_unique_name": "nfs_01",
                            "reclaim_policy": 1,
                            "trigger_strategy": {
                                "used_size": 1073741824,
                                "used_percentage": 0.8,
                            },
                            "trigger_period_seconds": 60,
                            "reclaim_step_size": 1073741824,
                            "reclaim_step_percentage": 10,
                        },
                        "data_storage_strategy": 2,
                        "meta_indexer_config": {
                            "max_key_count": 1000000,
                            "mutex_shard_num": 16,
                            "batch_key_size": 16,
                            "meta_storage_backend_config": {
                                "storage_type": "local",
                                "storage_uri": "",
                            },
                            "meta_cache_policy_config": {
                                "type": "LRU",
                                "capacity": 10000,
                            },
                        },
                    },
                    "event_report_storage_candidates": [storage_name],
                    "version": 1,
                },
            },
        )
        group_response.raise_for_status()
        group_body = group_response.json()
        group_code = group_body.get("header", {}).get("status", {}).get("code")
        assert group_code in {"OK", "DUPLICATE_ENTITY"}, group_body


async def test_real_kvcm_grpc_register_and_report_event() -> None:
    """Register an isolated instance and report node/heartbeat events over gRPC.

    Setup in the KVCM repo:

    ```bash
    bazel build //kv_cache_manager:kv_cache_manager_bin
    bazel-bin/kv_cache_manager/kv_cache_manager_bin \\
      --env kvcm.service.rpc_port=56010 \\
      --env kvcm.service.http_port=56020 \\
      --env kvcm.service.admin_rpc_port=56031 \\
      --env kvcm.service.admin_http_port=56040 \\
      --env kvcm.service.enable_debug_service=false
    ```

    Run in this repo:

    ```bash
    KVCM_REAL_GRPC_TARGET=127.0.0.1:56010 \\
    KVCM_REAL_ADMIN_HTTP_URL=http://127.0.0.1:56040 \\
    uv run pytest tests/e2e/test_kvcm_grpc.py -vv
    ```
    """
    suffix = uuid.uuid4().hex
    instance_id = f"subscriber-real-grpc-{suffix}"
    instance_group = "default"
    if _KVCM_REAL_ADMIN_HTTP_URL:
        instance_group = f"subscriber-real-grpc-group-{suffix}"
        await _setup_real_event_report_instance_group(
            _KVCM_REAL_ADMIN_HTTP_URL,
            instance_group=instance_group,
            storage_name=f"subscriber-real-grpc-l1p5-{suffix}",
        )
    client = GrpcKvCacheManagerClient(
        _KVCM_REAL_GRPC_TARGET,
        auto_discover_leader=True,
        request_timeout_seconds=2.0,
    )

    try:
        await client.start()
        register_response = await client.register_instance(
            {
                "trace_id": "real-grpc-register",
                "instance_group": instance_group,
                "instance_id": instance_id,
                "block_size": 16,
                "location_spec_infos": [{"name": "vllm_16", "size": 16}],
                "location_spec_groups": [
                    {"name": "default", "spec_names": ["vllm_16"]}
                ],
                "model_deployment": {
                    "model_name": "subscriber-real-grpc",
                    "dtype": "bytes",
                    "tp_size": 1,
                    "dp_size": 1,
                    "pp_size": 1,
                },
                "default_query_type": "QT_PREFIX_MATCH_WITH_MAMBA",
            }
        )
        report_request = {
            "trace_id": "real-grpc-report",
            "instance_id": instance_id,
            "host_ip_port": "127.0.0.1:8080",
            "storage_type": "ST_EVENT_REPORT_L1P5",
            "events": [
                {
                    "event_type": "EVENT_NODE_REGISTER",
                    "node_register": {"mediums": ["hbm"]},
                },
                {"event_type": "EVENT_HEARTBEAT"},
            ],
        }
        try:
            report_response = await client.report_event(report_request)
        except KvcmResponseRejectedError as exc:
            if "EventReportBackend not found" in str(exc):
                pytest.skip(
                    "real KVCM server has no event_report_l1p5 backend configured"
                )
            raise
    finally:
        await client.close()

    assert register_response["header"]["status"]["code"] == "OK"
    assert report_response["header"]["status"]["code"] == "OK"
