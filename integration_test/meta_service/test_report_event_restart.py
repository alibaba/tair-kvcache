#!/usr/bin/env python3

import argparse
import json
import unittest
from pathlib import Path

import test_report_event_snapshot as report_event


HOST = "192.168.1.251:8080"
BLOCK_KEY = 9250


def configure(args):
    report_event.BASE_URL = f"http://{args.host}:{args.http_port}"
    report_event.ADMIN_URL = f"http://{args.host}:{args.admin_http_port}"
    report_event.INSTANCE_ID = args.instance_id
    report_event.META_STORAGE_URI = args.meta_storage_uri
    report_event.ENABLE_LIVENESS_TIMING_TESTS = False

    client = report_event.KVCMClient(
        report_event.BASE_URL, report_event.ADMIN_URL
    )
    fixture = report_event.EventReportFunctionalTest
    fixture.client = client
    fixture.instance_id = args.instance_id
    if args.phase == "prepare":
        fixture._ensure_event_report_storage_registered()
    fixture._ensure_instance_group_created()
    fixture._ensure_instance_registered()
    return client


def prepare(args, client, checks):
    register = client.report_event(
        report_event._make_request(
            args.instance_id,
            HOST,
            [report_event._ev_node_register(["mem", "gpu"])],
            trace_id="restart_prepare_register",
        )
    )
    checks.assertTrue(register.get("snapshot_required"))

    before_uri = report_event._build_event_report_uri(
        HOST, "mem", {"source": "before_restart"}
    )
    snapshot = client.report_event(
        report_event._make_request(
            args.instance_id,
            HOST,
            [report_event._ev_block_snapshot([{
                "block_key": BLOCK_KEY,
                "medium": "mem",
                "specs": report_event._make_single_spec(
                    "linear_0", before_uri
                ),
            }])],
            trace_id="restart_prepare_snapshot",
        )
    )
    version = snapshot["committed_snapshot_version"]
    specs = report_event._wait_for_block_spec_names(
        client,
        args.instance_id,
        BLOCK_KEY,
        {"linear_0"},
        "restart_prepare_query",
    )
    report_event._assert_reporter_scope(
        checks,
        specs[0]["uri"],
        before_uri,
        args.instance_id,
        HOST,
        "mem",
        version,
    )
    Path(args.state_file).write_text(
        json.dumps({"version": version}), encoding="utf-8"
    )
    print(f"RESTART_PREPARE_OK version={version}")


def verify(args, client, checks):
    previous = json.loads(Path(args.state_file).read_text(encoding="utf-8"))
    report_event._wait_for_block_spec_names(
        client,
        args.instance_id,
        BLOCK_KEY,
        set(),
        "restart_verify_old_data_hidden_before_register",
    )

    register = client.report_event(
        report_event._make_request(
            args.instance_id,
            HOST,
            [report_event._ev_node_register(["mem", "gpu"])],
            trace_id="restart_verify_register",
        )
    )
    checks.assertTrue(register.get("snapshot_required"))
    checks.assertEqual(register.get("committed_snapshot_version", ""), "")

    heartbeat = client.report_event(
        report_event._make_request(
            args.instance_id,
            HOST,
            [report_event._ev_heartbeat({"phase": "after_restart"})],
            trace_id="restart_verify_heartbeat_before_snapshot",
        )
    )
    checks.assertTrue(heartbeat.get("snapshot_required"))
    checks.assertEqual(heartbeat.get("committed_snapshot_version", ""), "")
    report_event._wait_for_block_spec_names(
        client,
        args.instance_id,
        BLOCK_KEY,
        set(),
        "restart_verify_heartbeat_does_not_restore_old_data",
    )

    rejected = client.report_event(
        report_event._make_request(
            args.instance_id,
            HOST,
            [report_event._ev_block_add(
                BLOCK_KEY + 1,
                "gpu",
                report_event._make_single_spec(
                    "gpu_0",
                    report_event._build_event_report_uri(HOST, "gpu"),
                ),
            )],
            trace_id="restart_verify_delta_before_snapshot",
        ),
        check_ok=False,
    )
    checks.assertEqual(
        rejected["header"]["status"]["code"], "SNAPSHOT_REQUIRED"
    )

    after_uri = report_event._build_event_report_uri(
        HOST, "mem", {"source": "after_restart"}
    )
    snapshot = client.report_event(
        report_event._make_request(
            args.instance_id,
            HOST,
            [report_event._ev_block_snapshot([{
                "block_key": BLOCK_KEY,
                "medium": "mem",
                "specs": report_event._make_single_spec(
                    "linear_0", after_uri
                ),
            }])],
            trace_id="restart_verify_snapshot",
        )
    )
    version = snapshot["committed_snapshot_version"]
    checks.assertNotEqual(previous["version"], version)
    checks.assertFalse(snapshot.get("snapshot_required"))

    specs = report_event._wait_for_block_spec_names(
        client,
        args.instance_id,
        BLOCK_KEY,
        {"linear_0"},
        "restart_verify_query",
    )
    report_event._assert_reporter_scope(
        checks,
        specs[0]["uri"],
        after_uri,
        args.instance_id,
        HOST,
        "mem",
        version,
    )
    print(
        "RESTART_VERIFY_OK "
        f"old_version={previous['version']} new_version={version}"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("prepare", "verify"), required=True)
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--http-port", type=int, required=True)
    parser.add_argument("--admin-http-port", type=int, required=True)
    parser.add_argument("--instance-id", required=True)
    parser.add_argument("--meta-storage-uri", required=True)
    parser.add_argument("--state-file", required=True)
    args = parser.parse_args()

    client = configure(args)
    checks = unittest.TestCase()
    try:
        if args.phase == "prepare":
            prepare(args, client, checks)
        else:
            verify(args, client, checks)
    finally:
        client.close()


if __name__ == "__main__":
    main()
