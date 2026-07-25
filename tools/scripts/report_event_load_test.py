import unittest
from types import SimpleNamespace
from unittest.mock import patch

import report_event_load


def _missing_backend(storage_type):
    return report_event_load.KVCMRequestError(
        {
            "header": {
                "status": {
                    "code": "INSTANCE_NOT_EXIST",
                    "message": (
                        "EventReportBackend not found for instance: test, "
                        f"type: {storage_type}"
                    ),
                }
            }
        }
    )


def _args(storage_type="auto"):
    return SimpleNamespace(
        base_url="http://127.0.0.1:6382",
        timeout=1.0,
        instance_group="test_group",
        instance_id="test_instance",
        storage_type=storage_type,
        medium="mem",
        spec_name="tp0",
        spec_size=1024,
        model_name="test",
        dtype="FP8",
        use_mla=False,
        tp_size=1,
        dp_size=1,
        pp_size=1,
        extra="",
        user_data="",
        block_size=128,
        verify_keys_per_event=4,
        key_base=1000,
        allow_reuse_instance=False,
        add_qps=1,
        delete_qps=1,
        get_qps=1,
        snapshot_interval_sec=35.0,
        snapshot_drop_ratio=0.0,
        heartbeat_interval_sec=10.0,
        duration_sec=60,
        workers=4,
        queue_size=100,
        host_count=1,
        key_space=1000,
        query_blocks=8,
        add_batch_size=2,
        delete_batch_size=2,
    )


class FakeBootstrapClient:
    instances = []

    def __init__(self, _base_url, _timeout):
        self.calls = []
        self.__class__.instances.append(self)

    def post(self, api, payload):
        self.calls.append((api, payload))
        if api == "/api/registerInstance":
            return {"header": {"status": {"code": "OK"}}}
        if api == "/api/getHostCacheState":
            return {
                "header": {"status": {"code": "OK"}},
                "hosts": [],
            }

        event_type = payload["events"][0]["event_type"]
        if event_type == "EVENT_NODE_REGISTER":
            if payload["storage_type"] == "ST_EVENT_REPORT_L2":
                raise _missing_backend("event_report_l2")
            return {
                "header": {"status": {"code": "OK"}},
                "snapshot_required": True,
            }
        if event_type == "EVENT_BLOCK_SNAPSHOT":
            return {
                "header": {"status": {"code": "OK"}},
                "committed_snapshot_version": "a" * 32,
                "snapshot_required": False,
            }
        raise AssertionError(f"unexpected payload: {payload}")

    def close(self):
        pass


class ReportEventLoadTest(unittest.TestCase):
    def setUp(self):
        FakeBootstrapClient.instances.clear()

    def test_bootstrap_registers_instance_then_auto_selects_l1p5(self):
        args = _args()
        shadow = report_event_load.ShadowState(["10.0.0.1:8080"])
        with patch.object(
            report_event_load,
            "KVCMHttpClient",
            FakeBootstrapClient,
        ):
            report_event_load.bootstrap(
                args, ["10.0.0.1:8080"], shadow
            )

        calls = FakeBootstrapClient.instances[0].calls
        self.assertEqual(calls[0][0], "/api/registerInstance")
        self.assertEqual(
            calls[0][1]["instance_id"], "test_instance"
        )
        node_register_types = [
            payload["storage_type"]
            for api, payload in calls
            if api == "/api/reportEvent"
            and payload["events"][0]["event_type"]
            == "EVENT_NODE_REGISTER"
        ]
        self.assertEqual(
            node_register_types,
            ["ST_EVENT_REPORT_L2", "ST_EVENT_REPORT_L1P5"],
        )
        self.assertEqual(
            args.storage_type, "ST_EVENT_REPORT_L1P5"
        )
        self.assertEqual(
            shadow.host(
                "10.0.0.1:8080"
            ).committed_snapshot_version,
            "a" * 32,
        )

    def test_bootstrap_rejects_reusing_committed_reporter(self):
        class ExistingBootstrapClient(FakeBootstrapClient):
            def post(self, api, payload):
                response = super().post(api, payload)
                if (
                    api == "/api/reportEvent"
                    and payload["events"][0]["event_type"]
                    == "EVENT_NODE_REGISTER"
                ):
                    response["snapshot_required"] = False
                return response

        args = _args()
        shadow = report_event_load.ShadowState(["10.0.0.1:8080"])
        with patch.object(
            report_event_load,
            "KVCMHttpClient",
            ExistingBootstrapClient,
        ):
            with self.assertRaisesRegex(
                RuntimeError,
                "already has a committed snapshot.*fresh --instance-id",
            ):
                report_event_load.bootstrap(
                    args, ["10.0.0.1:8080"], shadow
                )

    def test_auto_reports_group_without_event_backend(self):
        class MissingClient:
            def post(self, _api, payload):
                raise _missing_backend(payload["storage_type"])

        args = _args()
        with self.assertRaisesRegex(
            RuntimeError,
            "exposes neither an L2 nor an L1P5 EventReport backend",
        ):
            report_event_load.register_first_reporter(
                MissingClient(), args, "10.0.0.1:8080"
            )

    def test_explicit_backend_error_explains_group_configuration(self):
        class MissingClient:
            def post(self, _api, payload):
                raise _missing_backend(payload["storage_type"])

        args = _args("ST_EVENT_REPORT_L2")
        with self.assertRaisesRegex(
            RuntimeError,
            "registerInstance succeeded.*event_report_storage_candidates",
        ):
            report_event_load.register_first_reporter(
                MissingClient(), args, "10.0.0.1:8080"
            )

    def test_retry_after_ms_accepts_json_string(self):
        error = report_event_load.KVCMRequestError(
            {
                "header": {
                    "status": {
                        "code": "SNAPSHOT_RATE_LIMITED",
                        "message": "retry later",
                    }
                },
                "retry_after_ms": "28489",
            }
        )
        self.assertEqual(error.retry_after_ms, 28489)

    def test_validate_rejects_run_with_no_scheduled_load(self):
        args = _args()
        args.add_qps = 0
        args.delete_qps = 0
        args.get_qps = 0
        args.duration_sec = 15
        args.snapshot_interval_sec = 35
        with self.assertRaisesRegex(
            ValueError, "duration 内不会触发周期 snapshot"
        ):
            report_event_load.validate_args(args)


if __name__ == "__main__":
    unittest.main()
