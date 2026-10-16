import argparse
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

# Bazel keeps sources under package/kvcm_ops while the wheel exposes kvcm_ops
# as a top-level namespace package.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from kvcm_ops.kvcm.storage.util import (
    add_event_report_sub_parser,
    add_pace_sub_parser,
    add_pace_ssd_sub_parser,
    gen_event_report_config_data,
    gen_pace_config_data,
    get_pace_storage_type,
)
from kvcm_ops.kvcm.storage import add_storage, update_storage
from kvcm_ops.kvcm.storage.update_storage import create_update_storage_data
from kvcm_ops.kvcm.instance_group.util import CacheConfig


class EventReportStorageArgsTest(unittest.TestCase):
    def _parse_args(self, *args):
        parser = argparse.ArgumentParser()
        subparsers = parser.add_subparsers(dest="storage_type", required=True)
        add_event_report_sub_parser(
            subparsers,
            "event_report_l1p5",
            "L1.5 event report storage options",
            "ST_EVENT_REPORT_L1P5",
        )
        return parser.parse_args(["event_report_l1p5", *args])

    def test_snapshot_min_interval_is_added_to_storage_spec(self):
        args = self._parse_args(
            "--heartbeat_timeout_ms",
            "30000",
            "--cleanup_grace_ms",
            "300000",
            "--liveness_check_interval_ms",
            "5000",
            "--snapshot_min_interval_ms",
            "1000",
        )

        self.assertEqual(
            {
                "heartbeat_timeout_ms": 30000,
                "cleanup_grace_ms": 300000,
                "liveness_check_interval_ms": 5000,
                "snapshot_min_interval_ms": 1000,
            },
            gen_event_report_config_data(args),
        )

    def test_snapshot_min_interval_is_optional(self):
        args = self._parse_args()

        self.assertNotIn(
            "snapshot_min_interval_ms",
            gen_event_report_config_data(args),
        )

    def test_snapshot_min_interval_must_be_positive(self):
        for invalid_value in ("0", "-1"):
            with self.subTest(invalid_value=invalid_value):
                with self.assertRaises(SystemExit):
                    self._parse_args("--snapshot_min_interval_ms", invalid_value)


class PaceStorageArgsTest(unittest.TestCase):
    def _parse_args(self, *args):
        parser = argparse.ArgumentParser()
        subparsers = parser.add_subparsers(dest="storage_type", required=True)
        add_pace_sub_parser(subparsers)
        return parser.parse_args(["pace", "--domain", "pace.meta", "--timeout", "5000", *args])

    def _parse_ssd_args(self, *args):
        parser = argparse.ArgumentParser()
        subparsers = parser.add_subparsers(dest="storage_type", required=True)
        add_pace_ssd_sub_parser(subparsers)
        return parser.parse_args(["pace_ssd", "--domain", "pace.meta", "--timeout", "5000", *args])

    def test_health_timeout_is_optional(self):
        spec = gen_pace_config_data(self._parse_args())
        self.assertNotIn("health_check_timeout", spec)

    def test_health_timeout_is_serialized(self):
        spec = gen_pace_config_data(self._parse_args(
            "--health_check_timeout", "2"))
        self.assertEqual(2, spec["health_check_timeout"])

    def test_health_timeout_must_be_positive(self):
        for value in ("0", "-1"):
            with self.subTest(value=value):
                with self.assertRaises(SystemExit):
                    self._parse_args("--health_check_timeout", value)

    def test_ssd_subcommand_sets_distinct_storage_type(self):
        args = self._parse_ssd_args()
        self.assertEqual(5, gen_pace_config_data(args)["media_type"])
        self.assertEqual("ST_TAIRMEMPOOL_SSD", get_pace_storage_type(args.media_type))

    def test_legacy_media_keeps_legacy_storage_type(self):
        args = self._parse_args()
        self.assertEqual(0, gen_pace_config_data(args)["media_type"])
        self.assertEqual("ST_TAIRMEMPOOL", get_pace_storage_type(args.media_type))

    def test_dram_media_keeps_legacy_storage_type(self):
        args = self._parse_args("--media_type", "2")
        self.assertEqual("ST_TAIRMEMPOOL", get_pace_storage_type(args.media_type))

    def test_ssd_media_is_rejected_by_legacy_pace_subcommand(self):
        with self.assertRaises(SystemExit):
            self._parse_args("--media_type", "5")

    def test_invalid_media_type_is_rejected(self):
        with self.assertRaises(SystemExit):
            self._parse_args("--media_type", "7")

    def test_gc_delete_policy_is_optional_and_accepts_explicit_booleans(self):
        for parse in (self._parse_args, self._parse_ssd_args):
            self.assertNotIn("skip_confirmed_missing_backend_delete", gen_pace_config_data(parse()))
            for value in ("true", "false"):
                with self.subTest(parser=parse.__name__, value=value):
                    args = parse("--skip_confirmed_missing_backend_delete", value)
                    self.assertEqual(
                        value == "true", gen_pace_config_data(args)["skip_confirmed_missing_backend_delete"])
            with self.assertRaises(SystemExit):
                parse("--skip_confirmed_missing_backend_delete", "invalid")

    def test_update_can_distinguish_omitted_media_type(self):
        parser = argparse.ArgumentParser()
        subparsers = parser.add_subparsers(dest="storage_type", required=True)
        add_pace_sub_parser(subparsers, media_type_default=None)

        args = parser.parse_args(["pace", "--domain", "pace.meta", "--timeout", "5000"])

        self.assertIsNone(args.media_type)

    def test_update_payload_always_carries_explicit_pace_storage_type(self):
        for media_type, expected_type in ((0, "ST_TAIRMEMPOOL"), (2, "ST_TAIRMEMPOOL"),
                                          (5, "ST_TAIRMEMPOOL_SSD")):
            with self.subTest(media_type=media_type):
                args = argparse.Namespace(unique_name="pace_storage", media_type=media_type, trace_id="test")
                data = create_update_storage_data(args, "tair_mem_pool", {"media_type": media_type})
                self.assertEqual(expected_type, data["storage"]["storage_type"])


class PaceStorageUpdateTest(unittest.TestCase):
    def _args(self, media_type=None, storage_type="pace"):
        return SimpleNamespace(
            domain="pace.meta",
            host="http://manager",
            media_type=media_type,
            service_discovery_url=None,
            timeout=5000,
            health_check_timeout=None,
            trace_id="trace",
            unique_name="pace_storage",
            verbose=False,
            storage_type=storage_type,
        )

    @staticmethod
    def _list_response(storage_type, media_type):
        return {
            "header": {"status": {"code": "OK"}},
            "storage": [
                {
                    "global_unique_name": "pace_storage",
                    "storage_type": storage_type,
                    "tair_mem_pool": {
                        "domain": "pace.meta",
                        "timeout": 5000,
                        "media_type": media_type,
                    },
                }
            ],
        }

    @patch.object(update_storage, "http_post_and_print")
    @patch.object(update_storage, "http_post")
    def test_update_preserves_and_overrides_health_timeout(self, mock_http_post, mock_post_and_print):
        response = self._list_response("ST_TAIRMEMPOOL", 2)
        response["storage"][0]["tair_mem_pool"].update(
            health_check_timeout=2, skip_confirmed_missing_backend_delete=True)
        mock_http_post.return_value = response
        args = self._args()
        update_storage.handle_pace(args)
        spec = mock_post_and_print.call_args.args[1]["storage"]["tair_mem_pool"]
        self.assertEqual(2, spec["health_check_timeout"])
        self.assertTrue(spec["skip_confirmed_missing_backend_delete"])
        args.health_check_timeout = 1
        update_storage.handle_pace(args)
        spec = mock_post_and_print.call_args.args[1]["storage"]["tair_mem_pool"]
        self.assertEqual(1, spec["health_check_timeout"])
        self.assertTrue(spec["skip_confirmed_missing_backend_delete"])

    @patch.object(update_storage, "http_post_and_print")
    @patch.object(update_storage, "http_post")
    def test_ssd_subcommand_preserves_ssd_type(self, mock_http_post, mock_post_and_print):
        mock_http_post.return_value = self._list_response("ST_TAIRMEMPOOL_SSD", 5)

        update_storage.handle_pace(self._args(media_type=5, storage_type="pace_ssd"))

        request = mock_post_and_print.call_args.args[1]
        self.assertEqual("ST_TAIRMEMPOOL_SSD", request["storage"]["storage_type"])
        self.assertEqual(5, request["storage"]["tair_mem_pool"]["media_type"])

    @patch.object(update_storage, "http_post_and_print")
    @patch.object(update_storage, "http_post")
    def test_omitted_media_type_preserves_legacy_media_5(self, mock_http_post, mock_post_and_print):
        for storage_type in ("ST_UNSPECIFIED", "ST_TAIRMEMPOOL"):
            with self.subTest(storage_type=storage_type):
                mock_http_post.return_value = self._list_response(storage_type, 5)

                update_storage.handle_pace(self._args())

                request = mock_post_and_print.call_args.args[1]
                self.assertEqual("ST_TAIRMEMPOOL", request["storage"]["storage_type"])
                self.assertEqual(5, request["storage"]["tair_mem_pool"]["media_type"])

    @patch.object(update_storage, "http_post_and_print")
    @patch.object(update_storage, "http_post")
    def test_explicit_legacy_media_type_preserves_legacy_type(self, mock_http_post, mock_post_and_print):
        mock_http_post.return_value = self._list_response("ST_TAIRMEMPOOL", 2)

        update_storage.handle_pace(self._args(media_type=2))

        request = mock_post_and_print.call_args.args[1]
        self.assertEqual("ST_TAIRMEMPOOL", request["storage"]["storage_type"])
        self.assertEqual(2, request["storage"]["tair_mem_pool"]["media_type"])

    @patch.object(update_storage, "http_post_and_print")
    @patch.object(update_storage, "http_post")
    def test_pace_subcommand_rejects_ssd_type(self, mock_http_post, mock_post_and_print):
        mock_http_post.return_value = self._list_response("ST_TAIRMEMPOOL_SSD", 5)

        with self.assertRaisesRegex(RuntimeError, "use the 'pace_ssd' subcommand"):
            update_storage.handle_pace(self._args())

        mock_post_and_print.assert_not_called()

    @patch.object(update_storage, "http_post_and_print")
    @patch.object(update_storage, "http_post")
    def test_pace_ssd_subcommand_rejects_legacy_type(self, mock_http_post, mock_post_and_print):
        mock_http_post.return_value = self._list_response("ST_TAIRMEMPOOL", 5)

        with self.assertRaisesRegex(RuntimeError, "use the 'pace' subcommand"):
            update_storage.handle_pace(self._args(media_type=5, storage_type="pace_ssd"))

        mock_post_and_print.assert_not_called()

    @patch.object(update_storage, "http_post_and_print")
    @patch.object(update_storage, "http_post")
    def test_omitted_media_type_rejects_missing_storage(self, mock_http_post, mock_post_and_print):
        mock_http_post.return_value = {"header": {"status": {"code": "OK"}}, "storage": []}

        with self.assertRaisesRegex(RuntimeError, "does not exist"):
            update_storage.handle_pace(self._args())

        mock_post_and_print.assert_not_called()

    @patch.object(update_storage, "http_post_and_print")
    @patch.object(update_storage, "http_post")
    def test_gc_delete_policy_is_preserved_or_explicitly_updated(self, mock_http_post, mock_post_and_print):
        field = "skip_confirmed_missing_backend_delete"
        for storage_type, media_type, subcommand in (
                ("ST_TAIRMEMPOOL", 2, "pace"), ("ST_TAIRMEMPOOL_SSD", 5, "pace_ssd")):
            for previous in (None, False, True):
                for requested in (None, False, True):
                    with self.subTest(storage_type=storage_type, previous=previous, requested=requested):
                        response = self._list_response(storage_type, media_type)
                        if previous is not None:
                            response["storage"][0]["tair_mem_pool"][field] = previous
                        mock_http_post.return_value = response
                        args = self._args(storage_type=subcommand)
                        setattr(args, field, requested)
                        update_storage.handle_pace(args)
                        expected = requested if requested is not None else bool(previous)
                        payload = mock_post_and_print.call_args.args[1]
                        self.assertEqual(expected, payload["storage"]["tair_mem_pool"][field])
                        self.assertEqual(media_type, payload["storage"]["tair_mem_pool"]["media_type"])
                        self.assertIs(requested, getattr(args, field))


class PaceServiceDiscoveryCliTest(unittest.TestCase):
    @patch.object(add_storage, "http_post_and_print")
    def test_add_defaults_to_direct_domain_and_accepts_explicit_url(self, mock_post):
        for subcommand in ("pace", "pace_ssd"):
            for requested in (None, "", "vipserver://new.meta"):
                with self.subTest(subcommand=subcommand, requested=requested):
                    argv = ["add_storage", "-u", "pace_storage", subcommand,
                            "--domain", "pace.meta", "--timeout", "30"]
                    if requested is not None:
                        argv.extend(["--service_discovery_url", requested])
                    with patch.object(sys, "argv", argv):
                        add_storage.main()
                    spec = mock_post.call_args.args[1]["storage"]["tair_mem_pool"]
                    self.assertEqual(requested or "", spec.get("service_discovery_url", ""))
                    self.assertEqual("pace.meta", spec["domain"])

    @patch.object(update_storage, "http_post_and_print")
    @patch.object(update_storage, "http_post")
    def test_update_preserves_omitted_url_and_allows_replace_or_clear(self, mock_http_post, mock_post):
        for storage_type, media_type, subcommand in (
                ("ST_TAIRMEMPOOL", 0, "pace"),
                ("ST_TAIRMEMPOOL", 2, "pace"),
                ("ST_TAIRMEMPOOL", 5, "pace"),
                ("ST_TAIRMEMPOOL_SSD", 5, "pace_ssd")):
            for previous in (None, "", "vipserver://old.meta"):
                for requested in (None, "", "vipserver://new.meta"):
                    with self.subTest(storage_type=storage_type, media_type=media_type,
                                      previous=previous, requested=requested):
                        response = PaceStorageUpdateTest._list_response(storage_type, media_type)
                        existing_spec = response["storage"][0]["tair_mem_pool"]
                        existing_spec.update(health_check_timeout=2, skip_confirmed_missing_backend_delete=True)
                        if previous is not None:
                            existing_spec["service_discovery_url"] = previous
                        mock_http_post.return_value = response
                        argv = ["update_storage", "-u", "pace_storage", subcommand,
                                "--domain", "pace.meta", "--timeout", "30"]
                        if requested is not None:
                            argv.extend(["--service_discovery_url", requested])
                        with patch.object(sys, "argv", argv):
                            update_storage.main()
                        storage = mock_post.call_args.args[1]["storage"]
                        spec = storage["tair_mem_pool"]
                        expected = requested if requested is not None else previous or ""
                        self.assertEqual(expected, spec.get("service_discovery_url", ""))
                        self.assertEqual(storage_type, storage["storage_type"])
                        self.assertEqual(media_type, spec["media_type"])
                        self.assertEqual(30, spec["timeout"])
                        self.assertEqual(2, spec["health_check_timeout"])
                        self.assertTrue(spec["skip_confirmed_missing_backend_delete"])
                        self.assertEqual(previous, existing_spec.get("service_discovery_url"))


class TairMempoolSsdPreferenceTest(unittest.TestCase):
    def test_ssd_preference_is_accepted(self):
        config = CacheConfig(data_storage_strategy="CPS_ALWAYS_TAIR_MEMPOOL_SSD")
        self.assertEqual("CPS_ALWAYS_TAIR_MEMPOOL_SSD", config.to_json_data()["data_storage_strategy"])


if __name__ == "__main__":
    unittest.main()
