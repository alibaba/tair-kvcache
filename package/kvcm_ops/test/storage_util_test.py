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
    add_pace_sub_parser,
    add_pace_ssd_sub_parser,
    gen_pace_config_data,
    get_pace_storage_type,
)
from kvcm_ops.kvcm.storage import update_storage
from kvcm_ops.kvcm.storage.update_storage import create_update_storage_data
from kvcm_ops.kvcm.instance_group.util import CacheConfig


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
            service_discovery_url="",
            timeout=5000,
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


class TairMempoolSsdPreferenceTest(unittest.TestCase):
    def test_ssd_preference_is_accepted(self):
        config = CacheConfig(data_storage_strategy="CPS_ALWAYS_TAIR_MEMPOOL_SSD")
        self.assertEqual("CPS_ALWAYS_TAIR_MEMPOOL_SSD", config.to_json_data()["data_storage_strategy"])


if __name__ == "__main__":
    unittest.main()
