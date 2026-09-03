import argparse
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

# Bazel keeps sources under package/kvcm_ops while the wheel exposes kvcm_ops
# as a top-level namespace package.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from kvcm_ops.kvcm.instance_group.util import (
    InstanceGroup,
    InstanceGroupQuota,
    StorageQuota,
    parse_bucket_boundaries,
    parse_instance_group_args,
    revisit_interval_buckets_value,
)
from kvcm_ops.kvcm.instance_group import update_instance_group


def _make_instance_group(revisit_interval_buckets=""):
    quota = InstanceGroupQuota(1000, [StorageQuota("ST_NFS", 100)])
    return InstanceGroup(
        name="g1",
        storage_candidates=["nfs_01"],
        instance_group_quota=quota,
        quota_group_name="default_quota_group",
        revisit_interval_buckets=revisit_interval_buckets,
    )


def _get_response(revisit_interval_buckets=None):
    instance_group = {
        "name": "g1",
        "storage_candidates": ["nfs_01"],
        "global_quota_group_name": "default_quota_group",
        "max_instance_count": 100,
        "quota": {
            "capacity": 30000000000,
            "quota_config": [{"storage_type": "ST_NFS", "capacity": "10000000000"}],
        },
        "cache_config": {
            "reclaim_strategy": {
                "storage_unique_name": "nfs_01",
                "reclaim_policy": "POLICY_LRU",
                "trigger_strategy": {"used_size": 0, "used_percentage": 0.8},
                "trigger_period_seconds": 60,
                "reclaim_step_size": 1,
                "reclaim_step_percentage": 10,
                "delay_before_delete_ms": 1000,
            },
            "data_storage_strategy": "CPS_PREFER_3FS",
            "meta_indexer_config": {
                "max_key_count": 10000,
                "mutex_shard_num": 1024,
                "batch_key_size": 128,
                "meta_storage_backend_config": {"storage_type": "local", "storage_uri": ""},
                "meta_cache_policy_config": {
                    "type": "LRU",
                    "capacity": 100000,
                    "cache_shard_bits": 0,
                    "high_pri_pool_ratio": 0.0,
                },
            },
        },
        "user_data": "",
        "version": 1,
    }
    if revisit_interval_buckets is not None:
        instance_group["revisit_interval_buckets"] = revisit_interval_buckets
    return {
        "header": {"status": {"code": "OK"}},
        "instance_group": instance_group,
    }


class ParseBucketBoundariesTest(unittest.TestCase):
    def test_valid_boundaries(self):
        self.assertEqual([1.0, 5.0, 30.0, 60.0], parse_bucket_boundaries("1,5,30,60"))
        self.assertEqual([0.5, 1.5, 5.0], parse_bucket_boundaries("0.5,1.5,5.0"))

    def test_whitespace_around_tokens_is_allowed(self):
        self.assertEqual([1.0, 5.0, 30.0], parse_bucket_boundaries(" 1 , 5 , 30 "))

    def test_single_boundary(self):
        self.assertEqual([60.0], parse_bucket_boundaries("60"))

    def test_invalid_values_are_rejected(self):
        invalid_values = (
            "5,1,30",      # not ascending
            "1,5,5,30",    # duplicate
            "-1,5,30",     # negative
            "0,5,30",      # zero
            "1s,5,30",     # trailing chars
            "abc,5,30",    # non numeric
            "1,,5",        # empty token
            ",1,5",        # leading comma
            "1,5,",        # trailing comma
            "inf",         # not finite
            "nan",         # not finite
            " ",           # whitespace only
        )
        for invalid_value in invalid_values:
            with self.subTest(value=invalid_value):
                with self.assertRaises(ValueError):
                    parse_bucket_boundaries(invalid_value)


class RevisitIntervalBucketsValueTest(unittest.TestCase):
    def test_valid_value_is_normalized(self):
        self.assertEqual("1,5,30", revisit_interval_buckets_value(" 1 , 5 ,30 "))

    def test_empty_string_clears_to_server_default(self):
        self.assertEqual("", revisit_interval_buckets_value(""))

    def test_invalid_value_raises_argument_type_error(self):
        with self.assertRaises(argparse.ArgumentTypeError):
            revisit_interval_buckets_value("5,1,30")


class InstanceGroupModelTest(unittest.TestCase):
    def test_field_defaults_to_empty(self):
        group = _make_instance_group()
        self.assertEqual("", group.to_json_data()["revisit_interval_buckets"])

    def test_json_round_trip_preserves_field(self):
        group = _make_instance_group("1,5,30,60")
        restored = InstanceGroup.from_json_data(group.to_json_data())
        self.assertEqual("1,5,30,60", restored.to_json_data()["revisit_interval_buckets"])

    def test_missing_json_field_defaults_to_empty(self):
        data = _make_instance_group("1,5,30").to_json_data()
        del data["revisit_interval_buckets"]
        restored = InstanceGroup.from_json_data(data)
        self.assertEqual("", restored.to_json_data()["revisit_interval_buckets"])

    def test_check_rejects_invalid_field(self):
        group = _make_instance_group("1,5,30")
        group._revisit_interval_buckets = "5,1,30"
        with self.assertRaises(RuntimeError):
            group.check()

    def test_constructor_rejects_invalid_field(self):
        with self.assertRaises(RuntimeError):
            _make_instance_group("1,,5")


class ParseInstanceGroupArgsTest(unittest.TestCase):
    def _parse(self, is_create, *extra_args):
        argv = ["prog", "--name", "g1"]
        if is_create:
            argv += ["--storage_candidates", "nfs_01"]
        argv += list(extra_args)
        with patch("sys.argv", argv):
            return parse_instance_group_args(is_create=is_create)

    def test_create_defaults_to_empty(self):
        args = self._parse(True)
        self.assertEqual("", args.revisit_interval_buckets)

    def test_create_accepts_and_normalizes_value(self):
        args = self._parse(True, "--revisit_interval_buckets", " 1 ,5 , 30")
        self.assertEqual("1,5,30", args.revisit_interval_buckets)

    def test_create_accepts_explicit_empty(self):
        args = self._parse(True, "--revisit_interval_buckets", "")
        self.assertEqual("", args.revisit_interval_buckets)

    def test_create_rejects_invalid_value(self):
        with self.assertRaises(SystemExit):
            self._parse(True, "--revisit_interval_buckets", "5,1,30")

    def test_update_omitted_keeps_no_attribute(self):
        args = self._parse(is_create=False)
        self.assertFalse(hasattr(args, "revisit_interval_buckets"))

    def test_update_accepts_explicit_value(self):
        args = self._parse(False, "--revisit_interval_buckets", "2,10,60")
        self.assertEqual("2,10,60", args.revisit_interval_buckets)

    def test_update_accepts_explicit_empty(self):
        args = self._parse(False, "--revisit_interval_buckets", "")
        self.assertEqual("", args.revisit_interval_buckets)

    def test_update_rejects_invalid_value(self):
        with self.assertRaises(SystemExit):
            self._parse(False, "--revisit_interval_buckets", "abc")


class UpdateInstanceGroupTest(unittest.TestCase):
    def _run_update(self, server_buckets, *extra_args):
        with patch("sys.argv", ["prog", "--name", "g1", *extra_args]), \
             patch.object(update_instance_group, "http_post") as mock_http_post:
            mock_http_post.side_effect = [
                _get_response(server_buckets),
                {"header": {"status": {"code": "OK"}}},
            ]
            update_instance_group.main()
            request = mock_http_post.call_args_list[1].args[2]
            return request["instance_group"]["revisit_interval_buckets"]

    def test_omitted_preserves_server_value(self):
        self.assertEqual("1,5,30", self._run_update("1,5,30", "--user_data", "changed"))

    def test_explicit_value_overrides_server_value(self):
        self.assertEqual("2,10,60", self._run_update("1,5,30", "--revisit_interval_buckets", "2,10,60"))

    def test_explicit_empty_clears_server_value(self):
        self.assertEqual("", self._run_update("1,5,30", "--revisit_interval_buckets", ""))

    def test_missing_server_field_stays_empty_when_omitted(self):
        self.assertEqual("", self._run_update(None, "--user_data", "changed"))


if __name__ == "__main__":
    unittest.main()
