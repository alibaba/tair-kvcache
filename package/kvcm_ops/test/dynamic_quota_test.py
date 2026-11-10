from pathlib import Path
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from kvcm_ops.kvcm.instance_group.util import (
    InstanceGroupQuota, StorageQuota, parse_instance_group_args, split_storage_quotas_value,
)
from kvcm_ops.kvcm.bootstrap import _apply_managed_group_fields, _build_new_instance_group, parse_environment


class DynamicQuotaTest(unittest.TestCase):
    def test_fixed_and_automatic_cli_roundtrip(self):
        quotas = split_storage_quotas_value("ST_TAIRMEMPOOL,auto,0.5,pool1|pool2;ST_TAIRMEMPOOL_SSD,160000")
        value = InstanceGroupQuota(-1, quotas).to_json_data()
        self.assertEqual(value, InstanceGroupQuota.from_json_data(value).to_json_data())
        self.assertEqual(["pool1", "pool2"], value["quota_config"][0]["storage_names"])
        self.assertEqual(0.5, value["quota_config"][0]["ratio"])
        self.assertEqual("fixed", value["quota_config"][1]["mode"])
        self.assertEqual(1.0, split_storage_quotas_value("ST_TAIRMEMPOOL,auto")[0]._ratio)

    def test_invalid_ratio_and_mode(self):
        for ratio in (0, -0.1, 1.1, float("nan"), float("inf")):
            with self.subTest(ratio=ratio), self.assertRaises(RuntimeError):
                StorageQuota("ST_TAIRMEMPOOL", mode="auto", ratio=ratio)
        with self.assertRaises(RuntimeError):
            StorageQuota("ST_TAIRMEMPOOL", mode="typo")
        with self.assertRaises(RuntimeError):
            InstanceGroupQuota(-2, [])

    def test_legacy_fixed_and_zero_are_preserved(self):
        self.assertEqual([], InstanceGroupQuota.from_json_data({"capacity": "-1"})._storage_qoutas)
        value = {"capacity": "0", "quota_config": [{"storage_type": "ST_NFS", "capacity": "0"}]}
        quota = InstanceGroupQuota.from_json_data(value)
        self.assertEqual(0, quota._capacity)
        self.assertEqual("fixed", quota._storage_qoutas[0]._mode)
        self.assertEqual(0, quota._storage_qoutas[0]._capacity)

    def test_new_group_default_and_update_arguments(self):
        with patch("sys.argv", ["prog", "--name", "group", "--storage_candidates", "pool"]):
            args = parse_instance_group_args(is_create=True)
        self.assertEqual(-1, args.quota_capacity)
        self.assertEqual([], args.quota_configs)
        with patch("sys.argv", ["prog", "--name", "group"]):
            args = parse_instance_group_args(is_create=False)
        self.assertFalse(hasattr(args, "quota_capacity"))
        self.assertFalse(hasattr(args, "quota_configs"))

    def test_bootstrap_preserves_existing_fixed_quota_until_explicitly_changed(self):
        environment = {
            "KVCM_PACE_STORAGE": '{"unique_name":"pool","domain":"http://meta.example","timeout":5}',
            "KVCM_INSTANCE_GROUP": '{"name":"group","meta_storage_backend_config":"local"}',
        }
        config = parse_environment(environment)
        group = _build_new_instance_group(config)
        self.assertEqual({"capacity": -1, "quota_config": []}, group["quota"])
        group["quota"] = {"capacity": 123, "quota_config": [{"storage_type": "ST_TAIRMEMPOOL", "capacity": 100}]}
        self.assertEqual(group["quota"], _apply_managed_group_fields(group, config)["quota"])
        environment["KVCM_INSTANCE_GROUP"] = (
            '{"name":"group","quota_capacity":-1,"meta_storage_backend_config":"local"}')
        updated = _apply_managed_group_fields(group, parse_environment(environment))
        self.assertEqual({"capacity": -1, "quota_config": []}, updated["quota"])


if __name__ == "__main__":
    unittest.main()
