import argparse
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

# Bazel keeps sources under package/kvcm_ops while the wheel exposes kvcm_ops
# as a top-level namespace package.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from kvcm_ops.kvcm.instance_group import update_instance_group
from kvcm_ops.kvcm.instance_group.util import (
    CacheConfig,
    InstanceGroup,
    InstanceGroupQuota,
    ReclaimStrategy,
    StorageQuota,
    instance_reclaim_budget_policy_value,
    parse_instance_group_args,
)


def _make_instance_group(instance_reclaim_budget_policy="USAGE_PROPORTIONAL"):
    quota = InstanceGroupQuota(1000, [StorageQuota("ST_NFS", 100)])
    reclaim_strategy = ReclaimStrategy(
        storage_unique_name="nfs_01",
        instance_reclaim_budget_policy=instance_reclaim_budget_policy,
    )
    return InstanceGroup(
        name="g1",
        storage_candidates=["nfs_01"],
        instance_group_quota=quota,
        quota_group_name="default_quota_group",
        cache_config=CacheConfig(reclaim_strategy=reclaim_strategy),
    )


class ReclaimStrategyBudgetPolicyTest(unittest.TestCase):
    def test_defaults_to_usage_proportional(self):
        self.assertEqual(
            "USAGE_PROPORTIONAL",
            ReclaimStrategy().to_json_data()["instance_reclaim_budget_policy"])

    def test_json_round_trip_preserves_fixed_per_instance(self):
        data = ReclaimStrategy(instance_reclaim_budget_policy="FIXED_PER_INSTANCE").to_json_data()
        restored = ReclaimStrategy.from_json_data(data)
        self.assertEqual(
            "FIXED_PER_INSTANCE",
            restored.to_json_data()["instance_reclaim_budget_policy"])

    def test_missing_json_field_defaults_to_usage_proportional(self):
        data = ReclaimStrategy(instance_reclaim_budget_policy="FIXED_PER_INSTANCE").to_json_data()
        del data["instance_reclaim_budget_policy"]
        restored = ReclaimStrategy.from_json_data(data)
        self.assertEqual(
            "USAGE_PROPORTIONAL",
            restored.to_json_data()["instance_reclaim_budget_policy"])

    def test_unknown_policy_is_rejected(self):
        data = ReclaimStrategy().to_json_data()
        data["instance_reclaim_budget_policy"] = "UNKNOWN"
        with self.assertRaises(RuntimeError):
            ReclaimStrategy.from_json_data(data)


class BudgetPolicyValueTest(unittest.TestCase):
    def test_valid_values(self):
        for value, expected in (
                ("usage_proportional", "USAGE_PROPORTIONAL"),
                ("USAGE_PROPORTIONAL", "USAGE_PROPORTIONAL"),
                ("fixed_per_instance", "FIXED_PER_INSTANCE"),
                ("FIXED_PER_INSTANCE", "FIXED_PER_INSTANCE")):
            with self.subTest(value=value):
                self.assertEqual(expected, instance_reclaim_budget_policy_value(value))

    def test_invalid_value(self):
        with self.assertRaises(argparse.ArgumentTypeError):
            instance_reclaim_budget_policy_value("yes")


class ParseInstanceGroupArgsTest(unittest.TestCase):
    def _parse(self, is_create, *extra_args):
        argv = ["prog", "--name", "g1"]
        if is_create:
            argv += ["--storage_candidates", "nfs_01"]
        argv += list(extra_args)
        with patch("sys.argv", argv):
            return parse_instance_group_args(is_create=is_create)

    def test_create_defaults_to_usage_proportional(self):
        self.assertEqual("USAGE_PROPORTIONAL", self._parse(True).instance_reclaim_budget_policy)

    def test_create_accepts_fixed_per_instance(self):
        self.assertEqual(
            "FIXED_PER_INSTANCE",
            self._parse(True, "--instance_reclaim_budget_policy", "fixed_per_instance")
            .instance_reclaim_budget_policy)

    def test_update_omitted_keeps_no_attribute(self):
        self.assertFalse(hasattr(self._parse(False), "instance_reclaim_budget_policy"))

    def test_update_accepts_explicit_value(self):
        self.assertEqual(
            "USAGE_PROPORTIONAL",
            self._parse(False, "--instance_reclaim_budget_policy", "usage_proportional")
            .instance_reclaim_budget_policy)


class UpdateInstanceGroupTest(unittest.TestCase):
    def _run_update(self, server_policy, *extra_args):
        instance_group = _make_instance_group(server_policy).to_json_data()
        with patch("sys.argv", ["prog", "--name", "g1", *extra_args]), \
             patch.object(update_instance_group, "http_post") as mock_http_post:
            mock_http_post.side_effect = [
                {"header": {"status": {"code": "OK"}}, "instance_group": instance_group},
                {"header": {"status": {"code": "OK"}}},
            ]
            update_instance_group.main()
            request = mock_http_post.call_args_list[1].args[2]
            return request["instance_group"]["cache_config"]["reclaim_strategy"][
                "instance_reclaim_budget_policy"]

    def test_omitted_preserves_fixed_per_instance_from_server(self):
        self.assertEqual(
            "FIXED_PER_INSTANCE",
            self._run_update("FIXED_PER_INSTANCE", "--user_data", "changed"))

    def test_explicit_value_overrides_server_value(self):
        self.assertEqual(
            "USAGE_PROPORTIONAL",
            self._run_update(
                "FIXED_PER_INSTANCE",
                "--instance_reclaim_budget_policy",
                "USAGE_PROPORTIONAL"))


if __name__ == "__main__":
    unittest.main()
