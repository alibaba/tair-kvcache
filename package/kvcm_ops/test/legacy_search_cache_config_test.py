from pathlib import Path
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from kvcm_ops.kvcm.instance_group import create_instance_group, update_instance_group
from kvcm_ops.kvcm.instance_group.util import (
    CacheConfig,
    InstanceGroup,
    InstanceGroupQuota,
    MetaIndexerConfig,
    ReclaimStrategy,
    StorageQuota,
    parse_instance_group_args,
)


class LegacySearchCacheConfigTest(unittest.TestCase):
    def setUp(self):
        self.placeholder = {"type": "LRU", "capacity": 10 * 1024, "cache_shard_bits": 6, "high_pri_pool_ratio": 0.0}
        self.policy = {
            "capacity": 3072,
            "type": "LRU",
            "cache_shard_bits": 9,
            "high_pri_pool_ratio": 0.25,
        }
        group = InstanceGroup(
            name="g1",
            storage_candidates=["nfs_01"],
            instance_group_quota=InstanceGroupQuota(1000, [StorageQuota("ST_NFS", 1000)]),
            quota_group_name="default_quota_group",
            cache_config=CacheConfig(
                reclaim_strategy=ReclaimStrategy(storage_unique_name="nfs_01"),
                meta_indexer_config=MetaIndexerConfig()),
        )
        self.group = group.to_json_data()
        self.group["cache_config"]["meta_indexer_config"]["meta_cache_policy_config"] = self.policy

    def _update(self, *args):
        with patch("sys.argv", ["prog", "--name", "g1", *args]), \
             patch.object(update_instance_group, "http_post") as http_post, \
             patch("builtins.print"):
            http_post.side_effect = [
                {"header": {"status": {"code": "OK"}}, "instance_group": self.group},
                {"header": {"status": {"code": "OK"}}},
            ]
            update_instance_group.main()
            self.assertEqual(2, http_post.call_count)
            return http_post.call_args_list[1].args[2]["instance_group"]

    def test_create_accepts_legacy_flags(self):
        argv = ["prog", "--name", "g1", "--storage_candidates", "nfs_01",
                "--search_cache_capacity", "3072", "--search_cache_shard_bits", "9"]
        with patch("sys.argv", argv):
            args = parse_instance_group_args(is_create=True)
        group = create_instance_group.create_instance_group(args).to_json_data()
        policy = group["cache_config"]["meta_indexer_config"]["meta_cache_policy_config"]
        self.assertEqual(dict(self.placeholder, capacity=3072, cache_shard_bits=9), policy)

    def test_quota_update_normalizes_legacy_policy(self):
        group = self._update("--quota_capacity", "2000")
        self.assertEqual(2000, group["quota"]["capacity"])
        self.assertEqual(
            dict(self.placeholder, capacity=3072, cache_shard_bits=9),
            group["cache_config"]["meta_indexer_config"]["meta_cache_policy_config"])

    def test_update_accepts_legacy_flags(self):
        group = self._update("--search_cache_capacity", "4096", "--search_cache_shard_bits", "10")
        policy = group["cache_config"]["meta_indexer_config"]["meta_cache_policy_config"]
        self.assertEqual(dict(self.placeholder, capacity=4096, cache_shard_bits=10), policy)

    def test_update_preserves_zero_values(self):
        self.group["cache_config"]["meta_indexer_config"]["meta_cache_policy_config"].update(
            capacity="0", cache_shard_bits=0)
        group = self._update("--quota_capacity", "2000")
        policy = group["cache_config"]["meta_indexer_config"]["meta_cache_policy_config"]
        self.assertEqual(dict(self.placeholder, capacity=0, cache_shard_bits=0), policy)

    def test_update_tolerates_response_without_legacy_policy(self):
        del self.group["cache_config"]["meta_indexer_config"]["meta_cache_policy_config"]
        group = self._update("--quota_capacity", "2000")
        policy = group["cache_config"]["meta_indexer_config"]["meta_cache_policy_config"]
        self.assertEqual(self.placeholder, policy)


if __name__ == "__main__":
    unittest.main()
