import argparse

from .util import *
from ..common.http_helper import *
from ...util.json_helper import *

'''
curl -g -vvv -X POST http://localhost:56040/api/createInstanceGroup \
  -H "Content-Type: application/json" \
  -H "Accept: application/json" \
  -d '{
    "trace_id": "default_trace_id",
    "instance_group": {
        "name": "hf3fs_group",
        "storage_candidates": ["3fs_01"],
        "global_quota_group_name": "default_quota",
        "max_instance_count": 100,
        "quota": {
            "capacity": 30000000000,
            "quota_config": [
                {
                    "storage_type": "ST_NFS",
                    "capacity": "10000000000"
                },
                {
                    "storage_type": "ST_3FS",
                    "capacity": "10000000000"
                },
                {
                    "storage_type": "ST_TAIRMEMPOOL",
                    "capacity": "10000000000"
                }
            ]
        },
        "cache_config": {
            "reclaim_strategy": {
                "storage_unique_name": "3fs_01",
                "reclaim_policy": "POLICY_LRU",
                "trigger_strategy": {
                    "used_size": 0,
                    "used_percentage": 0.8
                },
                "trigger_period_seconds": 60,
                "reclaim_step_size": 1073741824,
                "reclaim_step_percentage": 10,
                "delay_before_delete_ms": 1000
            },
            "data_storage_strategy": "CPS_PREFER_3FS",
            "meta_indexer_config": {
                "max_key_count": 10000,
                "mutex_shard_num": 1024,
                "meta_storage_backend_config": {
                    "storage_type": "local",
                    "storage_uri": ""
                },
                "meta_cache_policy_config": {
                    "type": "LRU",
                    "capacity": 100000,
                    "cache_shard_bits": 0,
                    "high_pri_pool_ratio": 0.0
                }
            }
        },
        "user_data": "test user data",
        "version": 1
    }
}'
'''

def create_instance_group(args) -> InstanceGroup:
    instance_group_quota = InstanceGroupQuota(args.quota_capacity, args.quota_configs)
    reclaim_strategy = ReclaimStrategy(storage_unique_name=args.storage_candidates[0],
                                       reclaim_policy=args.reclaim_policy,
                                       trigger_used_percentage=args.reclaim_used_percentage)
    meta_cache_policy_config = MetaCachePolicyConfig(capacity=args.search_cache_capacity,
                                                     cache_shard_bits=args.search_cache_shard_bits)
    meta_indexer_config = MetaIndexerConfig(max_key_count=args.max_key_count,
                                            mutex_shard_num=args.mutex_shard_num,
                                            batch_key_size=args.batch_key_size,
                                            meta_storage_backend_config=args.meta_storage_backend_config,
                                            meta_cache_policy_config = meta_cache_policy_config)
    cache_config = CacheConfig(data_storage_strategy=args.data_storage_strategy,
                               reclaim_strategy=reclaim_strategy,
                               meta_indexer_config=meta_indexer_config)
    instance_group = InstanceGroup(name = args.name,
                                   storage_candidates = args.storage_candidates,
                                   instance_group_quota = instance_group_quota,
                                   quota_group_name = "default_quota_group",
                                   max_instance_count = args.max_instance_count,
                                   cache_config = cache_config,
                                   user_data = args.user_data,
                                   version=1)
    return instance_group
    


def main():
    args = parse_instance_group_args(is_create=True)
    instance_group = create_instance_group(args)
    data = {
        "trace_id": args.trace_id,
        "instance_group": instance_group.to_json_data()
    }
    result = http_post(args.host, "/api/createInstanceGroup", data, args.verbose)
    pretty_print_json(result)

if __name__ == "__main__":
    main()