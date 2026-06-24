import argparse
import json
from .util import *
from ..common.http_helper import *
from ...util.json_helper import *

def get_current_instance_group(args):
    data = {
        "trace_id": args.trace_id + "_get",
        "name": args.name
    }
    result = http_post(args.host, "/api/getInstanceGroup", data, args.verbose)
    if result["header"]["status"]["code"] != "OK":
        raise RuntimeError(f"getInstanceGroup failed, result:[{result}]")
    print(result)
    return InstanceGroup.from_json_data(result["instance_group"])

def main():
    args = parse_instance_group_args(is_create=False)
    # print(args)
    current = get_current_instance_group(args)
    print("current instance group:")
    print(current.to_json_data())
    if hasattr(args, "storage_candidates"):
        current._storage_candidates = args.storage_candidates
    if hasattr(args, "user_data"):
        current._user_data = args.user_data
    if hasattr(args, "max_instance_count"):
        current._max_instance_count = args.max_instance_count
    if hasattr(args, "quota_capacity"):
        current._instance_group_quota._capacity = args.quota_capacity
    if hasattr(args, "quota_configs"):
        current._instance_group_quota._storage_qoutas = args.quota_configs
    if hasattr(args, "reclaim_policy"):
        current._cache_config._reclaim_strategy._reclaim_policy = args.reclaim_policy
    if hasattr(args, "reclaim_used_percentage"):
        current._cache_config._reclaim_strategy._trigger_used_percentage = args.reclaim_used_percentage
    if hasattr(args, "data_storage_strategy"):
        current._cache_config._data_storage_strategy = args.data_storage_strategy
    if hasattr(args, "max_key_count"):
        current._cache_config._meta_indexer_config._max_key_count = args.max_key_count
    if hasattr(args, "mutex_shard_num"):
        current._cache_config._meta_indexer_config._mutex_shard_num = args.mutex_shard_num
    if hasattr(args, "batch_key_size"):
        current._cache_config._meta_indexer_config._batch_key_size = args.batch_key_size
    if hasattr(args, "meta_storage_backend_config"):
        current._cache_config._meta_indexer_config._meta_storage_backend_config = args.meta_storage_backend_config
    if hasattr(args, "search_cache_capacity"):
        current._cache_config._meta_indexer_config._meta_cache_policy_config._capacity = args.search_cache_capacity
    if hasattr(args, "search_cache_shard_bits"):
        current._cache_config._meta_indexer_config._meta_cache_policy_config._cache_shard_bits = args.search_cache_shard_bits
    if hasattr(args, "extra_info") and args.extra_info:
        existing = {}
        if current._extra_info:
            existing = json.loads(current._extra_info)
        incoming = json.loads(args.extra_info)
        existing.update(incoming)
        current._extra_info = json.dumps(existing, ensure_ascii=False)
    if hasattr(args, "event_reporting_storage_candidates"):
        current._event_reporting_storage_candidates = args.event_reporting_storage_candidates
    current.check()
    current_version = current._version
    current._version += 1
    print("new instance group:")
    print(current.to_json_data())
    data = {
        "trace_id" : args.trace_id,
        "instance_group" : current.to_json_data(),
        "current_version" : current_version
    }
    result = http_post(args.host, "/api/updateInstanceGroup", data, args.verbose)
    pretty_print_json(result)

if __name__ == "__main__":
    main()