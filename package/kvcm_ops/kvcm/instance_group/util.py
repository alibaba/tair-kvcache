import argparse
from ..common.json_data import *
from ..common.common_args import *

class StorageQuota(JsonData):
    def __init__(self,
                 s_type: str,
                 capacity: int):
        self._type = s_type
        self._capacity = capacity
        self.check()
    
    def to_json_data(self) -> dict:
        return {
            "storage_type" : self._type,
            "capacity" : self._capacity
        }
    
    def check(self) -> bool:
        _type = self._type.upper()
        if not _type in ["ST_3FS", "ST_TAIRMEMPOOL", "ST_NFS"]:
            raise RuntimeError(f"storage type {self._type} invalid, support ST_3FS|ST_TAIRMEMPOOL|ST_NFS")
        self._type = _type
        if self._capacity <= 0:
            raise RuntimeError(f"StorageQuota capacity {self._capacity} <= 0")
        return True
    
    @classmethod
    def from_json_data(cls, json_data: dict):
        if JsonData.expect_exist("storage_type", json_data, str):
            s_type = json_data["storage_type"]
        if JsonData.expect_exist("capacity", json_data, (str, int)):
            capacity = int(json_data["capacity"])
        return cls(s_type, capacity)

def split_storage_quotas_value(value: str):
    storage_quota_strs = split_strs(value, ';')
    result = []
    for storage_quota_str in storage_quota_strs:
        storage_quota_str_split = split_strs(storage_quota_str)
        if len(storage_quota_str_split) != 2:
            raise argparse.ArgumentTypeError(f"Invalid storage_quota format: expected 'type,capacity', got '{value}'")
        s_type, capacity_str = storage_quota_str_split
        result.append(StorageQuota(s_type, int(capacity_str)))
    return result

class InstanceGroupQuota(JsonData):
    def __init__(self,
                 capacity: int,
                 storage_qoutas):
        self._capacity = capacity
        self._storage_qoutas = storage_qoutas
        self.check()
    
    def to_json_data(self) -> dict:
        quota_configs = []
        for storage_quota in self._storage_qoutas:
            quota_configs.append(storage_quota.to_json_data())
        return {
            "capacity" : self._capacity,
            "quota_config" : quota_configs
        }
    
    def check(self) -> bool:
        if self._capacity <= 0:
            raise RuntimeError(f"InstanceGroupQuota capacity {self._capacity} <= 0")
        for storage_qouta in self._storage_qoutas:
            storage_qouta.check()
        return True
    
    @classmethod
    def from_json_data(cls, json_data: dict):
        if JsonData.expect_exist("capacity", json_data, (str, int)):
            capacity = int(json_data["capacity"])
        if JsonData.expect_exist("quota_config", json_data, list):
            storage_qouta_objs = json_data["quota_config"]
            storage_qoutas = []
            for storage_quota_obj in storage_qouta_objs:
                storage_qoutas.append(StorageQuota.from_json_data(storage_quota_obj))
        return cls(capacity, storage_qoutas)

class ReclaimStrategy(JsonData):
    def __init__(self,
                 storage_unique_name: str = "",
                 reclaim_policy: str = "POLICY_LRU", # POLICY_LRU|POLICY_LFU|POLICY_TTL
                 trigger_used_percentage: float = 0.8,
                 trigger_used_size: int = 0,
                 trigger_period_seconds: int = 0,
                 reclaim_step_size: int = 0,
                 reclaim_step_percentage: float = 0.0,
                 delay_before_delete_ms: int = 1000
                 ):
        self._storage_unique_name = storage_unique_name
        self._reclaim_policy = reclaim_policy
        self._trigger_used_percentage = trigger_used_percentage
        self._trigger_used_size = trigger_used_size
        self._trigger_period_seconds = trigger_period_seconds
        self._reclaim_step_size = reclaim_step_size
        self._reclaim_step_percentage = reclaim_step_percentage
        self._delay_before_delete_ms = delay_before_delete_ms
        self.check()
    
    def to_json_data(self) -> dict:
        return {
            "storage_unique_name" : self._storage_unique_name,
            "reclaim_policy" : self._reclaim_policy,
            "trigger_strategy" : {
                "used_size" : self._trigger_used_size,
                "used_percentage" : self._trigger_used_percentage
            },
            "trigger_period_seconds" : self._trigger_period_seconds,
            "reclaim_step_size" : self._reclaim_step_size,
            "reclaim_step_percentage" : self._reclaim_step_percentage,
            "delay_before_delete_ms" : self._delay_before_delete_ms
        }
    
    def check(self) -> bool:
        _reclaim_policy = self._reclaim_policy.upper()
        if not _reclaim_policy in ["POLICY_LRU", "POLICY_LFU", "POLICY_TTL"]:
            raise RuntimeError(f"reclaim_policy {_reclaim_policy} invalid, support POLICY_LRU|POLICY_LFU|POLICY_TTL")
        self._reclaim_policy = _reclaim_policy
        return True
    
    @classmethod
    def from_json_data(cls, json_data: dict):
        if JsonData.expect_exist("storage_unique_name", json_data, str):
            storage_unique_name = json_data["storage_unique_name"]
        if JsonData.expect_exist("reclaim_policy", json_data, str):
            reclaim_policy = json_data["reclaim_policy"]
        if JsonData.expect_exist("trigger_strategy", json_data, dict):
            trigger_strategy = json_data["trigger_strategy"]
            if JsonData.expect_exist("used_percentage", trigger_strategy, (float, int)):
                trigger_used_percentage = float(trigger_strategy["used_percentage"])
            if JsonData.expect_exist("used_size", trigger_strategy, (str, int)):
                trigger_used_size = int(trigger_strategy["used_size"])
        if JsonData.expect_exist("trigger_period_seconds", json_data, (str, int)):
            trigger_period_seconds = int(json_data["trigger_period_seconds"])
        if JsonData.expect_exist("reclaim_step_size", json_data, (str, int)):
            reclaim_step_size = int(json_data["reclaim_step_size"])
        if JsonData.expect_exist("reclaim_step_percentage", json_data, (float, int)):
            reclaim_step_percentage = float(json_data["reclaim_step_percentage"])
        if JsonData.expect_exist("delay_before_delete_ms", json_data, (str, int)):
            delay_before_delete_ms = int(json_data["delay_before_delete_ms"])
        return cls(storage_unique_name, reclaim_policy, trigger_used_percentage, trigger_used_size, trigger_period_seconds, reclaim_step_size, reclaim_step_percentage, delay_before_delete_ms)

class MetaStorageBackendConfig(JsonData):
    def __init__(self,
                 storage_type: str = "local", # local|redis|cached
                 storage_uri: str = ""): # if set empty, no persistence
        self._storage_type = storage_type
        self._storage_uri = storage_uri
        self.check()
    
    def to_json_data(self) -> dict:
        return {
            "storage_type" : self._storage_type,
            "storage_uri" : self._storage_uri
        }
    
    def check(self) -> bool:
        _storage_type = self._storage_type.lower()
        if not _storage_type in ["local", "redis", "cached"]:
            raise RuntimeError(f"MetaStorageBackendConfig type {_storage_type} invalid, support local|redis|cached")
        self._storage_type = _storage_type
    
    @classmethod
    def from_json_data(cls, json_data: dict):
        if JsonData.expect_exist("storage_type", json_data, str):
            storage_type = json_data["storage_type"]
        if JsonData.expect_exist("storage_uri", json_data, str):
            storage_uri = json_data["storage_uri"]
        return cls(storage_type, storage_uri)

def meta_storage_backend_config_value(value: str):
    config_strs = split_strs(value)
    if len(config_strs) == 1:
        return MetaStorageBackendConfig(config_strs[0])
    if len(config_strs) == 2:
        return MetaStorageBackendConfig(config_strs[0], config_strs[1])
    raise argparse.ArgumentTypeError(f"Invalid config value, expect 'type' or 'type,uri', got '{value}'")

class MetaCachePolicyConfig(JsonData):
    def __init__(self,
                 capacity: int = 10 * 1024, # MB
                 m_type: str = "LRU",
                 cache_shard_bits: int = 6,
                 high_pri_pool_ratio: float = 0.0):
        self._capacity = capacity
        self._type = m_type
        self._cache_shard_bits = cache_shard_bits
        self._high_pri_pool_ratio = high_pri_pool_ratio
        self.check()

    def to_json_data(self) -> dict:
        return {
            "capacity" : self._capacity,
            "type" : self._type,
            "cache_shard_bits" : self._cache_shard_bits,
            "high_pri_pool_ratio" : self._high_pri_pool_ratio
        }
    
    def check(self) -> bool:
        _type = self._type.upper()
        if _type != "LRU":
            raise RuntimeError(f"MetaCachePolicyConfig type {_type} invalid, only support LRU now")
        self._type = _type
    
    @classmethod
    def from_json_data(cls, json_data: dict):
        if JsonData.expect_exist("capacity", json_data, (str, int)):
            capacity = int(json_data["capacity"])
        if JsonData.expect_exist("type", json_data, str):
            m_type = json_data["type"]
        if JsonData.expect_exist("cache_shard_bits", json_data, (str, int)):
            cache_shard_bits = int(json_data["cache_shard_bits"])
        if JsonData.expect_exist("high_pri_pool_ratio", json_data, (float, int)):
            high_pri_pool_ratio = json_data["high_pri_pool_ratio"]
        return cls(capacity, m_type, cache_shard_bits, high_pri_pool_ratio)

class MetaIndexerConfig(JsonData):
    def __init__(self,
                 max_key_count: int = 1 * 1000 * 1000 * 1000,
                 mutex_shard_num: int = 512,
                 batch_key_size: int = 128,
                 meta_storage_backend_config: MetaStorageBackendConfig = MetaStorageBackendConfig(),
                 meta_cache_policy_config : MetaCachePolicyConfig = MetaCachePolicyConfig()):
        self._max_key_count = max_key_count
        self._mutex_shard_num = mutex_shard_num
        self._batch_key_size = batch_key_size
        self._meta_storage_backend_config = meta_storage_backend_config
        self._meta_cache_policy_config = meta_cache_policy_config
        self.check()

    def to_json_data(self) -> dict:
        return {
            "max_key_count" : self._max_key_count,
            "mutex_shard_num" : self._mutex_shard_num,
            "batch_key_size" : self._batch_key_size,
            "meta_storage_backend_config" : self._meta_storage_backend_config.to_json_data(),
            "meta_cache_policy_config" : self._meta_cache_policy_config.to_json_data()
        }
    
    def check(self) -> bool:
        self._meta_storage_backend_config.check()
        self._meta_cache_policy_config.check()
    
    @classmethod
    def from_json_data(cls, json_data: dict):
        if JsonData.expect_exist("max_key_count", json_data, (str, int)):
            max_key_count = int(json_data["max_key_count"])
        if JsonData.expect_exist("mutex_shard_num", json_data, (str, int)):
            mutex_shard_num = int(json_data["mutex_shard_num"])
        if JsonData.expect_exist("batch_key_size", json_data, (str, int)):
            batch_key_size = int(json_data["batch_key_size"])
        if JsonData.expect_exist("meta_storage_backend_config", json_data, dict):
            meta_storage_backend_config = MetaStorageBackendConfig.from_json_data(json_data["meta_storage_backend_config"])
        if JsonData.expect_exist("meta_cache_policy_config", json_data, dict):
            meta_cache_policy_config = MetaCachePolicyConfig.from_json_data(json_data["meta_cache_policy_config"])
        return cls(max_key_count, mutex_shard_num, batch_key_size, meta_storage_backend_config, meta_cache_policy_config)

class CacheConfig(JsonData):
    def __init__(self,
                 data_storage_strategy: str = "CPS_PREFER_3FS",
                 reclaim_strategy: ReclaimStrategy = ReclaimStrategy(),
                 meta_indexer_config: MetaIndexerConfig = MetaIndexerConfig()):
        self._data_storage_strategy = data_storage_strategy
        self._reclaim_strategy = reclaim_strategy
        self._meta_indexer_config = meta_indexer_config
        self.check()

    def to_json_data(self) -> dict:
        return {
            "reclaim_strategy" : self._reclaim_strategy.to_json_data(),
            "data_storage_strategy" : self._data_storage_strategy,
            "meta_indexer_config" : self._meta_indexer_config.to_json_data()
        }
    
    def check(self) -> bool:
        _data_storage_strategy = self._data_storage_strategy.upper()
        if _data_storage_strategy not in ["CPS_ALWAYS_3FS", "CPS_PREFER_3FS", "CPS_ALWAYS_TAIR_MEMPOOL", "CPS_PREFER_TAIR_MEMPOOL"]:
            raise RuntimeError(f"data_storage_strategy {_data_storage_strategy} invalid, support CPS_ALWAYS_3FS|CPS_PREFER_3FS|CPS_ALWAYS_TAIR_MEMPOOL|CPS_PREFER_TAIR_MEMPOOL")
        self._data_storage_strategy = _data_storage_strategy
        self._reclaim_strategy.check()
        self._meta_indexer_config.check()

    @classmethod
    def from_json_data(cls, json_data: dict):
        if JsonData.expect_exist("data_storage_strategy", json_data, str):
            data_storage_strategy = json_data["data_storage_strategy"]
        if JsonData.expect_exist("reclaim_strategy", json_data, dict):
            reclaim_strategy = ReclaimStrategy.from_json_data(json_data["reclaim_strategy"])
        if JsonData.expect_exist("meta_indexer_config", json_data, dict):
            meta_indexer_config = MetaIndexerConfig.from_json_data(json_data["meta_indexer_config"])
        return cls(data_storage_strategy, reclaim_strategy, meta_indexer_config)

class InstanceGroup(JsonData):
    def __init__(self,
                 name: str,
                 storage_candidates,
                 instance_group_quota: InstanceGroupQuota,
                 quota_group_name: str,
                 max_instance_count: int = 100,
                 cache_config: CacheConfig = CacheConfig(),
                 user_data: str = "",
                 version: int = 1,
                 extra_info: str = "",
                 ):
        self._name = name
        self._storage_candidates = storage_candidates
        self._instance_group_quota = instance_group_quota
        self._quota_group_name = quota_group_name
        self._max_instance_count = max_instance_count
        self._cache_config = cache_config
        self._user_data = user_data
        self._version = version
        self._extra_info = extra_info
        self.check()

    def check(self):
        if not is_list_of_str(self._storage_candidates):
            raise RuntimeError(f"storage_candidates expect List[Str], real '{type(self._storage_candidates)}'")
        if self._extra_info:
            import json
            try:
                json.loads(self._extra_info)
            except json.JSONDecodeError as exc:
                raise RuntimeError(f"extra_info must be valid JSON, got: '{self._extra_info}'") from exc
        self._instance_group_quota.check()
        self._cache_config.check()

    def to_json_data(self) -> dict:
        return {
            "name" : self._name,
            "storage_candidates" : self._storage_candidates,
            "global_quota_group_name" : self._quota_group_name,
            "max_instance_count" : self._max_instance_count,
            "quota" : self._instance_group_quota.to_json_data(),
            "cache_config" : self._cache_config.to_json_data(),
            "user_data" : self._user_data,
            "version" : self._version,
            "extra_info" : self._extra_info,
        }
    
    @classmethod
    def from_json_data(cls, json_data: dict):
        if JsonData.expect_exist("name", json_data, str):
            name = json_data["name"]
        if JsonData.expect_exist("storage_candidates", json_data, list):
            storage_candidates = json_data["storage_candidates"]
        if JsonData.expect_exist("global_quota_group_name", json_data, str):
            quota_group_name = json_data["global_quota_group_name"]
        if JsonData.expect_exist("max_instance_count", json_data, (str, int)):
            max_instance_count = int(json_data["max_instance_count"])
        if JsonData.expect_exist("quota", json_data, dict):
            instance_group_quota = InstanceGroupQuota.from_json_data(json_data["quota"])
        if JsonData.expect_exist("cache_config", json_data, dict):
            cache_config = CacheConfig.from_json_data(json_data["cache_config"])
        if JsonData.expect_exist("user_data", json_data, str):
            user_data = json_data["user_data"]
        if JsonData.expect_exist("version", json_data, (str, int)):
            version = int(json_data["version"])
        extra_info = json_data.get("extra_info", "")
        return cls(name, storage_candidates, instance_group_quota, quota_group_name, max_instance_count, cache_config, user_data, version, extra_info)
    
# create or update
def parse_instance_group_args(is_create: bool):
    common_parser = create_common_parser()
    method = "create_instance_group" if is_create else "update_instance_group"
    parser = argparse.ArgumentParser(
        prog=f"python3 script.kvcm.storage.{method}",
        description=f"kvcm: {method}.",
        parents=[common_parser],
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--name",
        "-n",
        type=str,
        required=True,
        help="group name"
    )

    parser.add_argument(
        "--storage_candidates",
        "-s",
        type=split_strs,
        required=True if is_create else False,
        default=argparse.SUPPRESS,
        help="storage_candidates, eg. nfs_01 or nfs_01,nfs_02"
    )

    parser.add_argument(
        "--user_data",
        "-u",
        type=str,
        default="" if is_create else argparse.SUPPRESS,
        help="user_data"
    )

    parser.add_argument(
        "--max_instance_count",
        "-m",
        type=int,
        default=100 if is_create else argparse.SUPPRESS,
        help="max_instance_count"
    )

    parser.add_argument(
        "--quota_capacity",
        "-q",
        type=int,
        default=30000000000 if is_create else argparse.SUPPRESS,
        help="quota_capacity(bytes), eg. 30000000000"
    )

    parser.add_argument(
        "--quota_configs",
        type=split_storage_quotas_value,
        default=[StorageQuota("ST_NFS", 10000000000), StorageQuota("ST_3FS", 10000000000), StorageQuota("ST_TAIRMEMPOOL", 10000000000)] if is_create else argparse.SUPPRESS,
        help="quota_configs, eg. ST_NFS,10000000000;ST_3FS,10000000000;ST_TAIRMEMPOOL,10000000000"
    )

    parser.add_argument(
        "--reclaim_policy",
        type=str,
        default="POLICY_LRU" if is_create else argparse.SUPPRESS,
        help="reclaim_policy, POLICY_LRU, POLICY_LFU or, POLICY_TTL"
    )

    parser.add_argument(
        "--reclaim_used_percentage",
        type=float,
        default=0.8 if is_create else argparse.SUPPRESS,
        help="reclaim_used_percentage"
    )

    parser.add_argument(
        "--data_storage_strategy",
        type=str,
        default="CPS_PREFER_3FS" if is_create else argparse.SUPPRESS,
        help="data_storage_strategy, only support CPS_ALWAYS_3FS|CPS_PREFER_3FS|CPS_ALWAYS_TAIR_MEMPOOL|CPS_PREFER_TAIR_MEMPOOL now"
    )

    parser.add_argument(
        "--max_key_count",
        type=int,
        default=1 * 1000 * 1000 * 1000 if is_create else argparse.SUPPRESS,
        help="max_key_count"
    )

    parser.add_argument(
        "--mutex_shard_num",
        type=int,
        default=512 if is_create else argparse.SUPPRESS,
        help="meta_indexer_config.mutex_shard_num"
    )

    parser.add_argument(
        "--batch_key_size",
        type=int,
        default=128 if is_create else argparse.SUPPRESS,
        help="meta_indexer_config.batch_key_size"
    )

    parser.add_argument(
        "--meta_storage_backend_config",
        type=meta_storage_backend_config_value,
        default=MetaStorageBackendConfig() if is_create else argparse.SUPPRESS,
        help="meta_storage_backend_config, eg. local or local,/tmp/meta_tmp"
    )

    parser.add_argument(
        "--search_cache_capacity",
        type=int,
        default=10 * 1024 if is_create else argparse.SUPPRESS, # 10 * 1024 MB
        help="search_cache_capacity(MB)"
    )

    parser.add_argument(
        "--search_cache_shard_bits",
        type=int,
        default=6 if is_create else argparse.SUPPRESS,
        help="search_cache_shard_bits"
    )

    parser.add_argument(
        "--extra_info",
        type=str,
        default="" if is_create else argparse.SUPPRESS,
        help=(
            "Opaque JSON string passed through to clients (e.g. V6D). "
            "On update, the provided JSON is merged into existing extra_info."
        )
    )

    args = parser.parse_args()
    return args