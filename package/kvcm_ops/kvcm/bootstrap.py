import argparse
import copy
from enum import Enum
import json
import logging
import os
import time
from urllib.parse import parse_qs, urlsplit

from .common.http_helper import http_post
from .instance_group.util import (
    CacheConfig,
    InstanceGroup,
    InstanceGroupQuota,
    MetaCachePolicyConfig,
    MetaIndexerConfig,
    MetaStorageBackendConfig,
    ReclaimStrategy,
    StorageQuota,
    meta_storage_backend_config_value,
)


LOGGER = logging.getLogger("kvcm_ops.bootstrap")

DEFAULT_ADMIN_URL = "http://127.0.0.1:6492"
DEFAULT_MAX_INSTANCE_COUNT = 512
DEFAULT_QUOTA_CAPACITY = 1000000000
DEFAULT_MAX_KEY_COUNT = 1000000000
DEFAULT_MUTEX_SHARD_NUM = 131072
DEFAULT_BATCH_KEY_SIZE = 1024
DEFAULT_SEARCH_CACHE_CAPACITY = 10 * 1024
DEFAULT_SEARCH_CACHE_SHARD_BITS = 6
DEFAULT_RECLAIM_POLICY = "POLICY_LRU"
DEFAULT_RECLAIM_USED_PERCENTAGE = 0.8
CACHE_SHARD_BITS_UPPER_BOUND = 20
L1P5_STORAGE_ENV = "KVCM_L1P5_STORAGE"
L2_STORAGE_ENV = "KVCM_L2P_STORAGE"
PACE_STORAGE_ENV = "KVCM_PACE_STORAGE"
INSTANCE_GROUP_ENV = "KVCM_INSTANCE_GROUP"
INSTANCE_GROUP_UPDATE_RETRIES = 3
FOLLOWER_EXIT_CODE = 2
RESTART_REQUIRED_EXIT_CODE = 3


class BootstrapOutcome(Enum):
    COMPLETE = "complete"
    FOLLOWER = "follower"
    RESTART_REQUIRED = "restart_required"


# Keep aligned with EventReportStorageSpec defaults in
# kv_cache_manager/data_storage/storage_config.h.
EVENT_REPORT_DEFAULTS = {
    "heartbeat_timeout_ms": 30 * 1000,
    "cleanup_grace_ms": 5 * 60 * 1000,
    "liveness_check_interval_ms": 5 * 1000,
    "snapshot_min_interval_ms": 30 * 1000,
}

EVENT_REPORT_KEYS = set(EVENT_REPORT_DEFAULTS) | {"unique_name"}
PACE_STORAGE_KEYS = {
    "unique_name",
    "domain",
    "timeout",
    "service_discovery_url",
    "media_type",
}
INSTANCE_GROUP_KEYS = {
    "name",
    "user_data",
    "quota_capacity",
    "max_instance_count",
    "reclaim_policy",
    "reclaim_used_percentage",
    "max_key_count",
    "mutex_shard_num",
    "batch_key_size",
    "search_cache_capacity",
    "search_cache_shard_bits",
    "meta_storage_backend_config",
    "metadata_backend_mode",
}

POSITIVE_PASSTHROUGH_QUERY_KEYS = {
    "timeout_ms",
    "async_queue_count",
    "async_max_batch",
    "async_wait_us",
    "async_max_size",
    "async_sync_timeout_ms",
    "async_drain_ms",
    "client_max_pool_size",
    "sample_times",
}

NON_NEGATIVE_PASSTHROUGH_QUERY_KEYS = {
    "db",
    "async_enqueue_timeout_ms",
    "client_min_pool_size",
}


class BootstrapError(RuntimeError):
    pass


class NotLeaderError(BootstrapError):
    pass


class StorageBootstrapConfig:
    def __init__(self, env_name, unique_name, storage_type, spec, media_type_explicit=False):
        self.env_name = env_name
        self.unique_name = unique_name
        self.storage_type = storage_type
        self.spec = spec
        self.media_type_explicit = media_type_explicit


class BootstrapConfig:
    def __init__(self):
        self.l1p5_storage = None
        self.l2_storage = None
        self.pace_storage = None
        self.instance_group_name = ""
        self.user_data = ""
        self.meta_storage_type = ""
        self.meta_storage_uri = ""
        self.metadata_backend_mode = None
        self.max_instance_count = DEFAULT_MAX_INSTANCE_COUNT
        self.quota_capacity = DEFAULT_QUOTA_CAPACITY
        self.reclaim_policy = DEFAULT_RECLAIM_POLICY
        self.reclaim_used_percentage = DEFAULT_RECLAIM_USED_PERCENTAGE
        self.max_key_count = DEFAULT_MAX_KEY_COUNT
        self.mutex_shard_num = DEFAULT_MUTEX_SHARD_NUM
        self.batch_key_size = DEFAULT_BATCH_KEY_SIZE
        self.search_cache_capacity = DEFAULT_SEARCH_CACHE_CAPACITY
        self.search_cache_shard_bits = DEFAULT_SEARCH_CACHE_SHARD_BITS
        self.redis_host = ""
        self.redis_port = 0

    @property
    def enabled(self):
        return bool(self.storage_configs)

    @property
    def storage_configs(self):
        return [
            storage
            for storage in (self.l1p5_storage, self.l2_storage, self.pace_storage)
            if storage is not None
        ]

    @property
    def primary_storage(self):
        return self.pace_storage or self.l1p5_storage or self.l2_storage

    @property
    def primary_storage_name(self):
        return self.primary_storage.unique_name

    @property
    def primary_storage_type(self):
        return self.primary_storage.storage_type

    @property
    def event_report_storage_names(self):
        return [
            storage.unique_name
            for storage in (self.l1p5_storage, self.l2_storage)
            if storage is not None
        ]

    @property
    def data_storage_strategy(self):
        if self.pace_storage is not None:
            return "CPS_PREFER_TAIR_MEMPOOL"
        return "CPS_PREFER_3FS"


def _parse_json_object(environ, name, allowed_keys, required=False):
    raw_value = environ.get(name)
    if raw_value is None or not raw_value.strip():
        if required:
            raise BootstrapError("{} is required when any Storage is configured".format(name))
        return None
    try:
        value = json.loads(raw_value)
    except (TypeError, ValueError):
        raise BootstrapError("{} must be a valid JSON object".format(name))
    if not isinstance(value, dict):
        raise BootstrapError("{} must be a JSON object".format(name))
    unknown_keys = sorted(set(value) - allowed_keys)
    if unknown_keys:
        raise BootstrapError(
            "{} contains unknown fields: {}".format(name, ", ".join(unknown_keys)))
    return value


def _required_string(value, field, env_name):
    result = value.get(field)
    if not isinstance(result, str) or not result.strip():
        raise BootstrapError("{}.{} is required and must be a non-empty string".format(env_name, field))
    return result.strip()


def _optional_string(value, field, default, env_name):
    if field not in value:
        return default
    result = value[field]
    if not isinstance(result, str):
        raise BootstrapError("{}.{} must be a string".format(env_name, field))
    return result


def _integer(value, field, default, env_name, allow_zero=False):
    if field not in value:
        return default
    result = value[field]
    if type(result) is not int:
        raise BootstrapError("{}.{} must be an integer".format(env_name, field))
    if result < 0 or (result == 0 and not allow_zero):
        comparator = "non-negative" if allow_zero else "positive"
        raise BootstrapError("{}.{} must be {}".format(env_name, field, comparator))
    return result


def _number(value, field, default, env_name):
    if field not in value:
        return default
    result = value[field]
    if type(result) not in (int, float):
        raise BootstrapError("{}.{} must be a number".format(env_name, field))
    return float(result)


def _single_query_value(query, name):
    values = query.get(name)
    if not values:
        return None
    if len(values) != 1:
        raise BootstrapError("storage URI query parameter {} must appear once".format(name))
    if values[0] == "":
        raise BootstrapError("storage URI query parameter {} must not be empty".format(name))
    return values[0]


def _positive_query_int(query, name, allow_zero=False):
    raw_value = _single_query_value(query, name)
    if raw_value is None:
        return None
    try:
        value = int(raw_value)
    except (TypeError, ValueError):
        raise BootstrapError("storage URI query parameter {} must be an integer".format(name))
    if value < 0 or (value == 0 and not allow_zero):
        comparator = "non-negative" if allow_zero else "positive"
        raise BootstrapError("storage URI query parameter {} must be {}".format(name, comparator))
    return value


def _parse_storage_uri(uri):
    try:
        # KVCM StandardUri treats '#' in user-info as a literal character.
        # Escape it only in the validation copy so the original URI is passed
        # unchanged to KVCM and remains compatible with existing deployments.
        parsed = urlsplit(uri.replace("#", "%23"))
    except ValueError:
        raise BootstrapError("meta_storage_backend_config URI is not valid")
    if parsed.scheme.lower() != "redis":
        raise BootstrapError("meta_storage_backend_config URI scheme must be redis")
    if not parsed.hostname:
        raise BootstrapError("meta_storage_backend_config URI must include a host")
    try:
        port = parsed.port
    except ValueError:
        raise BootstrapError("meta_storage_backend_config URI contains an invalid port")
    if port is None or port <= 0 or port > 65535:
        raise BootstrapError("meta_storage_backend_config URI must include a port in range 1..65535")
    query = parse_qs(parsed.query, keep_blank_values=True)
    if _single_query_value(query, "cluster_name") is None:
        raise BootstrapError("meta_storage_backend_config URI must include cluster_name")
    return parsed.hostname, port, query


def _parse_meta_storage_backend_config(raw_value):
    try:
        parsed = meta_storage_backend_config_value(raw_value)
    except (argparse.ArgumentTypeError, RuntimeError, ValueError):
        raise BootstrapError(
            "KVCM_INSTANCE_GROUP.meta_storage_backend_config must use type,uri format "
            "with type redis or cached")
    parsed_data = parsed.to_json_data()
    storage_type = parsed_data["storage_type"]
    storage_uri = parsed_data["storage_uri"]
    if storage_type not in ("redis", "cached") or not storage_uri:
        raise BootstrapError(
            "KVCM_INSTANCE_GROUP.meta_storage_backend_config must use type,uri format "
            "with type redis or cached")
    return storage_type, storage_uri


def _validate_passthrough_query(query, meta_storage_type):
    for name in POSITIVE_PASSTHROUGH_QUERY_KEYS:
        _positive_query_int(query, name)
    for name in NON_NEGATIVE_PASSTHROUGH_QUERY_KEYS:
        _positive_query_int(query, name, allow_zero=True)
    queue_count = _positive_query_int(query, "async_queue_count")
    if queue_count is not None and queue_count > 2048:
        raise BootstrapError("storage URI query parameter async_queue_count must not exceed 2048")
    num_shard_bits = _positive_query_int(query, "num_shard_bits", allow_zero=True)
    if (meta_storage_type == "cached" and num_shard_bits is not None
            and num_shard_bits >= CACHE_SHARD_BITS_UPPER_BOUND):
        raise BootstrapError("storage URI query parameter num_shard_bits must be less than 20")


def _parse_event_report_storage(environ, env_name, storage_type):
    value = _parse_json_object(environ, env_name, EVENT_REPORT_KEYS)
    if value is None:
        return None
    spec = {}
    for field, default in EVENT_REPORT_DEFAULTS.items():
        spec[field] = _integer(value, field, default, env_name)
    return StorageBootstrapConfig(
        env_name,
        _required_string(value, "unique_name", env_name),
        storage_type,
        spec,
    )


def _parse_pace_storage(environ):
    value = _parse_json_object(environ, PACE_STORAGE_ENV, PACE_STORAGE_KEYS)
    if value is None:
        return None
    media_type = _integer(value, "media_type", 0, PACE_STORAGE_ENV, allow_zero=True)
    if media_type not in (0, 2):
        raise BootstrapError("{}.media_type must be 0 or 2".format(PACE_STORAGE_ENV))
    if "timeout" not in value:
        raise BootstrapError("{}.timeout is required and must be positive".format(PACE_STORAGE_ENV))
    spec = {
        "domain": _required_string(value, "domain", PACE_STORAGE_ENV),
        "timeout": _integer(value, "timeout", None, PACE_STORAGE_ENV),
        "service_discovery_url": _optional_string(
            value, "service_discovery_url", "", PACE_STORAGE_ENV),
        "media_type": media_type,
    }
    return StorageBootstrapConfig(
        PACE_STORAGE_ENV,
        _required_string(value, "unique_name", PACE_STORAGE_ENV),
        "ST_TAIRMEMPOOL",
        spec,
        media_type_explicit="media_type" in value,
    )


def parse_environment(environ=None):
    environ = os.environ if environ is None else environ
    config = BootstrapConfig()
    config.l1p5_storage = _parse_event_report_storage(
        environ, L1P5_STORAGE_ENV, "ST_EVENT_REPORT_L1P5")
    config.l2_storage = _parse_event_report_storage(
        environ, L2_STORAGE_ENV, "ST_EVENT_REPORT_L2")
    config.pace_storage = _parse_pace_storage(environ)

    storage_names = [storage.unique_name for storage in config.storage_configs]
    if len(storage_names) != len(set(storage_names)):
        raise BootstrapError("configured Storage unique_name values must be distinct")
    if not config.enabled:
        return config

    value = _parse_json_object(
        environ, INSTANCE_GROUP_ENV, INSTANCE_GROUP_KEYS, required=True)
    config.instance_group_name = _required_string(value, "name", INSTANCE_GROUP_ENV)
    config.user_data = _optional_string(
        value, "user_data", config.instance_group_name, INSTANCE_GROUP_ENV)
    config.quota_capacity = _integer(
        value, "quota_capacity", DEFAULT_QUOTA_CAPACITY, INSTANCE_GROUP_ENV)
    config.max_instance_count = _integer(
        value, "max_instance_count", DEFAULT_MAX_INSTANCE_COUNT, INSTANCE_GROUP_ENV)
    config.max_key_count = _integer(
        value, "max_key_count", DEFAULT_MAX_KEY_COUNT, INSTANCE_GROUP_ENV)
    config.mutex_shard_num = _integer(
        value, "mutex_shard_num", DEFAULT_MUTEX_SHARD_NUM, INSTANCE_GROUP_ENV)
    if config.mutex_shard_num & (config.mutex_shard_num - 1):
        raise BootstrapError("KVCM_INSTANCE_GROUP.mutex_shard_num must be a power of two")
    config.batch_key_size = _integer(
        value, "batch_key_size", DEFAULT_BATCH_KEY_SIZE, INSTANCE_GROUP_ENV)
    config.search_cache_capacity = _integer(
        value, "search_cache_capacity", DEFAULT_SEARCH_CACHE_CAPACITY, INSTANCE_GROUP_ENV)
    config.search_cache_shard_bits = _integer(
        value,
        "search_cache_shard_bits",
        DEFAULT_SEARCH_CACHE_SHARD_BITS,
        INSTANCE_GROUP_ENV,
        allow_zero=True,
    )
    if config.search_cache_shard_bits >= CACHE_SHARD_BITS_UPPER_BOUND:
        raise BootstrapError("KVCM_INSTANCE_GROUP.search_cache_shard_bits must be less than 20")

    config.reclaim_policy = _optional_string(
        value, "reclaim_policy", DEFAULT_RECLAIM_POLICY, INSTANCE_GROUP_ENV).upper()
    if config.reclaim_policy not in ("POLICY_LRU", "POLICY_LFU", "POLICY_TTL"):
        raise BootstrapError(
            "KVCM_INSTANCE_GROUP.reclaim_policy must be POLICY_LRU, POLICY_LFU, or POLICY_TTL")
    config.reclaim_used_percentage = _number(
        value,
        "reclaim_used_percentage",
        DEFAULT_RECLAIM_USED_PERCENTAGE,
        INSTANCE_GROUP_ENV,
    )
    if not 0 < config.reclaim_used_percentage <= 1:
        raise BootstrapError(
            "KVCM_INSTANCE_GROUP.reclaim_used_percentage must be in range (0, 1]")

    mode = value.get("metadata_backend_mode")
    if mode is not None:
        if type(mode) is not int or mode < 1 or mode > 4:
            raise BootstrapError(
                "KVCM_INSTANCE_GROUP.metadata_backend_mode must be an integer in range 1..4")
        config.metadata_backend_mode = mode

    raw_meta_storage = _required_string(
        value, "meta_storage_backend_config", INSTANCE_GROUP_ENV)
    config.meta_storage_type, config.meta_storage_uri = (
        _parse_meta_storage_backend_config(raw_meta_storage))
    config.redis_host, config.redis_port, query = _parse_storage_uri(config.meta_storage_uri)
    _validate_passthrough_query(query, config.meta_storage_type)
    return config


def _status_code(response):
    return response.get("header", {}).get("status", {}).get("code")


def _status_message(response):
    return response.get("header", {}).get("status", {}).get("message", "unknown error")


class AdminClient:
    def __init__(self, host=DEFAULT_ADMIN_URL):
        self.host = host.rstrip("/")
        self._trace_counter = 0

    def _trace_id(self, operation):
        self._trace_counter += 1
        return "bootstrap_{}_{}_{}".format(operation, int(time.time() * 1000), self._trace_counter)

    def post(self, api, data, operation):
        try:
            response = http_post(self.host, api, data, False)
        except Exception as exc:
            raise BootstrapError(
                "Admin API {} transport failed: {}: {}".format(
                    operation, type(exc).__name__, exc))
        code = _status_code(response)
        if code in ("SERVER_NOT_LEADER", 9):
            raise NotLeaderError(
                "Admin API {} rejected because this node is not leader".format(operation))
        if code not in ("OK", 1):
            raise BootstrapError(
                "Admin API {} failed with status code {}: {}".format(
                    operation, code, _status_message(response)))
        return response

    def check_health(self):
        return self.post(
            "/api/checkHealth",
            {"trace_id": self._trace_id("check_health")},
            "checkHealth",
        )

    def list_storage(self):
        response = self.post(
            "/api/listStorage",
            {"trace_id": self._trace_id("list_storage")},
            "listStorage",
        )
        return response.get("storage", [])

    def add_storage(self, storage):
        self.post(
            "/api/addStorage",
            {"trace_id": self._trace_id("add_storage"), "storage": storage},
            "addStorage",
        )

    def update_storage(self, storage):
        self.post(
            "/api/updateStorage",
            {
                "trace_id": self._trace_id("update_storage"),
                "storage": storage,
                "force_update": False,
            },
            "updateStorage",
        )

    def list_instance_groups(self):
        response = self.post(
            "/api/listInstanceGroup",
            {"trace_id": self._trace_id("list_group")},
            "listInstanceGroup",
        )
        return response.get("instance_group", [])

    def list_instance_info(self, instance_group_name):
        response = self.post(
            "/api/listInstanceInfo",
            {
                "trace_id": self._trace_id("list_instance"),
                "instance_group_name": instance_group_name,
            },
            "listInstanceInfo",
        )
        return response.get("instance_info", [])

    def create_instance_group(self, instance_group):
        self.post(
            "/api/createInstanceGroup",
            {
                "trace_id": self._trace_id("create_group"),
                "instance_group": instance_group,
            },
            "createInstanceGroup",
        )

    def update_instance_group(self, instance_group, current_version):
        self.post(
            "/api/updateInstanceGroup",
            {
                "trace_id": self._trace_id("update_group"),
                "instance_group": instance_group,
                "current_version": current_version,
            },
            "updateInstanceGroup",
        )


def _new_storage(storage_config, existing=None):
    if storage_config.storage_type == "ST_TAIRMEMPOOL":
        storage_spec = {}
        if existing is not None and isinstance(existing.get("tair_mem_pool"), dict):
            storage_spec.update(existing["tair_mem_pool"])
        storage_spec.update(storage_config.spec)
        if existing is not None and not storage_config.media_type_explicit:
            storage_spec["media_type"] = existing.get("tair_mem_pool", {}).get("media_type", 0)
        spec_name = "tair_mem_pool"
    else:
        storage_spec = {}
        if existing is not None and isinstance(existing.get("event_report"), dict):
            storage_spec.update(existing["event_report"])
        storage_spec.update(storage_config.spec)
        spec_name = "event_report"
    return {
        "global_unique_name": storage_config.unique_name,
        "storage_type": storage_config.storage_type,
        spec_name: storage_spec,
        "check_storage_available_when_open": True,
    }


def _storage_matches(storage_config, current):
    if current.get("storage_type") != storage_config.storage_type:
        return False
    spec_name = (
        "tair_mem_pool" if storage_config.storage_type == "ST_TAIRMEMPOOL" else "event_report")
    current_spec = current.get(spec_name)
    if not isinstance(current_spec, dict):
        return False
    for key, value in storage_config.spec.items():
        if (storage_config.storage_type == "ST_TAIRMEMPOOL" and key == "media_type"
                and not storage_config.media_type_explicit):
            continue
        current_value = current_spec.get(key)
        if type(value) is int:
            current_value = _as_int(current_value)
        if current_value != value:
            return False
    return True


def _validate_existing_pace_identity(storage_config, current):
    storage_spec = current.get("tair_mem_pool")
    if not isinstance(storage_spec, dict):
        raise BootstrapError(
            "storage '{}' is not a PACE storage".format(storage_config.unique_name))
    storage_type = current.get("storage_type", "ST_UNSPECIFIED")
    if storage_type not in ("ST_UNSPECIFIED", "ST_TAIRMEMPOOL"):
        raise BootstrapError(
            "storage '{}' has incompatible storage_type {}".format(
                storage_config.unique_name, storage_type))
    media_type = storage_spec.get("media_type", 0)
    if type(media_type) is not int or media_type not in (0, 2):
        raise BootstrapError(
            "storage '{}' has unsupported PACE media_type {}".format(
                storage_config.unique_name, media_type))
    if storage_config.media_type_explicit and storage_config.spec["media_type"] != media_type:
        raise BootstrapError(
            "storage '{}' PACE media_type cannot be changed; use a new unique_name".format(
                storage_config.unique_name))


def ensure_storages(client, config):
    current_by_name = {
        storage.get("global_unique_name"): storage
        for storage in client.list_storage()
    }
    for storage_config in config.storage_configs:
        current = current_by_name.get(storage_config.unique_name)
        if current is None:
            client.add_storage(_new_storage(storage_config))
            LOGGER.info("created managed storage %s", storage_config.unique_name)
            continue
        if storage_config.storage_type == "ST_TAIRMEMPOOL":
            _validate_existing_pace_identity(storage_config, current)
        if not _storage_matches(storage_config, current):
            client.update_storage(_new_storage(storage_config, current))
            LOGGER.info("updated managed storage %s", storage_config.unique_name)


def _build_new_instance_group(config):
    quota = InstanceGroupQuota(
        config.quota_capacity,
        [StorageQuota(config.primary_storage_type, config.quota_capacity)],
    )
    reclaim_strategy = ReclaimStrategy(
        storage_unique_name=config.primary_storage_name,
        reclaim_policy=config.reclaim_policy,
        trigger_used_percentage=config.reclaim_used_percentage,
    )
    meta_storage = MetaStorageBackendConfig(config.meta_storage_type, config.meta_storage_uri)
    meta_cache = MetaCachePolicyConfig(
        capacity=config.search_cache_capacity,
        cache_shard_bits=config.search_cache_shard_bits,
    )
    meta_indexer = MetaIndexerConfig(
        max_key_count=config.max_key_count,
        mutex_shard_num=config.mutex_shard_num,
        batch_key_size=config.batch_key_size,
        meta_storage_backend_config=meta_storage,
        meta_cache_policy_config=meta_cache,
    )
    cache_config = CacheConfig(
        data_storage_strategy=config.data_storage_strategy,
        reclaim_strategy=reclaim_strategy,
        meta_indexer_config=meta_indexer,
    )
    extra_info = ""
    if config.metadata_backend_mode is not None:
        extra_info = json.dumps(
            {"metadata_backend_mode": config.metadata_backend_mode}, sort_keys=True)
    return InstanceGroup(
        name=config.instance_group_name,
        storage_candidates=[config.primary_storage_name],
        instance_group_quota=quota,
        quota_group_name="default_quota_group",
        max_instance_count=config.max_instance_count,
        cache_config=cache_config,
        user_data=config.user_data,
        version=1,
        extra_info=extra_info,
        event_report_storage_candidates=config.event_report_storage_names,
    ).to_json_data()


def _as_int(value):
    try:
        return int(value)
    except (TypeError, ValueError):
        return value


def _extra_info_object(value):
    if not value:
        return {}
    if isinstance(value, dict):
        return copy.deepcopy(value)
    try:
        parsed = json.loads(value)
    except (TypeError, ValueError):
        raise BootstrapError("existing instance group extra_info is not valid JSON")
    if not isinstance(parsed, dict):
        raise BootstrapError("existing instance group extra_info must be a JSON object")
    return parsed


def _apply_managed_group_fields(current, config):
    updated = copy.deepcopy(current)
    updated["user_data"] = config.user_data
    updated["storage_candidates"] = [config.primary_storage_name]
    updated["event_report_storage_candidates"] = config.event_report_storage_names
    updated["max_instance_count"] = config.max_instance_count
    updated["quota"] = {
        "capacity": config.quota_capacity,
        "quota_config": [
            {
                "storage_type": config.primary_storage_type,
                "capacity": config.quota_capacity,
            }
        ],
    }
    cache_config = updated.setdefault("cache_config", {})
    cache_config["data_storage_strategy"] = config.data_storage_strategy
    reclaim_strategy = cache_config.setdefault("reclaim_strategy", {})
    reclaim_strategy["storage_unique_name"] = config.primary_storage_name
    reclaim_strategy["reclaim_policy"] = config.reclaim_policy
    trigger_strategy = reclaim_strategy.setdefault("trigger_strategy", {})
    trigger_strategy["used_percentage"] = config.reclaim_used_percentage
    meta_indexer = cache_config.setdefault("meta_indexer_config", {})
    meta_indexer["max_key_count"] = config.max_key_count
    meta_indexer["mutex_shard_num"] = config.mutex_shard_num
    meta_indexer["batch_key_size"] = config.batch_key_size
    meta_indexer["meta_storage_backend_config"] = {
        "storage_type": config.meta_storage_type,
        "storage_uri": config.meta_storage_uri,
    }
    meta_cache = meta_indexer.setdefault("meta_cache_policy_config", {})
    meta_cache["capacity"] = config.search_cache_capacity
    meta_cache["cache_shard_bits"] = config.search_cache_shard_bits
    extra_info = _extra_info_object(updated.get("extra_info", ""))
    if config.metadata_backend_mode is not None:
        extra_info["metadata_backend_mode"] = config.metadata_backend_mode
    else:
        extra_info.pop("metadata_backend_mode", None)
    updated["extra_info"] = json.dumps(extra_info, sort_keys=True, ensure_ascii=False)
    return updated


def _meta_indexer_view(group):
    cache_config = group.get("cache_config", {})
    meta_indexer = cache_config.get("meta_indexer_config", {})
    meta_storage = meta_indexer.get("meta_storage_backend_config", {})
    meta_cache = meta_indexer.get("meta_cache_policy_config", {})
    return {
        "max_key_count": _as_int(meta_indexer.get("max_key_count")),
        "mutex_shard_num": _as_int(meta_indexer.get("mutex_shard_num")),
        "batch_key_size": _as_int(meta_indexer.get("batch_key_size")),
        "meta_storage_type": meta_storage.get("storage_type"),
        "meta_storage_uri": meta_storage.get("storage_uri"),
        "search_cache_capacity": _as_int(meta_cache.get("capacity")),
        "search_cache_shard_bits": _as_int(meta_cache.get("cache_shard_bits")),
    }


def _managed_group_view(group, config):
    cache_config = group.get("cache_config", {})
    reclaim_strategy = cache_config.get("reclaim_strategy", {})
    trigger_strategy = reclaim_strategy.get("trigger_strategy", {})
    quota = group.get("quota", {})
    normalized_quota = []
    for item in quota.get("quota_config", []):
        normalized_quota.append({
            "storage_type": item.get("storage_type"),
            "capacity": _as_int(item.get("capacity")),
        })
    view = {
        "user_data": group.get("user_data"),
        "storage_candidates": group.get("storage_candidates", []),
        "event_report_storage_candidates": group.get("event_report_storage_candidates", []),
        "max_instance_count": _as_int(group.get("max_instance_count")),
        "quota": {
            "capacity": _as_int(quota.get("capacity")),
            "quota_config": normalized_quota,
        },
        "data_storage_strategy": cache_config.get("data_storage_strategy"),
        "reclaim_storage": reclaim_strategy.get("storage_unique_name"),
        "reclaim_policy": reclaim_strategy.get("reclaim_policy"),
        "reclaim_used_percentage": trigger_strategy.get("used_percentage"),
    }
    view.update(_meta_indexer_view(group))
    view["metadata_backend_mode"] = _extra_info_object(
        group.get("extra_info", "")).get("metadata_backend_mode")
    return view


def _find_group(groups, name):
    for group in groups:
        if group.get("name") == name:
            return group
    return None


def ensure_instance_group(client, config):
    desired_new = _build_new_instance_group(config)
    restart_required = False
    for attempt in range(INSTANCE_GROUP_UPDATE_RETRIES):
        current = _find_group(client.list_instance_groups(), config.instance_group_name)
        if current is None:
            try:
                client.create_instance_group(desired_new)
                LOGGER.info("created managed instance group %s", config.instance_group_name)
                return False
            except NotLeaderError:
                raise
            except BootstrapError:
                if attempt + 1 == INSTANCE_GROUP_UPDATE_RETRIES:
                    raise
                time.sleep(0.2 * (attempt + 1))
                continue

        updated = _apply_managed_group_fields(current, config)
        meta_indexer_changed = _meta_indexer_view(current) != _meta_indexer_view(updated)
        if meta_indexer_changed:
            # UpdateInstanceGroup does not reconfigure an already-created MetaIndexer.
            # Check before updating so a failed instance query cannot leave the Group
            # updated without preserving the required restart decision.
            restart_required = (
                restart_required
                or bool(client.list_instance_info(config.instance_group_name)))
        if _managed_group_view(current, config) == _managed_group_view(updated, config):
            return restart_required
        current_version = _as_int(current.get("version"))
        if not isinstance(current_version, int):
            raise BootstrapError("existing instance group version is invalid")
        updated["version"] = current_version + 1
        try:
            client.update_instance_group(updated, current_version)
            LOGGER.info("updated managed instance group %s", config.instance_group_name)
            return restart_required
        except NotLeaderError:
            raise
        except BootstrapError:
            if attempt + 1 == INSTANCE_GROUP_UPDATE_RETRIES:
                raise
            time.sleep(0.2 * (attempt + 1))

    raise BootstrapError("instance group update retries exhausted")


def verify_bootstrap(client, config):
    storage_by_name = {
        storage.get("global_unique_name"): storage
        for storage in client.list_storage()
    }
    for storage_config in config.storage_configs:
        current = storage_by_name.get(storage_config.unique_name)
        if current is None or not _storage_matches(storage_config, current):
            raise BootstrapError(
                "managed storage verification failed for {}".format(
                    storage_config.unique_name))

    current = _find_group(client.list_instance_groups(), config.instance_group_name)
    if current is None:
        raise BootstrapError("managed instance group verification failed: group is missing")
    desired = _build_new_instance_group(config)
    if _managed_group_view(current, config) != _managed_group_view(desired, config):
        raise BootstrapError("managed instance group verification failed: fields differ")


def bootstrap_once(config, client):
    if not config.enabled:
        LOGGER.info(
            "no Storage JSON is configured; set at least one Storage and "
            "KVCM_INSTANCE_GROUP, or configure KVCM manually")
        return BootstrapOutcome.COMPLETE
    health = client.check_health()
    if not health.get("is_health", False):
        raise BootstrapError("local KVCM Admin service is not healthy")
    if not health.get("is_leader", False):
        LOGGER.info("local KVCM is follower; bootstrap skipped")
        return BootstrapOutcome.FOLLOWER
    ensure_storages(client, config)
    restart_required = ensure_instance_group(client, config)
    verify_bootstrap(client, config)
    LOGGER.info(
        "bootstrap verified for group %s using meta storage %s at %s:%s",
        config.instance_group_name,
        config.meta_storage_type,
        config.redis_host,
        config.redis_port,
    )
    if restart_required:
        return BootstrapOutcome.RESTART_REQUIRED
    return BootstrapOutcome.COMPLETE


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    try:
        config = parse_environment()
        outcome = bootstrap_once(config, AdminClient())
        if outcome == BootstrapOutcome.FOLLOWER:
            return FOLLOWER_EXIT_CODE
        if outcome == BootstrapOutcome.RESTART_REQUIRED:
            return RESTART_REQUIRED_EXIT_CODE
        return 0
    except NotLeaderError:
        LOGGER.info("local KVCM lost leadership during bootstrap; retry later")
        return FOLLOWER_EXIT_CODE
    except BootstrapError as exc:
        LOGGER.exception("bootstrap failed: %s: %s", type(exc).__name__, exc)
        return 1
    except Exception as exc:
        LOGGER.exception(
            "bootstrap failed with unexpected %s: %s", type(exc).__name__, exc)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
