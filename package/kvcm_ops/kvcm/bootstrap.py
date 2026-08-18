import copy
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
)


LOGGER = logging.getLogger("kvcm_ops.bootstrap")

DEFAULT_ADMIN_URL = "http://127.0.0.1:6492"
DEFAULT_MAX_INSTANCE_COUNT = 512
DEFAULT_QUOTA_CAPACITY = 2087740652912
DEFAULT_MAX_KEY_COUNT = 1000000000
DEFAULT_MUTEX_SHARD_NUM = 131072
DEFAULT_BATCH_KEY_SIZE = 1024
DEFAULT_SEARCH_CACHE_CAPACITY = 10 * 1024
DEFAULT_SEARCH_CACHE_SHARD_BITS = 6
INSTANCE_GROUP_UPDATE_RETRIES = 3
FOLLOWER_EXIT_CODE = 2

BOOTSTRAP_QUERY_KEYS = {
    "max_instance_count",
    "quota_capacity",
    "max_key_count",
    "mutex_shard_num",
    "batch_key_size",
    "search_cache_capacity",
    "search_cache_shard_bits",
}

EVENT_REPORT_QUERY_KEYS = {
    "heartbeat_timeout_ms",
    "cleanup_grace_ms",
    "liveness_check_interval_ms",
    "snapshot_min_interval_ms",
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
    "num_shard_bits",
}


class BootstrapError(RuntimeError):
    pass


class NotLeaderError(BootstrapError):
    pass


class BootstrapConfig:
    def __init__(self):
        self.subscriber_enabled = False
        self.v6d_enabled = False
        self.instance_group_name = ""
        self.l1p5_storage_name = ""
        self.l2_storage_name = ""
        self.meta_storage_type = ""
        self.meta_storage_uri = ""
        self.metadata_backend_mode = None
        self.max_instance_count = DEFAULT_MAX_INSTANCE_COUNT
        self.quota_capacity = DEFAULT_QUOTA_CAPACITY
        self.max_key_count = DEFAULT_MAX_KEY_COUNT
        self.mutex_shard_num = DEFAULT_MUTEX_SHARD_NUM
        self.batch_key_size = DEFAULT_BATCH_KEY_SIZE
        self.search_cache_capacity = DEFAULT_SEARCH_CACHE_CAPACITY
        self.search_cache_shard_bits = DEFAULT_SEARCH_CACHE_SHARD_BITS
        self.event_report_spec = {}
        self.redis_host = ""
        self.redis_port = 0

    @property
    def enabled(self):
        return self.subscriber_enabled or self.v6d_enabled

    @property
    def primary_storage_name(self):
        if self.subscriber_enabled:
            return self.l1p5_storage_name
        return self.l2_storage_name

    @property
    def primary_storage_type(self):
        if self.subscriber_enabled:
            return "ST_EVENT_REPORT_L1P5"
        return "ST_EVENT_REPORT_L2"

    @property
    def event_report_storage_names(self):
        names = []
        if self.subscriber_enabled:
            names.append(self.l1p5_storage_name)
        if self.v6d_enabled:
            names.append(self.l2_storage_name)
        return names


def _parse_bool(environ, name):
    value = environ.get(name)
    if value is None:
        raise BootstrapError("{} is required and must be true or false".format(name))
    normalized = value.strip().lower()
    if normalized == "true":
        return True
    if normalized == "false":
        return False
    raise BootstrapError("{} must be true or false".format(name))


def _require_non_empty(environ, name):
    value = environ.get(name, "").strip()
    if not value:
        raise BootstrapError("{} is required".format(name))
    return value


def _single_query_value(query, name):
    values = query.get(name)
    if not values:
        return None
    if len(values) != 1:
        raise BootstrapError("storage URI query parameter {} must appear once".format(name))
    if values[0] == "":
        raise BootstrapError("storage URI query parameter {} must not be empty".format(name))
    return values[0]


def _positive_query_int(query, name, default, allow_zero=False):
    raw_value = _single_query_value(query, name)
    if raw_value is None:
        return default
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
        parsed = urlsplit(uri)
    except ValueError:
        raise BootstrapError("KVCM_META_STORAGE_URI is not a valid URI")
    if parsed.scheme.lower() != "redis":
        raise BootstrapError("KVCM_META_STORAGE_URI scheme must be redis")
    if not parsed.hostname:
        raise BootstrapError("KVCM_META_STORAGE_URI must include a host")
    if parsed.fragment:
        raise BootstrapError("KVCM_META_STORAGE_URI must not contain a fragment; URI-encode credentials")
    try:
        port = parsed.port
    except ValueError:
        raise BootstrapError("KVCM_META_STORAGE_URI contains an invalid port")
    if port is None or port <= 0 or port > 65535:
        raise BootstrapError("KVCM_META_STORAGE_URI must include a port in range 1..65535")
    query = parse_qs(parsed.query, keep_blank_values=True)
    cluster_name = _single_query_value(query, "cluster_name")
    if cluster_name is None:
        raise BootstrapError("KVCM_META_STORAGE_URI must include cluster_name")
    return parsed.hostname, port, query


def _validate_passthrough_query(query):
    for name in POSITIVE_PASSTHROUGH_QUERY_KEYS:
        _positive_query_int(query, name, None)
    for name in NON_NEGATIVE_PASSTHROUGH_QUERY_KEYS:
        _positive_query_int(query, name, None, allow_zero=True)
    queue_count = _positive_query_int(query, "async_queue_count", None)
    if queue_count is not None and queue_count > 2048:
        raise BootstrapError("storage URI query parameter async_queue_count must not exceed 2048")


def parse_environment(environ=None):
    environ = os.environ if environ is None else environ
    config = BootstrapConfig()
    config.subscriber_enabled = _parse_bool(environ, "KVCM_ENABLE_SUBSCRIBER_EVENT_REPORT")
    config.v6d_enabled = _parse_bool(environ, "KVCM_ENABLE_V6D_EVENT_REPORT")

    mode_value = environ.get("KVCM_METADATA_BACKEND_MODE", "").strip()
    if mode_value:
        try:
            config.metadata_backend_mode = int(mode_value)
        except ValueError:
            raise BootstrapError("KVCM_METADATA_BACKEND_MODE must be an integer in range 1..4")
        if config.metadata_backend_mode < 1 or config.metadata_backend_mode > 4:
            raise BootstrapError("KVCM_METADATA_BACKEND_MODE must be an integer in range 1..4")

    if not config.enabled:
        return config

    config.instance_group_name = _require_non_empty(environ, "KVCM_INSTANCE_GROUP_NAME")
    if config.subscriber_enabled:
        config.l1p5_storage_name = _require_non_empty(
            environ, "KVCM_EVENT_REPORT_L1P5_STORAGE_NAME")
    if config.v6d_enabled:
        config.l2_storage_name = _require_non_empty(
            environ, "KVCM_EVENT_REPORT_L2_STORAGE_NAME")

    config.meta_storage_type = _require_non_empty(environ, "KVCM_META_STORAGE_TYPE").lower()
    if config.meta_storage_type not in ("redis", "cached"):
        raise BootstrapError("KVCM_META_STORAGE_TYPE must be redis or cached")
    config.meta_storage_uri = _require_non_empty(environ, "KVCM_META_STORAGE_URI")
    config.redis_host, config.redis_port, query = _parse_storage_uri(config.meta_storage_uri)
    _validate_passthrough_query(query)

    config.max_instance_count = _positive_query_int(
        query, "max_instance_count", DEFAULT_MAX_INSTANCE_COUNT)
    config.quota_capacity = _positive_query_int(
        query, "quota_capacity", DEFAULT_QUOTA_CAPACITY)
    config.max_key_count = _positive_query_int(
        query, "max_key_count", DEFAULT_MAX_KEY_COUNT)
    config.mutex_shard_num = _positive_query_int(
        query, "mutex_shard_num", DEFAULT_MUTEX_SHARD_NUM)
    if config.mutex_shard_num & (config.mutex_shard_num - 1):
        raise BootstrapError("storage URI query parameter mutex_shard_num must be a power of two")
    config.batch_key_size = _positive_query_int(
        query, "batch_key_size", DEFAULT_BATCH_KEY_SIZE)
    config.search_cache_capacity = _positive_query_int(
        query, "search_cache_capacity", DEFAULT_SEARCH_CACHE_CAPACITY)
    config.search_cache_shard_bits = _positive_query_int(
        query, "search_cache_shard_bits", DEFAULT_SEARCH_CACHE_SHARD_BITS, allow_zero=True)

    for name in EVENT_REPORT_QUERY_KEYS:
        raw_value = _single_query_value(query, name)
        if raw_value is None:
            continue
        config.event_report_spec[name] = _positive_query_int(query, name, None)
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
            raise NotLeaderError("Admin API {} rejected because this node is not leader".format(operation))
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


def _storage_type_for_name(config, name):
    if name == config.l1p5_storage_name:
        return "ST_EVENT_REPORT_L1P5"
    return "ST_EVENT_REPORT_L2"


def _new_storage(config, name, storage_type, existing=None):
    event_report = {}
    if existing is not None and isinstance(existing.get("event_report"), dict):
        event_report.update(existing["event_report"])
    event_report.update(config.event_report_spec)
    return {
        "global_unique_name": name,
        "storage_type": storage_type,
        "event_report": event_report,
        "check_storage_available_when_open": True,
    }


def _storage_matches(config, current, storage_type):
    if current.get("storage_type") != storage_type:
        return False
    current_spec = current.get("event_report", {})
    for key, value in config.event_report_spec.items():
        if _as_int(current_spec.get(key, -1)) != value:
            return False
    return True


def ensure_storages(client, config):
    current_by_name = {
        storage.get("global_unique_name"): storage
        for storage in client.list_storage()
    }
    for name in config.event_report_storage_names:
        storage_type = _storage_type_for_name(config, name)
        current = current_by_name.get(name)
        if current is None:
            client.add_storage(_new_storage(config, name, storage_type))
            LOGGER.info("created managed storage %s", name)
        elif not _storage_matches(config, current, storage_type):
            client.update_storage(_new_storage(config, name, storage_type, current))
            LOGGER.info("updated managed storage %s", name)


def _build_new_instance_group(config):
    quota = InstanceGroupQuota(
        config.quota_capacity,
        [StorageQuota(config.primary_storage_type, config.quota_capacity)],
    )
    reclaim_strategy = ReclaimStrategy(storage_unique_name=config.primary_storage_name)
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
        reclaim_strategy=reclaim_strategy,
        meta_indexer_config=meta_indexer,
    )
    extra_info = ""
    if config.metadata_backend_mode is not None:
        extra_info = json.dumps(
            {"metadata_backend_mode": config.metadata_backend_mode},
            sort_keys=True,
        )
    return InstanceGroup(
        name=config.instance_group_name,
        storage_candidates=[config.primary_storage_name],
        instance_group_quota=quota,
        quota_group_name="default_quota_group",
        max_instance_count=config.max_instance_count,
        cache_config=cache_config,
        user_data=config.instance_group_name,
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
    reclaim_strategy = cache_config.setdefault("reclaim_strategy", {})
    reclaim_strategy["storage_unique_name"] = config.primary_storage_name
    meta_indexer = cache_config.setdefault("meta_indexer_config", {})
    meta_indexer["max_key_count"] = config.max_key_count
    meta_indexer["mutex_shard_num"] = config.mutex_shard_num
    meta_indexer["batch_key_size"] = config.batch_key_size
    meta_indexer["meta_storage_backend_config"] = {
        "storage_type": config.meta_storage_type,
        "storage_uri": config.meta_storage_uri,
    }
    if config.metadata_backend_mode is not None:
        extra_info = _extra_info_object(updated.get("extra_info", ""))
        extra_info["metadata_backend_mode"] = config.metadata_backend_mode
        updated["extra_info"] = json.dumps(extra_info, sort_keys=True, ensure_ascii=False)
    return updated


def _managed_group_view(group, config):
    cache_config = group.get("cache_config", {})
    reclaim_strategy = cache_config.get("reclaim_strategy", {})
    meta_indexer = cache_config.get("meta_indexer_config", {})
    meta_storage = meta_indexer.get("meta_storage_backend_config", {})
    quota = group.get("quota", {})
    quota_config = quota.get("quota_config", [])
    normalized_quota = []
    for item in quota_config:
        normalized_quota.append({
            "storage_type": item.get("storage_type"),
            "capacity": _as_int(item.get("capacity")),
        })
    view = {
        "storage_candidates": group.get("storage_candidates", []),
        "event_report_storage_candidates": group.get("event_report_storage_candidates", []),
        "max_instance_count": _as_int(group.get("max_instance_count")),
        "quota": {
            "capacity": _as_int(quota.get("capacity")),
            "quota_config": normalized_quota,
        },
        "reclaim_storage": reclaim_strategy.get("storage_unique_name"),
        "max_key_count": _as_int(meta_indexer.get("max_key_count")),
        "mutex_shard_num": _as_int(meta_indexer.get("mutex_shard_num")),
        "batch_key_size": _as_int(meta_indexer.get("batch_key_size")),
        "meta_storage_type": meta_storage.get("storage_type"),
        "meta_storage_uri": meta_storage.get("storage_uri"),
    }
    if config.metadata_backend_mode is not None:
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
    for attempt in range(INSTANCE_GROUP_UPDATE_RETRIES):
        current = _find_group(client.list_instance_groups(), config.instance_group_name)
        if current is None:
            try:
                client.create_instance_group(desired_new)
                LOGGER.info("created managed instance group %s", config.instance_group_name)
                return
            except NotLeaderError:
                raise
            except BootstrapError:
                if attempt + 1 == INSTANCE_GROUP_UPDATE_RETRIES:
                    raise
                time.sleep(0.2 * (attempt + 1))
                continue

        updated = _apply_managed_group_fields(current, config)
        if _managed_group_view(current, config) == _managed_group_view(updated, config):
            return
        current_version = _as_int(current.get("version"))
        if not isinstance(current_version, int):
            raise BootstrapError("existing instance group version is invalid")
        updated["version"] = current_version + 1
        try:
            client.update_instance_group(updated, current_version)
            LOGGER.info("updated managed instance group %s", config.instance_group_name)
            return
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
    for name in config.event_report_storage_names:
        storage_type = _storage_type_for_name(config, name)
        current = storage_by_name.get(name)
        if current is None or not _storage_matches(config, current, storage_type):
            raise BootstrapError("managed storage verification failed for {}".format(name))

    current = _find_group(client.list_instance_groups(), config.instance_group_name)
    if current is None:
        raise BootstrapError("managed instance group verification failed: group is missing")
    desired = _build_new_instance_group(config)
    if _managed_group_view(current, config) != _managed_group_view(desired, config):
        raise BootstrapError("managed instance group verification failed: fields differ")


def bootstrap_once(config, client):
    if not config.enabled:
        LOGGER.info("event reporting is disabled; bootstrap skipped")
        return True
    health = client.check_health()
    if not health.get("is_health", False):
        raise BootstrapError("local KVCM Admin service is not healthy")
    if not health.get("is_leader", False):
        LOGGER.info("local KVCM is follower; bootstrap skipped")
        return False
    ensure_storages(client, config)
    ensure_instance_group(client, config)
    verify_bootstrap(client, config)
    LOGGER.info(
        "bootstrap verified for group %s using meta storage %s at %s:%s",
        config.instance_group_name,
        config.meta_storage_type,
        config.redis_host,
        config.redis_port,
    )
    return True


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    try:
        config = parse_environment()
        if not bootstrap_once(config, AdminClient()):
            return FOLLOWER_EXIT_CODE
        return 0
    except NotLeaderError:
        LOGGER.info("local KVCM lost leadership during bootstrap; retry later")
        return FOLLOWER_EXIT_CODE
    except BootstrapError as exc:
        LOGGER.exception("bootstrap failed: %s: %s", type(exc).__name__, exc)
        return 1
    except Exception as exc:
        LOGGER.exception(
            "bootstrap failed with unexpected %s: %s",
            type(exc).__name__,
            exc,
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
