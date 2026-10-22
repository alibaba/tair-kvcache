import copy
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch
from urllib.parse import parse_qs, urlsplit


sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from kvcm_ops.kvcm.bootstrap import (  # noqa: E402
    AdminClient,
    BootstrapError,
    BootstrapOutcome,
    _apply_managed_group_fields,
    _build_new_instance_group,
    _managed_group_view,
    bootstrap_once,
    ensure_instance_group,
    main,
    parse_environment,
)


GROUP_NAME = "event-group"
L1P5_NAME = "event-l1p5"
L2_NAME = "event-l2"
PACE_NAME = "pace-l1"
META_STORAGE_URI = (
    "redis://user:p%40ss@redis.example:6379/?cluster_name=test"
    "&async_queue_count=8")
LOCAL_STORAGE_URI = "local://?capacity=65536&num_shard_bits=15&sample_times=1024"


def storage_env(unique_name):
    return json.dumps({"unique_name": unique_name})


def group_env(**overrides):
    value = {
        "name": GROUP_NAME,
        "meta_storage_backend_config": "cached,{}".format(META_STORAGE_URI),
    }
    value.update(overrides)
    return json.dumps(value)


BASE_ENV = {
    "KVCM_L1P5_STORAGE": storage_env(L1P5_NAME),
    "KVCM_L2P_STORAGE": storage_env(L2_NAME),
    "KVCM_INSTANCE_GROUP": group_env(),
}


class FakeAdminClient:
    def __init__(self, is_leader=True):
        self.is_leader = is_leader
        self.storages = []
        self.groups = []
        self.instances = []
        self.add_calls = 0
        self.update_storage_calls = 0
        self.create_group_calls = 0
        self.update_group_calls = 0
        self.list_instance_calls = 0
        self.list_instance_error = None
        self.version_conflicts_remaining = 0
        self.create_group_failures_remaining = 0

    def check_health(self):
        return {"is_health": True, "is_leader": self.is_leader}

    def list_storage(self):
        return copy.deepcopy(self.storages)

    def add_storage(self, storage):
        self.add_calls += 1
        self.storages.append(copy.deepcopy(storage))

    def update_storage(self, storage):
        self.update_storage_calls += 1
        self.storages = [
            copy.deepcopy(storage)
            if item["global_unique_name"] == storage["global_unique_name"] else item
            for item in self.storages
        ]

    def list_instance_groups(self):
        return copy.deepcopy(self.groups)

    def list_instance_info(self, instance_group_name):
        self.list_instance_calls += 1
        if self.list_instance_error is not None:
            raise self.list_instance_error
        return copy.deepcopy(self.instances)

    def create_instance_group(self, instance_group):
        self.create_group_calls += 1
        if self.create_group_failures_remaining > 0:
            self.create_group_failures_remaining -= 1
            raise BootstrapError("temporary create failure")
        self.groups.append(copy.deepcopy(instance_group))

    def update_instance_group(self, instance_group, current_version):
        self.update_group_calls += 1
        if self.version_conflicts_remaining > 0:
            self.version_conflicts_remaining -= 1
            raise BootstrapError("version conflict")
        for index, current in enumerate(self.groups):
            if current["name"] == instance_group["name"]:
                if int(current["version"]) != current_version:
                    raise BootstrapError("version conflict")
                self.groups[index] = copy.deepcopy(instance_group)
                return
        raise BootstrapError("group missing")


class EnvironmentParsingTest(unittest.TestCase):
    def test_all_storage_types_and_primary_priority(self):
        environ = dict(BASE_ENV)
        environ["KVCM_PACE_STORAGE"] = json.dumps({
            "unique_name": PACE_NAME,
            "domain": "http://pace.example",
            "timeout": 30,
            "service_discovery_url": "http://discovery.example",
            "media_type": 2,
        })

        config = parse_environment(environ)

        self.assertEqual(PACE_NAME, config.primary_storage_name)
        self.assertEqual("ST_TAIRMEMPOOL", config.primary_storage_type)
        self.assertEqual([L1P5_NAME, L2_NAME], config.event_report_storage_names)
        self.assertEqual("CPS_PREFER_TAIR_MEMPOOL", config.data_storage_strategy)

    def test_l1p5_minimal_environment_uses_defaults(self):
        config = parse_environment({
            "KVCM_L1P5_STORAGE": storage_env(L1P5_NAME),
            "KVCM_INSTANCE_GROUP": group_env(),
        })

        self.assertEqual(1000000000, config.quota_capacity)
        self.assertEqual(512, config.max_instance_count)
        self.assertEqual("POLICY_LRU", config.reclaim_policy)
        self.assertEqual(0.8, config.reclaim_used_percentage)
        self.assertEqual(30000, config.l1p5_storage.spec["heartbeat_timeout_ms"])
        self.assertEqual(GROUP_NAME, config.user_data)

    def test_empty_storage_environment_skips_group_requirement(self):
        for environ in ({}, {"KVCM_L1P5_STORAGE": "  "}):
            with self.subTest(environ=environ):
                config = parse_environment(environ)
                self.assertFalse(config.enabled)

    def test_eight_storage_combinations(self):
        for l1p5_enabled in (False, True):
            for l2_enabled in (False, True):
                for pace_enabled in (False, True):
                    environ = {}
                    if l1p5_enabled:
                        environ["KVCM_L1P5_STORAGE"] = storage_env(L1P5_NAME)
                    if l2_enabled:
                        environ["KVCM_L2P_STORAGE"] = storage_env(L2_NAME)
                    if pace_enabled:
                        environ["KVCM_PACE_STORAGE"] = json.dumps({
                            "unique_name": PACE_NAME,
                            "domain": "http://pace.example",
                            "timeout": 30,
                        })
                    if environ:
                        environ["KVCM_INSTANCE_GROUP"] = group_env()
                    config = parse_environment(environ)
                    self.assertEqual(
                        int(l1p5_enabled) + int(l2_enabled) + int(pace_enabled),
                        len(config.storage_configs),
                    )
                    if pace_enabled:
                        self.assertEqual(PACE_NAME, config.primary_storage_name)
                    elif l1p5_enabled:
                        self.assertEqual(L1P5_NAME, config.primary_storage_name)
                    elif l2_enabled:
                        self.assertEqual(L2_NAME, config.primary_storage_name)

    def test_any_storage_requires_instance_group(self):
        with self.assertRaisesRegex(BootstrapError, "KVCM_INSTANCE_GROUP is required"):
            parse_environment({"KVCM_L1P5_STORAGE": storage_env(L1P5_NAME)})

    def test_unknown_fields_and_wrong_types_are_rejected(self):
        cases = (
            ({"KVCM_L1P5_STORAGE": json.dumps({
                "unique_name": L1P5_NAME, "unknown": 1}),
              "KVCM_INSTANCE_GROUP": group_env()}, "unknown fields"),
            ({"KVCM_L1P5_STORAGE": json.dumps({"unique_name": 1}),
              "KVCM_INSTANCE_GROUP": group_env()}, "non-empty string"),
            ({"KVCM_L1P5_STORAGE": json.dumps({
                "unique_name": L1P5_NAME, "heartbeat_timeout_ms": True}),
              "KVCM_INSTANCE_GROUP": group_env()}, "must be an integer"),
            ({"KVCM_L1P5_STORAGE": storage_env(L1P5_NAME),
              "KVCM_INSTANCE_GROUP": group_env(quota_capacity=0)}, "must be positive"),
        )
        for environ, message in cases:
            with self.subTest(message=message):
                with self.assertRaisesRegex(BootstrapError, message):
                    parse_environment(environ)

    def test_duplicate_storage_names_are_rejected(self):
        environ = dict(BASE_ENV)
        environ["KVCM_L2P_STORAGE"] = storage_env(L1P5_NAME)
        with self.assertRaisesRegex(BootstrapError, "must be distinct"):
            parse_environment(environ)

    def test_pace_requires_positive_timeout(self):
        for pace_value in (
                {"unique_name": PACE_NAME, "domain": "http://pace.example"},
                {"unique_name": PACE_NAME, "domain": "http://pace.example", "timeout": 0}):
            with self.subTest(pace_value=pace_value):
                with self.assertRaisesRegex(BootstrapError, "timeout.*positive"):
                    parse_environment({
                        "KVCM_PACE_STORAGE": json.dumps(pace_value),
                        "KVCM_INSTANCE_GROUP": group_env(),
                    })

    def test_new_pace_defaults_media_type_to_zero(self):
        config = parse_environment({
            "KVCM_PACE_STORAGE": json.dumps({
                "unique_name": PACE_NAME,
                "domain": "http://pace.example",
                "timeout": 30,
            }),
            "KVCM_INSTANCE_GROUP": group_env(),
        })
        self.assertEqual(0, config.pace_storage.spec["media_type"])

    def test_group_options_are_read_from_group_json_not_redis_uri(self):
        environ = dict(BASE_ENV)
        environ["KVCM_INSTANCE_GROUP"] = group_env(
            quota_capacity=123,
            max_instance_count=7,
            max_key_count=456,
            mutex_shard_num=8,
            batch_key_size=9,
            search_cache_capacity=10,
            search_cache_shard_bits=3,
            metadata_backend_mode=4,
            user_data="custom",
        )

        config = parse_environment(environ)

        self.assertEqual(123, config.quota_capacity)
        self.assertEqual(7, config.max_instance_count)
        self.assertEqual(456, config.max_key_count)
        self.assertEqual(8, config.mutex_shard_num)
        self.assertEqual(4, config.metadata_backend_mode)
        self.assertEqual("custom", config.user_data)

    def test_cached_uri_fills_missing_backend_defaults(self):
        uri = "redis://redis.example:6379/?cluster_name=test"
        environ = dict(BASE_ENV)
        environ["KVCM_INSTANCE_GROUP"] = group_env(
            meta_storage_backend_config="cached,{}".format(uri))

        config = parse_environment(environ)
        group = _build_new_instance_group(config)
        query = parse_qs(urlsplit(config.meta_storage_uri).query)

        self.assertEqual({
            "timeout_ms": ["1000"],
            "retry_count": ["3"],
            "num_shard_bits": ["15"],
            "sample_times": ["1024"],
            "persistent_type": ["async_redis"],
            "cache_type": ["local"],
            "async_queue_count": ["32"],
            "async_max_batch": ["204800"],
            "async_max_size": ["204800"],
            "async_enqueue_timeout_ms": ["10"],
            "async_wait_us": ["1000000"],
        }, {key: query[key] for key in (
            "timeout_ms", "retry_count", "num_shard_bits", "sample_times",
            "persistent_type", "cache_type", "async_queue_count", "async_max_batch",
            "async_max_size", "async_enqueue_timeout_ms", "async_wait_us")})
        self.assertEqual(
            config.meta_storage_uri,
            group["cache_config"]["meta_indexer_config"]
            ["meta_storage_backend_config"]["storage_uri"],
        )

    def test_explicit_uri_values_override_bootstrap_defaults(self):
        uri = (
            "redis://redis.example:6379/?cluster_name=test"
            "&timeout_ms=2500&retry_count=4&num_shard_bits=8"
            "&persistent_type=redis&cache_type=local")
        environ = dict(BASE_ENV)
        environ["KVCM_INSTANCE_GROUP"] = group_env(
            meta_storage_backend_config="cached,{}".format(uri))

        config = parse_environment(environ)
        query = parse_qs(urlsplit(config.meta_storage_uri).query)

        self.assertTrue(config.meta_storage_uri.startswith(uri + "&"))
        self.assertEqual(["2500"], query["timeout_ms"])
        self.assertEqual(["4"], query["retry_count"])
        self.assertEqual(["8"], query["num_shard_bits"])
        self.assertEqual(["redis"], query["persistent_type"])
        self.assertNotIn("async_queue_count", query)
        self.assertEqual(["1024"], query["sample_times"])

    def test_redis_backend_only_gets_shared_defaults(self):
        uri = "redis://redis.example:6379/?cluster_name=test"
        environ = dict(BASE_ENV)
        environ["KVCM_INSTANCE_GROUP"] = group_env(
            meta_storage_backend_config="redis,{}".format(uri))

        config = parse_environment(environ)
        query = parse_qs(urlsplit(config.meta_storage_uri).query)

        self.assertEqual(["1000"], query["timeout_ms"])
        self.assertEqual(["3"], query["retry_count"])
        self.assertNotIn("persistent_type", query)
        self.assertNotIn("async_queue_count", query)

    def test_local_uri_preserves_explicit_parameters_without_redis_defaults(self):
        environ = dict(BASE_ENV)
        environ["KVCM_INSTANCE_GROUP"] = group_env(
            meta_storage_backend_config="local,{}".format(LOCAL_STORAGE_URI))

        config = parse_environment(environ)
        backend = _build_new_instance_group(config)["cache_config"]["meta_indexer_config"][
            "meta_storage_backend_config"]

        self.assertEqual("local", config.meta_storage_type)
        self.assertEqual(LOCAL_STORAGE_URI, config.meta_storage_uri)
        self.assertEqual({"storage_type": "local", "storage_uri": LOCAL_STORAGE_URI},
                         {key: backend[key] for key in ("storage_type", "storage_uri")})
        self.assertEqual("", config.redis_host)
        self.assertEqual(0, config.redis_port)

    def test_bare_local_uses_backend_defaults(self):
        environ = dict(BASE_ENV)
        environ["KVCM_INSTANCE_GROUP"] = group_env(meta_storage_backend_config="local")

        config = parse_environment(environ)

        self.assertEqual("local", config.meta_storage_type)
        self.assertEqual("", config.meta_storage_uri)

    def test_local_uri_rejects_invalid_parameters(self):
        invalid = (
            ("capacity=0", "capacity.*positive"),
            ("capacity=17592186044416", "capacity.*too large"),
            ("capacity=-1", "capacity.*integer"),
            ("capacity=abc", "capacity.*integer"),
            ("capacity=%31", "capacity.*integer"),
            ("capacity=+1", "capacity.*integer"),
            ("capacity=1&capacity=2", "capacity.*once"),
            ("num_shard_bits=-1", "num_shard_bits.*integer"),
            ("num_shard_bits=20", "num_shard_bits.*less than 20"),
            ("sample_times=0", "sample_times.*positive"),
            ("sample_times=18446744073709551616", "sample_times.*too large"),
            ("sample_times=abc", "sample_times.*integer"),
        )
        for query, message in invalid:
            with self.subTest(query=query):
                environ = dict(BASE_ENV)
                environ["KVCM_INSTANCE_GROUP"] = group_env(
                    meta_storage_backend_config="local,local://?{}".format(query))
                with self.assertRaisesRegex(BootstrapError, message):
                    parse_environment(environ)

    def test_local_uri_rejects_wrong_scheme_or_address(self):
        for uri in ("redis://?capacity=1", "local://host?capacity=1",
                    "local:///path?capacity=1", "local://?capacity=1#"):
            with self.subTest(uri=uri):
                environ = dict(BASE_ENV)
                environ["KVCM_INSTANCE_GROUP"] = group_env(
                    meta_storage_backend_config="local,{}".format(uri))
                with self.assertRaisesRegex(BootstrapError, "local URI"):
                    parse_environment(environ)

    def test_raw_hash_in_password_is_supported_and_preserved(self):
        raw_uri = "redis://user:p#ss@redis.example:6379/?cluster_name=test"
        environ = dict(BASE_ENV)
        environ["KVCM_INSTANCE_GROUP"] = group_env(
            meta_storage_backend_config="cached,{}".format(raw_uri))

        config = parse_environment(environ)
        group = _build_new_instance_group(config)

        self.assertEqual("redis.example", config.redis_host)
        self.assertTrue(config.meta_storage_uri.startswith(raw_uri + "&"))
        self.assertIn("async_queue_count=32", config.meta_storage_uri)
        self.assertEqual(
            config.meta_storage_uri,
            group["cache_config"]["meta_indexer_config"]
            ["meta_storage_backend_config"]["storage_uri"],
        )

    def test_uri_error_does_not_expose_credentials(self):
        secret = "do-not-log-this"
        environ = dict(BASE_ENV)
        environ["KVCM_INSTANCE_GROUP"] = group_env(
            meta_storage_backend_config=(
                "cached,redis://user:{}@redis.example/".format(secret)))
        with self.assertRaises(BootstrapError) as context:
            parse_environment(environ)
        self.assertNotIn(secret, str(context.exception))


class ErrorReportingTest(unittest.TestCase):
    def test_invalid_meta_storage_config_does_not_log_credentials(self):
        secret = "synthetic-password-for-log-check"
        environ = dict(BASE_ENV)
        environ["KVCM_INSTANCE_GROUP"] = group_env(
            meta_storage_backend_config=(
                "cached,redis://user:{}@redis.example:6379/?cluster_name=test,true,invalid"
                .format(secret)))
        with patch.dict("os.environ", environ, clear=True), \
             self.assertLogs("kvcm_ops.bootstrap", level="ERROR") as logs:
            self.assertEqual(1, main([]))
        full_log = "\n".join(logs.output)
        self.assertIn("meta_storage_backend_config must use", full_log)
        self.assertNotIn(secret, full_log)
        self.assertNotIn("ArgumentTypeError", full_log)

    def test_bootstrap_command_uses_custom_admin_host(self):
        with patch("kvcm_ops.kvcm.bootstrap.parse_environment",
                   return_value=parse_environment({})), \
             patch("kvcm_ops.kvcm.bootstrap.AdminClient") as admin_client, \
             patch("kvcm_ops.kvcm.bootstrap.bootstrap_once",
                   return_value=BootstrapOutcome.COMPLETE):
            self.assertEqual(0, main(["--host", "http://127.0.0.1:6500"]))
        admin_client.assert_called_once_with("http://127.0.0.1:6500")

    @patch("kvcm_ops.kvcm.bootstrap.http_post")
    def test_admin_transport_error_keeps_exception_details(self, mock_http_post):
        mock_http_post.side_effect = RuntimeError("connection refused")
        with self.assertRaises(BootstrapError) as context:
            AdminClient().post("/api/listStorage", {}, "listStorage")
        self.assertIn("listStorage", str(context.exception))
        self.assertIn("connection refused", str(context.exception))

    @patch("kvcm_ops.kvcm.bootstrap.http_post")
    def test_list_instance_info_uses_group_name(self, mock_http_post):
        mock_http_post.return_value = {
            "header": {"status": {"code": "OK"}},
            "instance_info": [{"instance_id": "i-1"}],
        }
        result = AdminClient().list_instance_info(GROUP_NAME)
        self.assertEqual([{"instance_id": "i-1"}], result)
        request_data = mock_http_post.call_args[0][2]
        self.assertEqual(
            GROUP_NAME,
            request_data["instance_group_name"],
        )


class DesiredConfigurationTest(unittest.TestCase):
    def test_pace_primary_drives_quota_reclaim_and_strategy(self):
        environ = dict(BASE_ENV)
        environ["KVCM_PACE_STORAGE"] = json.dumps({
            "unique_name": PACE_NAME,
            "domain": "http://pace.example",
            "timeout": 30,
        })
        group = _build_new_instance_group(parse_environment(environ))

        self.assertEqual([PACE_NAME], group["storage_candidates"])
        self.assertEqual([L1P5_NAME, L2_NAME], group["event_report_storage_candidates"])
        self.assertEqual("ST_TAIRMEMPOOL", group["quota"]["quota_config"][0]["storage_type"])
        self.assertEqual(PACE_NAME, group["cache_config"]["reclaim_strategy"]["storage_unique_name"])
        self.assertEqual("CPS_PREFER_TAIR_MEMPOOL", group["cache_config"]["data_storage_strategy"])

    def test_explicit_meta_backend_flags_reach_created_and_reconciled_group(self):
        environ = dict(BASE_ENV)
        environ["KVCM_INSTANCE_GROUP"] = group_env(
            meta_storage_backend_config=(
                "cached,redis://redis.example:6379/?cluster_name=test,true,false"))
        config = parse_environment(environ)

        created = _build_new_instance_group(config)
        created_backend = created["cache_config"]["meta_indexer_config"][
            "meta_storage_backend_config"]
        self.assertTrue(created_backend["memory_primary"])
        self.assertFalse(created_backend["force_deleting_async_enqueue"])

        created_backend["memory_primary"] = False
        created_backend["force_deleting_async_enqueue"] = True
        reconciled = _apply_managed_group_fields(created, config)
        reconciled_backend = reconciled["cache_config"]["meta_indexer_config"][
            "meta_storage_backend_config"]
        self.assertTrue(reconciled_backend["memory_primary"])
        self.assertFalse(reconciled_backend["force_deleting_async_enqueue"])

    def test_update_manages_configured_fields_and_preserves_other_fields(self):
        config = parse_environment(dict(BASE_ENV))
        current = _build_new_instance_group(config)
        current["revisit_interval_buckets"] = "1,5,30"
        current["extra_info"] = '{"keep":"value","metadata_backend_mode":2}'
        config.metadata_backend_mode = 4
        config.user_data = "new-user-data"

        updated = _apply_managed_group_fields(current, config)

        self.assertEqual("new-user-data", updated["user_data"])
        self.assertEqual("1,5,30", updated["revisit_interval_buckets"])
        self.assertEqual(
            {"keep": "value", "metadata_backend_mode": 4},
            json.loads(updated["extra_info"]),
        )

    def test_unset_metadata_mode_removes_only_managed_key(self):
        config = parse_environment(dict(BASE_ENV))
        current = _build_new_instance_group(config)
        current["extra_info"] = '{"metadata_backend_mode":3,"keep":true}'
        updated = _apply_managed_group_fields(current, config)
        self.assertEqual({"keep": True}, json.loads(updated["extra_info"]))


class BootstrapReconciliationTest(unittest.TestCase):
    def test_first_run_creates_group_with_local_meta_backend(self):
        environ = dict(BASE_ENV)
        environ["KVCM_INSTANCE_GROUP"] = group_env(
            meta_storage_backend_config="local,{}".format(LOCAL_STORAGE_URI))
        client = FakeAdminClient()

        self.assertEqual(BootstrapOutcome.COMPLETE,
                         bootstrap_once(parse_environment(environ), client))

        self.assertEqual(1, client.create_group_calls)
        self.assertEqual(2, client.add_calls)
        backend = client.groups[0]["cache_config"]["meta_indexer_config"][
            "meta_storage_backend_config"]
        self.assertEqual("local", backend["storage_type"])
        self.assertEqual(LOCAL_STORAGE_URI, backend["storage_uri"])

    def test_first_run_creates_and_second_run_is_idempotent(self):
        config = parse_environment(dict(BASE_ENV))
        client = FakeAdminClient()

        self.assertEqual(BootstrapOutcome.COMPLETE, bootstrap_once(config, client))
        self.assertEqual(BootstrapOutcome.COMPLETE, bootstrap_once(config, client))

        self.assertEqual(2, client.add_calls)
        self.assertEqual(0, client.update_storage_calls)
        self.assertEqual(1, client.create_group_calls)
        self.assertEqual(0, client.update_group_calls)

    def test_manual_changes_survive_regular_restart(self):
        config = parse_environment(dict(BASE_ENV))
        client = FakeAdminClient()
        bootstrap_once(config, client)
        client.groups[0]["quota"]["capacity"] = 200
        client.storages[0]["event_report"]["heartbeat_timeout_ms"] = 1000

        self.assertEqual(BootstrapOutcome.COMPLETE, bootstrap_once(config, client))
        self.assertEqual(200, client.groups[0]["quota"]["capacity"])
        self.assertEqual(1000, client.storages[0]["event_report"]["heartbeat_timeout_ms"])
        self.assertEqual(0, client.update_group_calls)
        self.assertEqual(0, client.update_storage_calls)

    def test_manually_created_group_is_not_adopted(self):
        config = parse_environment(dict(BASE_ENV))
        client = FakeAdminClient()
        manual_group = _build_new_instance_group(config)
        manual_group["quota"]["capacity"] = 200
        client.groups.append(manual_group)

        self.assertEqual(BootstrapOutcome.COMPLETE, bootstrap_once(config, client))
        self.assertEqual(200, client.groups[0]["quota"]["capacity"])
        self.assertEqual([], client.storages)
        self.assertEqual(0, client.create_group_calls)
        self.assertEqual(0, client.add_calls)

    def test_initialization_rejects_conflicting_storage_before_creating_others(self):
        config = parse_environment(dict(BASE_ENV))
        client = FakeAdminClient()
        client.storages.append({
            "global_unique_name": L1P5_NAME,
            "storage_type": "ST_EVENT_REPORT_L1P5",
            "event_report": {"heartbeat_timeout_ms": 1},
        })

        with self.assertRaisesRegex(BootstrapError, "use --reconcile"):
            bootstrap_once(config, client)
        self.assertEqual(0, client.add_calls)
        self.assertEqual(0, client.create_group_calls)

    def test_group_creation_retry_reuses_created_storage(self):
        config = parse_environment(dict(BASE_ENV))
        client = FakeAdminClient()
        client.create_group_failures_remaining = 1

        self.assertEqual(BootstrapOutcome.COMPLETE, bootstrap_once(config, client))
        self.assertEqual(2, client.create_group_calls)
        self.assertEqual(2, client.add_calls)
        self.assertEqual(1, len(client.groups))

    def test_no_storage_skips_admin_api(self):
        config = parse_environment({})
        client = FakeAdminClient(is_leader=False)
        self.assertEqual(BootstrapOutcome.COMPLETE, bootstrap_once(config, client))
        self.assertEqual([], client.storages)

    def test_follower_does_not_write(self):
        config = parse_environment(dict(BASE_ENV))
        client = FakeAdminClient(is_leader=False)
        self.assertEqual(BootstrapOutcome.FOLLOWER, bootstrap_once(config, client))
        self.assertEqual([], client.storages)
        self.assertEqual([], client.groups)

    def test_same_storage_name_with_changed_config_updates_storage(self):
        config = parse_environment(dict(BASE_ENV))
        client = FakeAdminClient()
        bootstrap_once(config, client)
        client.storages[0]["event_report"]["heartbeat_timeout_ms"] = 1

        self.assertEqual(BootstrapOutcome.COMPLETE, bootstrap_once(config, client, reconcile=True))
        self.assertEqual(1, client.update_storage_calls)
        self.assertEqual(30000, client.storages[0]["event_report"]["heartbeat_timeout_ms"])

    def test_storage_name_change_adds_new_storage_and_updates_group(self):
        client = FakeAdminClient()
        bootstrap_once(parse_environment(dict(BASE_ENV)), client)
        environ = dict(BASE_ENV)
        environ["KVCM_L1P5_STORAGE"] = storage_env("event-l1p5-new")

        outcome = bootstrap_once(parse_environment(environ), client, reconcile=True)

        self.assertEqual(BootstrapOutcome.COMPLETE, outcome)
        self.assertEqual({L1P5_NAME, L2_NAME, "event-l1p5-new"}, {
            item["global_unique_name"] for item in client.storages})
        self.assertEqual(["event-l1p5-new"], client.groups[0]["storage_candidates"])
        self.assertEqual(
            ["event-l1p5-new", L2_NAME],
            client.groups[0]["event_report_storage_candidates"],
        )

    def test_l1p5_to_l2_keeps_old_storage_and_updates_group(self):
        first_env = {
            "KVCM_L1P5_STORAGE": storage_env(L1P5_NAME),
            "KVCM_INSTANCE_GROUP": group_env(),
        }
        second_env = {
            "KVCM_L2P_STORAGE": storage_env(L2_NAME),
            "KVCM_INSTANCE_GROUP": group_env(),
        }
        client = FakeAdminClient()
        bootstrap_once(parse_environment(first_env), client)

        outcome = bootstrap_once(parse_environment(second_env), client, reconcile=True)

        self.assertEqual(BootstrapOutcome.COMPLETE, outcome)
        self.assertEqual({L1P5_NAME, L2_NAME}, {
            item["global_unique_name"] for item in client.storages})
        self.assertEqual([L2_NAME], client.groups[0]["storage_candidates"])
        self.assertEqual([L2_NAME], client.groups[0]["event_report_storage_candidates"])

    def test_meta_change_without_instance_updates_without_restart(self):
        client = FakeAdminClient()
        bootstrap_once(parse_environment(dict(BASE_ENV)), client)
        environ = dict(BASE_ENV)
        environ["KVCM_INSTANCE_GROUP"] = group_env(max_key_count=123)

        outcome = bootstrap_once(parse_environment(environ), client, reconcile=True)

        self.assertEqual(BootstrapOutcome.COMPLETE, outcome)
        self.assertEqual(1, client.list_instance_calls)
        self.assertEqual(123, client.groups[0]["cache_config"]["meta_indexer_config"]["max_key_count"])

    def test_meta_change_with_instance_updates_and_requires_one_restart(self):
        client = FakeAdminClient()
        bootstrap_once(parse_environment(dict(BASE_ENV)), client)
        client.instances = [{"instance_id": "i-1"}]
        environ = dict(BASE_ENV)
        environ["KVCM_INSTANCE_GROUP"] = group_env(max_key_count=123)
        config = parse_environment(environ)

        self.assertEqual(BootstrapOutcome.RESTART_REQUIRED, bootstrap_once(config, client, reconcile=True))
        self.assertEqual(BootstrapOutcome.COMPLETE, bootstrap_once(config, client, reconcile=True))
        self.assertEqual(1, client.update_group_calls)
        self.assertEqual(1, client.list_instance_calls)

    def test_reconcile_redis_to_local_updates_group_and_requires_restart(self):
        client = FakeAdminClient()
        bootstrap_once(parse_environment(dict(BASE_ENV)), client)
        client.instances = [{"instance_id": "i-1"}]
        previous_version = client.groups[0]["version"]
        environ = dict(BASE_ENV)
        environ["KVCM_L2P_STORAGE"] = storage_env("event-l2-new")
        environ["KVCM_INSTANCE_GROUP"] = group_env(
            meta_storage_backend_config="local,{}".format(LOCAL_STORAGE_URI))
        config = parse_environment(environ)

        with patch("kvcm_ops.kvcm.bootstrap.parse_environment", return_value=config), \
             patch("kvcm_ops.kvcm.bootstrap.AdminClient", return_value=client):
            status = main(["--reconcile"])

        self.assertEqual(3, status)
        self.assertEqual(previous_version + 1, client.groups[0]["version"])
        backend = client.groups[0]["cache_config"]["meta_indexer_config"][
            "meta_storage_backend_config"]
        self.assertEqual("local", backend["storage_type"])
        self.assertEqual(LOCAL_STORAGE_URI, backend["storage_uri"])
        self.assertEqual({L1P5_NAME, L2_NAME, "event-l2-new"}, {
            storage["global_unique_name"] for storage in client.storages})
        self.assertEqual([L1P5_NAME], client.groups[0]["storage_candidates"])
        self.assertEqual([L1P5_NAME, "event-l2-new"],
                         client.groups[0]["event_report_storage_candidates"])
        self.assertEqual(1, client.update_group_calls)
        self.assertEqual(1, client.list_instance_calls)
        self.assertEqual(BootstrapOutcome.COMPLETE,
                         bootstrap_once(config, client, reconcile=True))

    def test_reconcile_command_reports_required_restart(self):
        client = FakeAdminClient()
        bootstrap_once(parse_environment(dict(BASE_ENV)), client)
        client.instances = [{"instance_id": "i-1"}]
        environ = dict(BASE_ENV)
        environ["KVCM_INSTANCE_GROUP"] = group_env(max_key_count=123)
        config = parse_environment(environ)

        with patch("kvcm_ops.kvcm.bootstrap.parse_environment", return_value=config), \
             patch("kvcm_ops.kvcm.bootstrap.AdminClient", return_value=client), \
             self.assertLogs("kvcm_ops.bootstrap", level="WARNING") as logs:
            status = main(["--reconcile"])

        self.assertEqual(3, status)
        self.assertTrue(any("restart KVCM" in item for item in logs.output))

    def test_instance_query_failure_does_not_update_group(self):
        client = FakeAdminClient()
        bootstrap_once(parse_environment(dict(BASE_ENV)), client)
        original_group = copy.deepcopy(client.groups[0])
        client.list_instance_error = BootstrapError("list instance failed")
        environ = dict(BASE_ENV)
        environ["KVCM_INSTANCE_GROUP"] = group_env(max_key_count=123)

        with self.assertRaisesRegex(BootstrapError, "list instance failed"):
            bootstrap_once(parse_environment(environ), client, reconcile=True)

        self.assertEqual(0, client.update_group_calls)
        self.assertEqual(original_group, client.groups[0])

    def test_non_meta_group_change_does_not_query_instances_or_restart(self):
        client = FakeAdminClient()
        bootstrap_once(parse_environment(dict(BASE_ENV)), client)
        environ = dict(BASE_ENV)
        environ["KVCM_INSTANCE_GROUP"] = group_env(quota_capacity=123)
        outcome = bootstrap_once(parse_environment(environ), client, reconcile=True)
        self.assertEqual(BootstrapOutcome.COMPLETE, outcome)
        self.assertEqual(0, client.list_instance_calls)

    def test_pace_media_type_omitted_preserves_existing(self):
        environ = {
            "KVCM_PACE_STORAGE": json.dumps({
                "unique_name": PACE_NAME,
                "domain": "http://pace.example",
                "timeout": 30,
                "media_type": 2,
            }),
            "KVCM_INSTANCE_GROUP": group_env(),
        }
        client = FakeAdminClient()
        bootstrap_once(parse_environment(environ), client)
        environ["KVCM_PACE_STORAGE"] = json.dumps({
            "unique_name": PACE_NAME,
            "domain": "http://pace-new.example",
            "timeout": 30,
        })

        bootstrap_once(parse_environment(environ), client, reconcile=True)

        self.assertEqual(1, client.update_storage_calls)
        self.assertEqual(2, client.storages[0]["tair_mem_pool"]["media_type"])

    def test_pace_explicit_media_type_change_is_rejected(self):
        environ = {
            "KVCM_PACE_STORAGE": json.dumps({
                "unique_name": PACE_NAME,
                "domain": "http://pace.example",
                "timeout": 30,
                "media_type": 2,
            }),
            "KVCM_INSTANCE_GROUP": group_env(),
        }
        client = FakeAdminClient()
        bootstrap_once(parse_environment(environ), client)
        environ["KVCM_PACE_STORAGE"] = json.dumps({
            "unique_name": PACE_NAME,
            "domain": "http://pace.example",
            "timeout": 30,
            "media_type": 0,
        })

        with self.assertRaisesRegex(BootstrapError, "cannot be changed"):
            bootstrap_once(parse_environment(environ), client, reconcile=True)

    def test_version_conflict_is_retried(self):
        config = parse_environment(dict(BASE_ENV))
        client = FakeAdminClient()
        bootstrap_once(config, client)
        client.groups[0]["max_instance_count"] = 1
        client.version_conflicts_remaining = 1

        self.assertFalse(ensure_instance_group(client, config))
        self.assertEqual(2, client.update_group_calls)
        self.assertEqual(512, client.groups[0]["max_instance_count"])


if __name__ == "__main__":
    unittest.main()
