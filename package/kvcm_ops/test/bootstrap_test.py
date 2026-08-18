import copy
from pathlib import Path
import sys
import unittest
from unittest.mock import patch


sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from kvcm_ops.kvcm.bootstrap import (  # noqa: E402
    BootstrapError,
    AdminClient,
    _apply_managed_group_fields,
    _build_new_instance_group,
    _managed_group_view,
    bootstrap_once,
    ensure_instance_group,
    parse_environment,
)


BASE_ENV = {
    "KVCM_ENABLE_SUBSCRIBER_EVENT_REPORT": "true",
    "KVCM_ENABLE_V6D_EVENT_REPORT": "true",
    "KVCM_INSTANCE_GROUP_NAME": "event-group",
    "KVCM_EVENT_REPORT_L1P5_STORAGE_NAME": "event-l1p5",
    "KVCM_EVENT_REPORT_L2_STORAGE_NAME": "event-l2",
    "KVCM_META_STORAGE_TYPE": "cached",
    "KVCM_META_STORAGE_URI": (
        "redis://user:p%40ss@redis.example:6379/?cluster_name=test"
        "&max_instance_count=512&quota_capacity=2087740652912"
        "&max_key_count=1000000000&mutex_shard_num=131072&batch_key_size=1024"
        "&async_queue_count=8&heartbeat_timeout_ms=30000"
    ),
}


class FakeAdminClient:
    def __init__(self, is_leader=True):
        self.is_leader = is_leader
        self.storages = []
        self.groups = []
        self.add_calls = 0
        self.update_storage_calls = 0
        self.create_group_calls = 0
        self.update_group_calls = 0
        self.version_conflicts_remaining = 0

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

    def create_instance_group(self, instance_group):
        self.create_group_calls += 1
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
    def test_dual_report_configuration(self):
        config = parse_environment(dict(BASE_ENV))

        self.assertEqual("event-l1p5", config.primary_storage_name)
        self.assertEqual(
            ["event-l1p5", "event-l2"],
            config.event_report_storage_names,
        )
        self.assertEqual(512, config.max_instance_count)
        self.assertEqual(2087740652912, config.quota_capacity)
        self.assertEqual(30000, config.event_report_spec["heartbeat_timeout_ms"])

    def test_four_report_switch_combinations(self):
        for subscriber, v6d, expected in (
            (False, False, []),
            (True, False, ["event-l1p5"]),
            (False, True, ["event-l2"]),
            (True, True, ["event-l1p5", "event-l2"]),
        ):
            with self.subTest(subscriber=subscriber, v6d=v6d):
                environ = dict(BASE_ENV)
                environ["KVCM_ENABLE_SUBSCRIBER_EVENT_REPORT"] = str(subscriber).lower()
                environ["KVCM_ENABLE_V6D_EVENT_REPORT"] = str(v6d).lower()
                config = parse_environment(environ)
                self.assertEqual(expected, config.event_report_storage_names)

    def test_invalid_boolean_is_rejected(self):
        environ = dict(BASE_ENV)
        environ["KVCM_ENABLE_V6D_EVENT_REPORT"] = "yes"

        with self.assertRaisesRegex(BootstrapError, "must be true or false"):
            parse_environment(environ)

    def test_mutex_shards_must_be_power_of_two(self):
        environ = dict(BASE_ENV)
        environ["KVCM_META_STORAGE_URI"] = environ["KVCM_META_STORAGE_URI"].replace(
            "mutex_shard_num=131072", "mutex_shard_num=7")

        with self.assertRaisesRegex(BootstrapError, "power of two"):
            parse_environment(environ)

    def test_metadata_backend_mode_is_optional_and_bounded(self):
        config = parse_environment(dict(BASE_ENV))
        self.assertIsNone(config.metadata_backend_mode)

        for mode in range(1, 5):
            environ = dict(BASE_ENV)
            environ["KVCM_METADATA_BACKEND_MODE"] = str(mode)
            self.assertEqual(mode, parse_environment(environ).metadata_backend_mode)

        environ = dict(BASE_ENV)
        environ["KVCM_METADATA_BACKEND_MODE"] = "5"
        with self.assertRaisesRegex(BootstrapError, "range 1..4"):
            parse_environment(environ)

    def test_uri_error_does_not_expose_credentials(self):
        environ = dict(BASE_ENV)
        secret = "do-not-log-this"
        environ["KVCM_META_STORAGE_URI"] = "redis://user:{}@redis.example/".format(secret)

        with self.assertRaises(BootstrapError) as context:
            parse_environment(environ)
        self.assertNotIn(secret, str(context.exception))


class ErrorReportingTest(unittest.TestCase):
    @patch("kvcm_ops.kvcm.bootstrap.http_post")
    def test_admin_transport_error_keeps_exception_details(self, mock_http_post):
        mock_http_post.side_effect = RuntimeError("connection refused")

        with self.assertRaises(BootstrapError) as context:
            AdminClient().post("/api/listStorage", {}, "listStorage")

        self.assertIn("listStorage", str(context.exception))
        self.assertIn("RuntimeError", str(context.exception))
        self.assertIn("connection refused", str(context.exception))


class DesiredConfigurationTest(unittest.TestCase):
    def test_v6d_only_uses_l2_for_primary_and_quota(self):
        environ = dict(BASE_ENV)
        environ["KVCM_ENABLE_SUBSCRIBER_EVENT_REPORT"] = "false"
        config = parse_environment(environ)

        group = _build_new_instance_group(config)

        self.assertEqual(["event-l2"], group["storage_candidates"])
        self.assertEqual(["event-l2"], group["event_report_storage_candidates"])
        self.assertEqual(
            "ST_EVENT_REPORT_L2",
            group["quota"]["quota_config"][0]["storage_type"],
        )

    def test_update_preserves_unmanaged_fields_and_extra_info(self):
        environ = dict(BASE_ENV)
        environ["KVCM_METADATA_BACKEND_MODE"] = "4"
        config = parse_environment(environ)
        current = _build_new_instance_group(config)
        current["user_data"] = "keep-user-data"
        current["revisit_interval_buckets"] = "1,5,30"
        current["extra_info"] = '{"keep":"value","metadata_backend_mode":2}'

        updated = _apply_managed_group_fields(current, config)

        self.assertEqual("keep-user-data", updated["user_data"])
        self.assertEqual("1,5,30", updated["revisit_interval_buckets"])
        self.assertEqual(
            {"keep": "value", "metadata_backend_mode": 4},
            __import__("json").loads(updated["extra_info"]),
        )

    def test_unset_metadata_mode_preserves_existing_value(self):
        config = parse_environment(dict(BASE_ENV))
        current = _build_new_instance_group(config)
        current["extra_info"] = '{"metadata_backend_mode":3,"keep":true}'

        updated = _apply_managed_group_fields(current, config)

        self.assertEqual(current["extra_info"], updated["extra_info"])


class BootstrapReconciliationTest(unittest.TestCase):
    def test_first_run_creates_and_second_run_is_idempotent(self):
        config = parse_environment(dict(BASE_ENV))
        client = FakeAdminClient()

        self.assertTrue(bootstrap_once(config, client))
        self.assertTrue(bootstrap_once(config, client))

        self.assertEqual(2, client.add_calls)
        self.assertEqual(0, client.update_storage_calls)
        self.assertEqual(1, client.create_group_calls)
        self.assertEqual(0, client.update_group_calls)
        expected = _managed_group_view(_build_new_instance_group(config), config)
        self.assertEqual(expected, _managed_group_view(client.groups[0], config))

    def test_follower_does_not_write(self):
        config = parse_environment(dict(BASE_ENV))
        client = FakeAdminClient(is_leader=False)

        self.assertFalse(bootstrap_once(config, client))

        self.assertEqual([], client.storages)
        self.assertEqual([], client.groups)

    def test_same_storage_name_updates_storage_in_place(self):
        config = parse_environment(dict(BASE_ENV))
        client = FakeAdminClient()
        bootstrap_once(config, client)
        client.storages[0]["event_report"]["heartbeat_timeout_ms"] = 1

        bootstrap_once(config, client)

        self.assertEqual(1, client.update_storage_calls)
        self.assertEqual(2, len(client.storages))
        self.assertEqual(
            30000,
            client.storages[0]["event_report"]["heartbeat_timeout_ms"],
        )

    def test_storage_name_change_creates_new_and_keeps_old_storage(self):
        config = parse_environment(dict(BASE_ENV))
        client = FakeAdminClient()
        bootstrap_once(config, client)
        environ = dict(BASE_ENV)
        environ["KVCM_EVENT_REPORT_L2_STORAGE_NAME"] = "event-l2-new"
        updated_config = parse_environment(environ)

        bootstrap_once(updated_config, client)

        self.assertEqual(
            {"event-l1p5", "event-l2", "event-l2-new"},
            {storage["global_unique_name"] for storage in client.storages},
        )
        self.assertEqual(
            ["event-l1p5", "event-l2-new"],
            client.groups[0]["event_report_storage_candidates"],
        )

    def test_instance_group_name_change_creates_new_and_keeps_old_group(self):
        config = parse_environment(dict(BASE_ENV))
        client = FakeAdminClient()
        bootstrap_once(config, client)
        environ = dict(BASE_ENV)
        environ["KVCM_INSTANCE_GROUP_NAME"] = "event-group-new"

        bootstrap_once(parse_environment(environ), client)

        self.assertEqual(
            {"event-group", "event-group-new"},
            {group["name"] for group in client.groups},
        )

    def test_uri_change_updates_group_in_place(self):
        config = parse_environment(dict(BASE_ENV))
        client = FakeAdminClient()
        bootstrap_once(config, client)
        environ = dict(BASE_ENV)
        environ["KVCM_META_STORAGE_URI"] = environ["KVCM_META_STORAGE_URI"].replace(
            "cluster_name=test", "cluster_name=test-new")

        bootstrap_once(parse_environment(environ), client)

        self.assertEqual(1, client.create_group_calls)
        self.assertEqual(1, client.update_group_calls)
        self.assertIn(
            "cluster_name=test-new",
            client.groups[0]["cache_config"]["meta_indexer_config"]
            ["meta_storage_backend_config"]["storage_uri"],
        )

    def test_version_conflict_is_retried(self):
        config = parse_environment(dict(BASE_ENV))
        client = FakeAdminClient()
        bootstrap_once(config, client)
        client.groups[0]["max_instance_count"] = 1
        client.version_conflicts_remaining = 1

        ensure_instance_group(client, config)

        self.assertEqual(2, client.update_group_calls)
        self.assertEqual(512, client.groups[0]["max_instance_count"])


if __name__ == "__main__":
    unittest.main()
