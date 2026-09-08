import contextlib
import io
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from kvcm_ops.kvcm.instance_group import update_instance_group
from kvcm_ops.kvcm.instance_group.create_instance_group import create_instance_group
from kvcm_ops.kvcm.instance_group.util import MetaIndexerConfig, parse_instance_group_args


def _create_group(extra_args=()):
    with patch.object(sys, "argv", ["create_instance_group", "--name", "g1", "--storage_candidates", "nfs_01",
                                   *extra_args]):
        return create_instance_group(parse_instance_group_args(is_create=True)).to_json_data()


class InstanceGroupMutexTest(unittest.TestCase):
    def test_model_default_and_round_trip(self):
        config = MetaIndexerConfig().to_json_data()
        self.assertIs(config.pop("mutex_enabled"), True)
        self.assertIs(MetaIndexerConfig.from_json_data(config).to_json_data()["mutex_enabled"], True)
        for enabled in (True, False):
            with self.subTest(enabled=enabled):
                config["mutex_enabled"] = enabled
                self.assertIs(MetaIndexerConfig.from_json_data(config).to_json_data()["mutex_enabled"], enabled)

    def test_model_rejects_non_boolean(self):
        config = MetaIndexerConfig().to_json_data()
        for invalid in ("false", "true", 0, 1, None):
            with self.subTest(invalid=invalid), self.assertRaises(RuntimeError):
                config["mutex_enabled"] = invalid
                MetaIndexerConfig.from_json_data(config)

    def test_create_cli(self):
        for arguments, expected in (((), True), (("--mutex_enabled", "false"), False),
                                    (("--mutex_enabled", "true"), True)):
            with self.subTest(arguments=arguments):
                group = _create_group(arguments)
                self.assertIs(group["cache_config"]["meta_indexer_config"]["mutex_enabled"], expected)

    def test_update_cli_preserves_or_changes_flag(self):
        for stored in (None, True, False):
            for requested in (None, True, False):
                with self.subTest(stored=stored, requested=requested):
                    group = _create_group()
                    meta_config = group["cache_config"]["meta_indexer_config"]
                    if stored is None:
                        del meta_config["mutex_enabled"]
                    else:
                        meta_config["mutex_enabled"] = stored
                    argv = ["update_instance_group", "--name", "g1", "--user_data", "changed"]
                    if requested is not None:
                        argv.extend(["--mutex_enabled", str(requested).lower()])
                    ok = {"header": {"status": {"code": "OK"}}}
                    with patch.object(sys, "argv", argv), \
                            patch.object(update_instance_group, "http_post",
                                         side_effect=[{**ok, "instance_group": group}, ok]) as post, \
                            contextlib.redirect_stdout(io.StringIO()):
                        update_instance_group.main()
                    request = post.call_args_list[1].args[2]
                    expected = requested if requested is not None else (True if stored is None else stored)
                    self.assertIs(request["instance_group"]["cache_config"]["meta_indexer_config"]["mutex_enabled"],
                                  expected)
                    self.assertEqual(request["instance_group"]["user_data"], "changed")
                    self.assertEqual(request["instance_group"]["version"], request["current_version"] + 1)

    def test_cli_rejects_invalid_boolean(self):
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as error:
            _create_group(("--mutex_enabled", "0"))
        self.assertEqual(error.exception.code, 2)


if __name__ == "__main__":
    unittest.main()
