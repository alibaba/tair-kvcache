import os
import tempfile
import unittest
from unittest import mock

from integration_test.testlib.module_base import ModuleBase
from integration_test.testlib.test_base import TestBase


class TestBaseWorkdirTest(unittest.TestCase):
    def setUp(self):
        self.harness = TestBase()
        self.harness._testMethodName = "concurrent_case"

    def test_bazel_tmpdir_isolates_mutable_workdir_from_runfiles(self):
        with tempfile.TemporaryDirectory() as test_tmpdir:
            with mock.patch.dict(os.environ, {"TEST_TMPDIR": test_tmpdir}):
                expected_workdir = os.path.join(test_tmpdir, "concurrent_case")
                self.assertEqual(expected_workdir, self.harness.get_workdir())

                with mock.patch.object(ModuleBase, "create_symlink") as create_symlink:
                    self.harness._init_dirs(None)

                expected_source_root = os.path.abspath(
                    os.path.join(os.path.dirname(__file__), "../")
                )
                self.assertEqual(expected_workdir, self.harness.workdir)
                self.assertEqual(expected_source_root, self.harness.path_root)
                create_symlink.assert_called_once_with(
                    os.path.join(expected_source_root, "install_root"),
                    os.path.join(expected_workdir, "install_root"),
                )

    def test_non_bazel_run_preserves_source_relative_workdir(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            expected_workdir = os.path.join(
                os.path.abspath(os.path.join(os.path.dirname(__file__), "../")),
                "concurrent_case",
            )
            self.assertEqual(expected_workdir, self.harness.get_workdir())


if __name__ == "__main__":
    unittest.main()
