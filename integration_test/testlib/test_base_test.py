import os
import subprocess
import sys
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
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


class WorkerExitTest(unittest.TestCase):
    def test_wait_does_not_match_another_process_command_line(self):
        exited = subprocess.Popen([sys.executable, '-c', 'pass'])
        exited.wait(timeout=5)
        lookalike = subprocess.Popen([
            sys.executable, '-c', 'import time; time.sleep(30)',
            'kvcm_exit_test_worker', str(exited.pid),
        ])
        try:
            with ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(
                    ModuleBase()._wait_exit, 'kvcm_exit_test_worker', exited.pid,
                )
                try:
                    future.result(timeout=2)
                    self.assertIsNone(lookalike.poll())
                finally:
                    lookalike.terminate()
                    lookalike.wait(timeout=5)
        finally:
            if lookalike.poll() is None:
                lookalike.kill()
                lookalike.wait(timeout=5)

    @mock.patch('integration_test.testlib.module_base.subprocess.check_output')
    def test_wait_reports_a_live_process_instead_of_hanging(self, ps):
        ps.return_value = 'Sl\n'
        with self.assertRaisesRegex(TimeoutError, r'pid=123 .*state=Sl'):
            ModuleBase()._wait_exit('worker', 123, timeout=0)

    @mock.patch('integration_test.testlib.module_base.subprocess.check_output')
    def test_exited_zombie_does_not_block_cleanup(self, ps):
        ps.return_value = 'Z\n'
        ModuleBase()._wait_exit('worker', 123, timeout=0)


if __name__ == "__main__":
    unittest.main()
