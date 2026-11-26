import os
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

from integration_test.testlib.module_base import ModuleBase
from integration_test.testlib.test_base import TestBase


class ModuleBaseProcessTest(unittest.TestCase):
    def setUp(self):
        self.harness = ModuleBase()

    def _start_process(self, code, *args):
        process = subprocess.Popen([sys.executable, "-c", code, *args])

        def cleanup():
            if process.poll() is None:
                process.kill()
            process.wait(timeout=5)

        self.addCleanup(cleanup)
        return process

    def test_wait_exit_ignores_pid_digits_in_another_process_command(self):
        exited = self._start_process("pass")
        exited.wait(timeout=5)
        other = self._start_process("import time; time.sleep(60)", str(exited.pid))

        self.harness._wait_exit("python", exited.pid, timeout=0.1)

        self.assertIsNone(other.poll())

    def test_wait_exit_accepts_an_unreaped_zombie(self):
        exited = self._start_process("pass")
        os.waitid(os.P_PID, exited.pid, os.WEXITED | os.WNOWAIT)

        self.harness._wait_exit("python", exited.pid, timeout=0.1)

        self.assertEqual(0, exited.wait(timeout=5))

    def test_wait_exit_reports_a_live_process_without_killing_it(self):
        process = self._start_process("import time; time.sleep(60)")

        with self.assertRaisesRegex(TimeoutError, f"python .*pid={process.pid}.*did not exit"):
            self.harness._wait_exit("python", process.pid, timeout=0.05)

        self.assertIsNone(process.poll())


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
