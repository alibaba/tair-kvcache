import os
import subprocess
import sys
import tempfile
import unittest
import venv
from pathlib import Path


class WheelInstallTest(unittest.TestCase):
    def setUp(self):
        self.wheel_path = Path(sys.argv[1]).resolve()
        self.start_server_script = Path(sys.argv[2]).resolve()
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)

        self.temp_path = Path(self.temp_dir.name)
        self.venv_path = self.temp_path / "venv"
        venv.EnvBuilder(with_pip=True).create(self.venv_path)
        self.python = self.venv_path / "bin" / "python"

        self.env = os.environ.copy()
        for variable in ("PYTHONHOME", "PYTHONPATH", "PYTHONUSERBASE"):
            self.env.pop(variable, None)
        self.env["PIP_DISABLE_PIP_VERSION_CHECK"] = "1"
        self.env["PYTHONNOUSERSITE"] = "1"
        self.env["PATH"] = str(self.venv_path / "bin") + os.pathsep + self.env["PATH"]

    def run_command(self, *args):
        result = subprocess.run(
            args,
            cwd=self.temp_path,
            env=self.env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        self.assertEqual(0, result.returncode, msg=result.stdout)
        return result.stdout

    def install_from_start_script(self):
        return self.run_command(
            "bash",
            "-c",
            'source "$1"; KVCM_OPS_WHEEL_PATH="$2"; install_kvcm_ops',
            "bash",
            self.start_server_script,
            self.wheel_path,
        )

    def test_install_resolves_requests_and_starts_http_command(self):
        self.run_command(
            self.python,
            "-I",
            "-c",
            "import importlib.util; "
            "assert importlib.util.find_spec('requests') is None",
        )

        self.install_from_start_script()
        self.run_command(self.python, "-m", "pip", "check")
        self.run_command(
            self.python,
            "-I",
            "-c",
            "from importlib.metadata import version; "
            "assert version('requests') == '2.32.5'; "
            "import certifi, charset_normalizer, idna, requests, urllib3",
        )

        top_level_help = self.run_command(
            self.python, "-I", "-m", "kvcm_ops", "--help"
        )
        self.assertIn("KVCM script entry", top_level_help)

        http_command_help = self.run_command(
            self.python,
            "-I",
            "-m",
            "kvcm_ops",
            "list_instance",
            "--help",
        )
        self.assertIn("kvcm: list_instance.", http_command_help)

        # A preinstalled image must start without pip, a wheel file or network access.
        self.run_command(self.python, "-m", "pip", "uninstall", "-y", "pip")
        self.env["PIP_NO_INDEX"] = "1"
        self.wheel_path = self.temp_path / "missing.whl"
        self.install_from_start_script()

    def test_installed_cli_with_missing_requests_is_repaired(self):
        self.run_command(
            self.python,
            "-m",
            "pip",
            "install",
            "--no-deps",
            self.wheel_path,
        )
        self.run_command(
            self.python,
            "-I",
            "-c",
            "import kvcm_ops; import importlib.util; "
            "assert importlib.util.find_spec('requests') is None",
        )

        self.install_from_start_script()
        self.run_command(self.python, "-m", "pip", "check")
        self.run_command(self.python, "-I", "-m", "kvcm_ops", "list_instance", "--help")


if __name__ == "__main__":
    unittest.main(argv=[sys.argv[0]])
