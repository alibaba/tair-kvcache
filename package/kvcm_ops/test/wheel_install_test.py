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

    def test_install_resolves_requests_and_starts_http_command(self):
        self.run_command(
            self.python,
            "-I",
            "-c",
            "import importlib.util; "
            "assert importlib.util.find_spec('requests') is None",
        )

        self.run_command(
            self.python,
            "-m",
            "pip",
            "install",
            "--no-input",
            self.wheel_path,
        )
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
        self.assertIn("kvcm: list_intance.", http_command_help)


if __name__ == "__main__":
    unittest.main(argv=[sys.argv[0]])
