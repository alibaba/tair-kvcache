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
            'source "$1"; KVCM_OPS_WHEEL_PATH="$2"; PYTHON_BIN="$3"; install_kvcm_ops',
            "bash",
            self.start_server_script,
            self.wheel_path,
            self.python,
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

    def test_same_version_cli_without_bootstrap_is_repaired(self):
        self.install_from_start_script()
        self.run_command(
            self.python,
            "-I",
            "-c",
            "from importlib.util import find_spec; from pathlib import Path; "
            "Path(find_spec('kvcm_ops.kvcm.bootstrap').origin).unlink()",
        )
        self.install_from_start_script()
        self.run_command(
            self.python,
            "-I",
            "-c",
            "from importlib.metadata import version; "
            "assert version('kvcm_ops') == '0.1.0'; "
            "import kvcm_ops.kvcm.bootstrap, requests",
        )

    def test_preinstalled_bootstrap_does_not_require_pip_or_wheel(self):
        self.install_from_start_script()
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


class StartServerPythonTest(unittest.TestCase):
    def check_python_environment(self, image_python_exists, image_first):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            image_bin = temp_path / "image python" / "bin"
            path_bin = temp_path / "system" / "bin"
            image_bin.mkdir(parents=True)
            path_bin.mkdir(parents=True)
            image_python = image_bin / "python3"
            path_python = path_bin / "python3"
            python_log = temp_path / "python.log"
            ready_file = temp_path / "ops-ready"
            python_stub = """#!/bin/bash
printf '%s\\t%s\\n' "$0" "$*" >> "$TEST_PYTHON_LOG"
if [ "$1" = "-c" ] && [[ "$2" == *"import kvcm_ops"* ]]; then
    [ -f "$TEST_OPS_READY" ]
    exit $?
fi
if [ "$1" = "-m" ] && [ "$2" = "pip" ]; then
    touch "$TEST_OPS_READY"
fi
"""
            if image_python_exists:
                image_python.write_text(python_stub)
                image_python.chmod(0o755)
                path_python.write_text("#!/bin/bash\nexit 91\n")
            else:
                path_python.write_text(python_stub)
            path_python.chmod(0o755)

            script = temp_path / "start_server.sh"
            script.write_text(
                Path(sys.argv[2]).read_text().replace(
                    "/opt/ac2/bin/python3", str(image_python)
                )
            )
            env = os.environ.copy()
            bins = [image_bin, path_bin] if image_first else [path_bin, image_bin]
            env["PATH"] = os.pathsep.join(str(path) for path in bins) + os.pathsep + env["PATH"]
            env["TEST_PYTHON_LOG"] = str(python_log)
            env["TEST_OPS_READY"] = str(ready_file)
            result = subprocess.run(
                [
                    "bash",
                    "-c",
                    'original_path=$PATH; source "$1"; set +x; '
                    'KVCM_OPS_WHEEL_PATH="$2"; install_kvcm_ops || exit $?; '
                    'command() { if [ "$1" = -v ] && [ "$2" = curl ]; then return 1; '
                    'else builtin command "$@"; fi; }; '
                    'server_pid=$$; ADMIN_URL=http://127.0.0.1:6492; '
                    'wait_for_admin_health || exit $?; '
                    'run_bootstrap_command || exit $?; [ "$PATH" = "$original_path" ]',
                    "bash",
                    str(script),
                    str(temp_path / "packaged.whl"),
                ],
                cwd=temp_path,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                timeout=10,
            )
            self.assertEqual(0, result.returncode, msg=result.stdout)
            calls = [line.split("\t", 1) for line in python_log.read_text().splitlines()]
            selected_python = image_python if image_python_exists else path_python
            self.assertEqual([str(selected_python)] * 4, [call[0] for call in calls])
            self.assertIn("kvcm_ops.kvcm.bootstrap", calls[0][1])
            self.assertIn("-m pip install --force-reinstall", calls[1][1])
            self.assertIn("requests.get", calls[2][1])
            self.assertEqual(
                "-m kvcm_ops bootstrap --host http://127.0.0.1:6492", calls[3][1]
            )

    def test_whale_system_python_precedes_image_python(self):
        self.check_python_environment(image_python_exists=True, image_first=False)

    def test_spectrum_image_python_is_first(self):
        self.check_python_environment(image_python_exists=True, image_first=True)

    def test_without_ac2_uses_python_from_path(self):
        self.check_python_environment(image_python_exists=False, image_first=False)


if __name__ == "__main__":
    unittest.main(argv=[sys.argv[0]])
