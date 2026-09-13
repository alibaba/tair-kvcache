"""Exercise the Spectrum entrypoint without a sidecar, allocator or server."""

import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest


class StartServerSpectrumTest(unittest.TestCase):
    def run_entrypoint(self, malloc_conf=None, ready=True):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            script = root / "start_server_spectrum.sh"
            shutil.copyfile(Path(__file__).with_name(script.name), script)
            # The real wrapper still checks its sidecar and execs its sibling.
            # Stub only those external endpoints; never invoke pip or KVCM.
            curl = root / "curl"
            curl.write_text("#!/bin/sh\nprintf '%s' " + ("200" if ready else "503") + "\n")
            curl.chmod(0o755)
            server = root / "start_server.sh"
            server.write_text(
                "#!/bin/bash\n"
                'printf "SERVER_ENV=%s\\n" "$MALLOC_CONF"\n'
                'printf "SERVER_ARG=%s\\n" "$@"\n'
            )
            server.chmod(0o755)
            env = dict(os.environ)
            env.pop("MALLOC_CONF", None)
            if malloc_conf is not None:
                env["MALLOC_CONF"] = malloc_conf
            env.update(
                PATH=str(root) + os.pathsep + os.environ["PATH"],
                SPECTRUM_PROBE_URL="http://127.0.0.1/unused-test-probe",
                SPECTRUM_PROBE_MAX_ATTEMPTS="1",
                SPECTRUM_PROBE_INTERVAL_SEC="0",
            )
            return subprocess.run(
                ["bash", str(script), "--config", "path with spaces"],
                env=env, capture_output=True, text=True, timeout=10,
            )

    def test_default_and_explicit_options(self):
        for configured, expected in (
            (None, "narenas:8"),
            ("", "narenas:8"),
            ("background_thread:true", "narenas:8,background_thread:true"),
            ("narenas:32,dirty_decay_ms:5000", "narenas:8,narenas:32,dirty_decay_ms:5000"),
        ):
            with self.subTest(configured=configured):
                result = self.run_entrypoint(configured)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIn("SERVER_ENV=" + expected, result.stdout.splitlines())
                self.assertIn("SERVER_ARG=--config", result.stdout.splitlines())
                self.assertIn("SERVER_ARG=path with spaces", result.stdout.splitlines())
                self.assertIn("Spectrum jemalloc configuration: MALLOC_CONF=" + expected, result.stdout)

    def test_sidecar_failure_does_not_start_server(self):
        result = self.run_entrypoint(ready=False)
        self.assertNotEqual(result.returncode, 0)
        self.assertNotIn("SERVER_ENV=", result.stdout)


if __name__ == "__main__":
    unittest.main()
