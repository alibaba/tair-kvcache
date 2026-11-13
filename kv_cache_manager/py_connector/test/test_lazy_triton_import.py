"""The connector must import on a host that has no Triton.

Triton is needed only by the built-in GPU data plane, and only once a backend
is built; importing the connector -- or the data-plane module itself -- must
not pull it in. Checked in a subprocess, because in-process the modules are
already imported and ``vllm_stubs`` stands Triton in whenever the real wheel
is absent, so the parent process could not tell what the import would have
done. A negative control proves the blocker actually bites.
"""

import os
import subprocess
import sys
import unittest
from pathlib import Path

#: Root to put on the child's PYTHONPATH: the directory holding
#: ``kv_cache_manager``, which is the checkout root and, under bazel, the
#: runfiles workspace root alike.
ROOT = Path(__file__).resolve().parents[3]

PRELUDE = """
import sys

# What a host without the Triton wheel looks like: any `import triton` from
# here on raises ImportError.
sys.modules["triton"] = None

from kv_cache_manager.py_connector.test import vllm_stubs  # noqa: F401
"""

CONNECTOR = (
    PRELUDE
    + """
import kv_cache_manager.py_connector.vllm.v1_connector  # noqa: F401
import kv_cache_manager.py_connector.vllm.data_transfer  # noqa: F401

loaded = [n for n in sys.modules if n.startswith("kv_cache_manager.py_connector.kernel")]
assert not loaded, loaded
print("imported")
"""
)

#: Negative control: the same blocker must stop the Triton kernels.
KERNELS = (
    PRELUDE
    + """
import kv_cache_manager.py_connector.kernel.batch_gather_scatter_helper  # noqa: F401
print("imported")
"""
)


class TestImportWithoutTriton(unittest.TestCase):
    def _child(self, script):
        env = dict(os.environ)
        path = [str(ROOT)]
        if env.get("PYTHONPATH"):
            path.append(env["PYTHONPATH"])
        env["PYTHONPATH"] = os.pathsep.join(path)
        return subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True,
            text=True,
            env=env,
            timeout=300,
        )

    def test_connector_imports_with_triton_blocked(self):
        done = self._child(CONNECTOR)
        self.assertEqual(done.returncode, 0, done.stderr)
        self.assertIn("imported", done.stdout)

    def test_the_block_actually_bites(self):
        # Without this control the test above would also pass on a host where
        # the blocker silently stopped working.
        done = self._child(KERNELS)
        self.assertNotEqual(done.returncode, 0)
        self.assertIn("triton", done.stderr)


if __name__ == "__main__":
    unittest.main()
