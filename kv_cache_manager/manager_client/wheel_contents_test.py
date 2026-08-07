import sys
import unittest
import zipfile
from pathlib import Path


class WheelContentsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.wheel_path = Path(sys.argv[1])
        cls.archive = zipfile.ZipFile(cls.wheel_path)
        cls.names = set(cls.archive.namelist())

    @classmethod
    def tearDownClass(cls):
        cls.archive.close()

    def test_contains_only_manager_client_runtime(self):
        expected_files = {
            "kv_cache_manager/manager_client/__init__.py",
            "kv_cache_manager/py_connector/common/_version_info.py",
            "kv_cache_manager/py_connector/common/logger.py",
            "kv_cache_manager/py_connector/common/manager_client.py",
            "kv_cache_manager/py_connector/common/service_discovery.py",
            "kv_cache_manager/py_connector/common/service_discovery_factory.py",
            "kv_cache_manager/py_connector/common/static_service_discovery.py",
            "stub_source/kv_cache_manager/py_connector/common/__init__.py",
            "stub_source/kv_cache_manager/py_connector/common/spectrum_service_discovery.py",
        }
        runtime_python_files = {name for name in self.names if name.endswith(".py")}
        self.assertEqual(expected_files, runtime_python_files)
        self.assertFalse(any(name.endswith((".so", ".pyd")) for name in self.names))

    def test_declares_requests_dependency(self):
        metadata_name = next(
            name for name in self.names if name.endswith(".dist-info/METADATA")
        )
        metadata = self.archive.read(metadata_name).decode("utf-8")
        self.assertIn("Name: tair-kvcache-manager-client", metadata)
        self.assertIn("License: Apache-2.0", metadata)
        self.assertIn(
            "Classifier: License :: OSI Approved :: Apache Software License",
            metadata,
        )
        self.assertIn("Requires-Dist: requests>=2.31.0,<3", metadata)
        self.assertIn("Requires-Python: >=3.9", metadata)

    def test_contains_license(self):
        license_name = next(
            name for name in self.names if name.endswith(".dist-info/LICENSE")
        )
        license_text = self.archive.read(license_name).decode("utf-8")
        self.assertIn("Apache License", license_text)
        self.assertIn("Version 2.0", license_text)


if __name__ == "__main__":
    unittest.main(argv=[sys.argv[0]])
