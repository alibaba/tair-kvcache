import sys
import unittest
import zipfile
from pathlib import Path


class KvMetaObjectClientWheelTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.wheel_path = Path(sys.argv[1])
        cls.archive = zipfile.ZipFile(cls.wheel_path)
        cls.names = set(cls.archive.namelist())

    @classmethod
    def tearDownClass(cls):
        cls.archive.close()

    def test_wheel_is_readable_and_contains_the_complete_runtime(self):
        self.assertIsNone(self.archive.testzip())
        self.assertIn("kv_cache_manager/client/__init__.py", self.names)
        self.assertIn(
            "kv_cache_manager/client/kv_meta_object_client.py",
            self.names,
        )
        native_extensions = {
            name
            for name in self.names
            if name.startswith("kv_cache_manager/client/pybind/kvcm_py_client")
            and name.endswith((".so", ".pyd"))
        }
        self.assertEqual(len(native_extensions), 1)
        self.assertFalse(any(name.endswith((".cc", ".h")) for name in self.names))
        self.assertFalse(any("/test/" in name for name in self.names))

    def test_wheel_metadata_declares_the_supported_python_contract(self):
        metadata_names = [
            name for name in self.names if name.endswith(".dist-info/METADATA")
        ]
        self.assertEqual(len(metadata_names), 1)
        metadata = self.archive.read(metadata_names[0]).decode("utf-8")

        self.assertIn("Name: kvcm_py_client", metadata)
        self.assertIn("Requires-Python: >=3.9", metadata)
        self.assertIn("Classifier: Programming Language :: Python :: 3", metadata)


if __name__ == "__main__":
    unittest.main(argv=[sys.argv[0]])
