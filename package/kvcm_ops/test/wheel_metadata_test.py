import email
import sys
import unittest
import zipfile
from pathlib import Path


class WheelMetadataTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        wheel_path = Path(sys.argv[1])
        requirements_path = Path(sys.argv[2])
        cls.expected_requirements = [
            line.strip()
            for line in requirements_path.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        ]

        with zipfile.ZipFile(wheel_path) as archive:
            metadata_names = [
                name
                for name in archive.namelist()
                if name.endswith(".dist-info/METADATA")
            ]
            if len(metadata_names) != 1:
                raise AssertionError(
                    f"expected exactly one METADATA file, found {metadata_names}"
                )
            cls.metadata = email.message_from_bytes(archive.read(metadata_names[0]))

    def test_declares_shared_runtime_requirements(self):
        self.assertEqual(["requests==2.32.5"], self.expected_requirements)
        self.assertEqual(
            self.expected_requirements,
            self.metadata.get_all("Requires-Dist", []),
        )


if __name__ == "__main__":
    unittest.main(argv=[sys.argv[0]])
