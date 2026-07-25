import argparse
from pathlib import Path
import sys
import unittest

# Bazel keeps sources under package/kvcm_ops while the wheel exposes kvcm_ops
# as a top-level namespace package.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from kvcm_ops.kvcm.storage.util import (
    add_event_report_sub_parser,
    gen_event_report_config_data,
)


class EventReportStorageArgsTest(unittest.TestCase):
    def _parse_args(self, *args):
        parser = argparse.ArgumentParser()
        subparsers = parser.add_subparsers(dest="storage_type", required=True)
        add_event_report_sub_parser(
            subparsers,
            "event_report_l1p5",
            "L1.5 event report storage options",
            "ST_EVENT_REPORT_L1P5",
        )
        return parser.parse_args(["event_report_l1p5", *args])

    def test_snapshot_min_interval_is_added_to_storage_spec(self):
        args = self._parse_args(
            "--heartbeat_timeout_ms",
            "30000",
            "--cleanup_grace_ms",
            "300000",
            "--liveness_check_interval_ms",
            "5000",
            "--snapshot_min_interval_ms",
            "1000",
        )

        self.assertEqual(
            {
                "heartbeat_timeout_ms": 30000,
                "cleanup_grace_ms": 300000,
                "liveness_check_interval_ms": 5000,
                "snapshot_min_interval_ms": 1000,
            },
            gen_event_report_config_data(args),
        )

    def test_snapshot_min_interval_is_optional(self):
        args = self._parse_args()

        self.assertNotIn(
            "snapshot_min_interval_ms",
            gen_event_report_config_data(args),
        )

    def test_snapshot_min_interval_must_be_positive(self):
        for invalid_value in ("0", "-1"):
            with self.subTest(invalid_value=invalid_value):
                with self.assertRaises(SystemExit):
                    self._parse_args("--snapshot_min_interval_ms", invalid_value)


if __name__ == "__main__":
    unittest.main()
