#!/usr/bin/env python3

import json
import tempfile
import unittest
from pathlib import Path

from run.multi_instance_replay import _inspect_optimizer_trace


class MultiInstanceReplayTest(unittest.TestCase):
    def test_inspect_accepts_standard_trace(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            trace_path = Path(temp_dir) / "instance_a.jsonl"
            rows = [
                {
                    "type": "get",
                    "instance_id": "instance-a",
                    "trace_id": "short-read",
                    "timestamp_ns": 1000,
                    "keys": [],
                    "input_len": 128,
                    "query_type": "prefix_match",
                    "block_mask": [],
                },
                {
                    "type": "write",
                    "instance_id": "instance-a",
                    "trace_id": "short-write",
                    "timestamp_ns": 1001,
                    "keys": [1],
                    "input_len": "ignored",
                },
            ]
            with trace_path.open("w", encoding="utf-8") as f:
                for row in rows:
                    f.write(json.dumps(row) + "\n")

            self.assertEqual(_inspect_optimizer_trace(str(trace_path)), "instance-a")

    def test_inspect_rejects_non_array_keys(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            trace_path = Path(temp_dir) / "instance_a.jsonl"
            row = {
                "type": "get",
                "instance_id": "instance-a",
                "timestamp_ns": 1000,
                "keys": 0,
                "input_len": 128,
                "query_type": "prefix_match",
                "block_mask": [],
            }
            trace_path.write_text(json.dumps(row) + "\n", encoding="utf-8")

            with self.assertRaisesRegex(SystemExit, "must contain keys array"):
                _inspect_optimizer_trace(str(trace_path))

    def test_inspect_accepts_request_trace(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            trace_path = Path(temp_dir) / "instance_a.jsonl"
            row = {
                "type": "request",
                "instance_id": "instance-a",
                "trace_id": "request",
                "timestamp_ns": 1000,
                "keys": [1],
                "input_len": 128,
                "block_mask": [],
            }
            trace_path.write_text(json.dumps(row) + "\n", encoding="utf-8")

            self.assertEqual(_inspect_optimizer_trace(str(trace_path)), "instance-a")

if __name__ == "__main__":
    unittest.main()
