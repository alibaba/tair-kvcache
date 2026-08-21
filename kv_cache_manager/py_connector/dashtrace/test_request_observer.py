import json
import struct
import tempfile
import time
import unittest
from dataclasses import dataclass, field

from kv_cache_manager.py_connector.dashtrace.request_observer import (
    RequestObserver,
    extract_request_id,
    extract_token_ids,
)
from kv_cache_manager.py_connector.dashtrace.trace_recorder import (
    TraceRecorder,
    TraceRecorderConfig,
)


@dataclass
class _Tensor:
    name: str
    datatype: str = "INT64"
    int64_contents: list[int] = field(default_factory=list)
    int_contents: list[int] = field(default_factory=list)
    bytes_contents: object = b""
    contents: object = None


class _Request:
    def __init__(self, request_id="request-1", inputs=None, parameters=None):
        self.id = request_id
        self.inputs = inputs or []
        self.parameters = parameters or {}

    def get_input(self, name):
        return next((item for item in self.inputs if item.name == name), None)


class _Forwarder:
    def __init__(self):
        self.items = []
        self.closed = False

    def submit_observation(self, observation):
        self.items.append(observation)
        return True

    def close(self):
        self.closed = True


class _Sink:
    def __init__(self, enabled):
        self.enabled = enabled
        self.items = []
        self.closed = False

    def submit(self, observation):
        if not self.enabled:
            return False
        self.items.append(observation)
        return True

    submit_observation = submit

    def close(self):
        self.closed = True


class _FailingSink(_Sink):
    def submit(self, observation):
        raise RuntimeError("synthetic sink failure")

    submit_observation = submit


class RequestObserverTest(unittest.TestCase):
    def test_extracts_typed_and_raw_token_tensors(self):
        request = _Request(inputs=[_Tensor("input_ids", int64_contents=[1, 2])])
        self.assertEqual([1, 2], extract_token_ids(request))

        request = _Request(
            inputs=[
                _Tensor(
                    "prompt_token_ids",
                    datatype="INT32",
                    bytes_contents=struct.pack("<3i", 3, 4, 5),
                )
            ]
        )
        self.assertEqual([3, 4, 5], extract_token_ids(request))

        request = _Request(
            inputs=[
                _Tensor(
                    "input_token_ids",
                    datatype="INT64",
                    bytes_contents=[struct.pack("<2q", 6, 7)],
                )
            ]
        )
        self.assertEqual([6, 7], extract_token_ids(request))

    def test_extracts_parameter_fallback_and_request_id(self):
        request = _Request(
            request_id="",
            parameters={"token_ids": [8, 9], "trace_id": "trace-9"},
        )
        self.assertEqual([8, 9], extract_token_ids(request))
        self.assertEqual("trace-9", extract_request_id(request))

    def test_extracts_nested_proto_contents_and_raw_input(self):
        nested = _Tensor("nested", int_contents=[21, 22])
        request = _Request(inputs=[_Tensor("input_ids", contents=nested)])
        self.assertEqual([21, 22], extract_token_ids(request))

        request = _Request(inputs=[_Tensor("input_ids", datatype="INT32")])
        request.raw_input_contents = [struct.pack("<2i", 31, 32)]
        self.assertEqual([31, 32], extract_token_ids(request))

    def test_observer_records_and_forwards_without_waiting_for_io(self):
        with tempfile.TemporaryDirectory() as directory:
            recorder = TraceRecorder(
                TraceRecorderConfig(
                    directory=directory,
                    queue_capacity=8,
                    segment_bytes=1024,
                    max_disk_bytes=2048,
                )
            )
            forwarder = _Forwarder()
            observer = RequestObserver("instance-1", recorder, forwarder)
            self.assertTrue(
                observer.observe(
                    _Request(inputs=[_Tensor("input_ids", int64_contents=[11, 12])])
                )
            )
            observer.close()

            files = list(__import__("pathlib").Path(directory).glob("*.jsonl"))
            self.assertEqual(1, len(files))
            record = json.loads(files[0].read_text().strip())
            self.assertEqual("request-1", record["trace_id"])
            self.assertEqual("instance-1", record["instance_id"])
            self.assertEqual([11, 12], record["token_ids"])
            self.assertEqual(["request-1"], [item.trace_id for item in forwarder.items])
            self.assertEqual([11, 12], list(forwarder.items[0].token_ids))
            self.assertTrue(forwarder.closed)

    def test_recording_and_reporting_are_independent_sinks(self):
        request = _Request(inputs=[_Tensor("input_ids", int64_contents=[41, 42])])
        modes = (
            (True, False, True),
            (False, True, True),
            (True, True, True),
            (False, False, False),
        )
        for record_enabled, report_enabled, expected in modes:
            with self.subTest(
                record_enabled=record_enabled,
                report_enabled=report_enabled,
            ):
                recorder = _Sink(record_enabled)
                reporter = _Sink(report_enabled)
                observer = RequestObserver("instance-1", recorder, reporter)

                self.assertEqual(expected, observer.observe(request))
                self.assertEqual(int(record_enabled), len(recorder.items))
                self.assertEqual(int(report_enabled), len(reporter.items))
                if record_enabled and report_enabled:
                    self.assertIs(recorder.items[0], reporter.items[0])
                    self.assertIs(
                        recorder.items[0].token_ids,
                        reporter.items[0].token_ids,
                    )
                observer.close()
                self.assertTrue(recorder.closed)
                self.assertTrue(reporter.closed)

    def test_observer_assigns_one_monotonic_sequence_to_both_sinks(self):
        recorder = _Sink(True)
        reporter = _Sink(True)
        observer = RequestObserver("instance-1", recorder, reporter)
        for request_id in ("r-1", "r-2", "r-3"):
            self.assertTrue(
                observer.observe(
                    _Request(
                        request_id=request_id,
                        inputs=[_Tensor("input_ids", int64_contents=[1])],
                    )
                )
            )
        observer.close()

        self.assertEqual([0, 1, 2], [item.sequence for item in recorder.items])
        self.assertEqual([0, 1, 2], [item.sequence for item in reporter.items])
        self.assertEqual(
            [item.timestamp_ns for item in recorder.items],
            [item.timestamp_ns for item in reporter.items],
        )

    def test_sink_failure_does_not_block_the_other_sink(self):
        request = _Request(inputs=[_Tensor("input_ids", int64_contents=[51])])

        recorder = _FailingSink(True)
        reporter = _Sink(True)
        observer = RequestObserver("instance-1", recorder, reporter)
        self.assertTrue(observer.observe(request))
        self.assertEqual(1, len(reporter.items))
        observer.close()

        recorder = _Sink(True)
        reporter = _FailingSink(True)
        observer = RequestObserver("instance-1", recorder, reporter)
        self.assertTrue(observer.observe(request))
        self.assertEqual(1, len(recorder.items))
        observer.close()

    def test_recorder_rotates_and_removes_old_segments(self):
        with tempfile.TemporaryDirectory() as directory:
            recorder = TraceRecorder(
                TraceRecorderConfig(
                    directory=directory,
                    queue_capacity=32,
                    segment_bytes=120,
                    max_disk_bytes=240,
                )
            )
            for index in range(10):
                recorder.record(f"r-{index}", "instance", list(range(8)))
            deadline = time.time() + 2
            while recorder._queue.qsize() and time.time() < deadline:
                time.sleep(0.01)
            recorder.close()

            paths = list(__import__("pathlib").Path(directory).glob("*.jsonl"))
            self.assertLessEqual(sum(path.stat().st_size for path in paths), 360)


if __name__ == "__main__":
    unittest.main()
