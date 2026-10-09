import contextlib
import io
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import call, patch

# Support both the Bazel source layout and the wheel's namespace package.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from kvcm_ops.kvcm.instance import list_instance


def ok_response(**fields):
    return {"header": {"status": {"code": "OK"}}, **fields}


class ListInstanceTest(unittest.TestCase):
    def setUp(self):
        contexts = contextlib.ExitStack()
        self.addCleanup(contexts.close)
        self.http_post = contexts.enter_context(patch.object(list_instance, "http_post"))
        self.output = contexts.enter_context(contextlib.redirect_stdout(io.StringIO()))

    def run_command(self, *args):
        with patch("sys.argv", ["list_instance", *args]):
            list_instance.main()
        return json.loads(self.output.getvalue().split("respose:\n", 1)[1])

    def test_all_groups_preserve_instance_details_and_identify_groups(self):
        first = {"instance_id": "i1", "instance_group_name": "g1", "block_size": 128}
        second = {"instance_id": "i2", "instance_group_name": "g1", "block_size": 64}
        third = {"instance_id": "i3", "instance_group_name": "g2", "block_size": 256}
        self.http_post.side_effect = [
            ok_response(instance_group=[{"name": "g1"}, {"name": "empty"}, {"name": "g2"}]),
            ok_response(instance_info=[first, second]),
            ok_response(),
            ok_response(instance_info=[third]),
        ]

        result = self.run_command("-H", "http://manager:6492", "-t", "test-trace", "-v")

        self.assertEqual(ok_response(instance_info=[first, second, third]), result)
        self.assertEqual([
            call("http://manager:6492", "/api/listInstanceGroup", {"trace_id": "test-trace"}, True),
            *[
                call("http://manager:6492", "/api/listInstanceInfo",
                     {"trace_id": "test-trace", "instance_group_name": group}, True)
                for group in ("g1", "empty", "g2")
            ],
        ], self.http_post.call_args_list)

    def test_missing_group_field_is_filled_from_queried_group(self):
        self.http_post.side_effect = [
            ok_response(instance_group=[{"name": "g1"}]),
            ok_response(instance_info=[{"instance_id": "i1"}]),
        ]
        self.assertEqual(
            [{"instance_id": "i1", "instance_group_name": "g1"}],
            self.run_command()["instance_info"],
        )

    def test_no_groups_returns_empty_list(self):
        self.http_post.return_value = ok_response()
        self.assertEqual(ok_response(instance_info=[]), self.run_command())
        self.http_post.assert_called_once()

    def test_named_group_keeps_existing_response_and_single_request(self):
        response = ok_response(instance_info=[{"instance_id": "i1", "instance_group_name": "g1"}])
        response["header"]["request_id"] = "request-1"
        self.http_post.return_value = response
        self.assertEqual(response, self.run_command("-n", "g1"))
        self.http_post.assert_called_once_with(
            "http://localhost:6492", "/api/listInstanceInfo",
            {"trace_id": "default_trace_id", "instance_group_name": "g1"}, False,
        )

    def test_group_listing_failure_is_not_reported_as_empty_success(self):
        self.http_post.return_value = {"header": {"status": {"code": "INTERNAL_ERROR"}}}
        with self.assertRaisesRegex(RuntimeError, "listInstanceGroup failed"):
            self.run_command()
        self.assertEqual("", self.output.getvalue())
        self.http_post.assert_called_once()

    def test_instance_listing_failure_does_not_print_partial_results(self):
        self.http_post.side_effect = [
            ok_response(instance_group=[{"name": "g1"}, {"name": "g2"}, {"name": "g3"}]),
            ok_response(instance_info=[{"instance_id": "i1", "instance_group_name": "g1"}]),
            {"header": {"status": {"code": "INTERNAL_ERROR"}}},
        ]
        with self.assertRaisesRegex(RuntimeError, "listInstanceInfo failed for instance group 'g2'"):
            self.run_command()
        self.assertEqual("", self.output.getvalue())
        self.assertEqual(3, self.http_post.call_count)


if __name__ == "__main__":
    unittest.main()
