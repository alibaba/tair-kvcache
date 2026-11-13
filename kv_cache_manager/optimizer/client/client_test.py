import unittest

from kv_cache_manager.optimizer.client.base import OptimizerClientBase
from kv_cache_manager.protocol.protobuf import optimizer_service_pb2 as pb


class OptimizerClientTest(unittest.TestCase):
    def test_selects_exact_ttl_without_capacity(self):
        response = pb.TraceQueryResponse(total_blocks=2, input_token_len=8)
        for ttl, hits in [(10, 0), (30, 2)]:
            result = response.ttl_results.add(ttl_seconds=ttl)
            result.theoretical_result.max_hit_count = hits
        self.assertEqual((2, [], 0), OptimizerClientBase.parse_trace_query(response, 10))
        self.assertEqual((2, [], 2), OptimizerClientBase.parse_trace_query(response, 30))
        with self.assertRaises(ValueError):
            OptimizerClientBase.parse_trace_query(response, 60)

    def test_capacity_results_belong_to_selected_ttl(self):
        response = pb.TraceQueryResponse(total_blocks=2)
        result = response.ttl_results.add(ttl_seconds=30)
        result.capacity_results.add(capacity_gb=1.0, cache_hit_count=1)
        result.theoretical_result.max_hit_count = 2
        self.assertEqual((2, [(1.0, 1)], 2), OptimizerClientBase.parse_trace_query(response, 30))
        response.ttl_results.add(ttl_seconds=30)
        with self.assertRaises(ValueError):
            OptimizerClientBase.parse_trace_query(response, 30)


if __name__ == "__main__":
    unittest.main()
