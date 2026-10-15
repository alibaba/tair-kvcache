import unittest

import grpc

from integration_test.testlib.test_base import TestBase
from kv_cache_manager.protocol.protobuf.debug_service_pb2_grpc import DebugServiceStub
from kv_cache_manager.protocol.protobuf.meta_service_pb2 import (
    GetCacheMetaDetailRequest,
    LocationSpecInfo,
    ModelDeployment,
    RegisterInstanceRequest,
)
from kv_cache_manager.protocol.protobuf.meta_service_pb2_grpc import MetaServiceStub


class DebugServiceGrpcTest(TestBase, unittest.TestCase):
    def setUp(self):
        self.init_default()
        rpc_port = self.worker_manager.get_worker(0).env.rpc_port
        self.channel = grpc.insecure_channel(f"localhost:{rpc_port}")

    def tearDown(self):
        self.channel.close()
        self.cleanup()

    def test_get_cache_meta_detail(self):
        meta = MetaServiceStub(self.channel)
        debug = DebugServiceStub(self.channel)
        self.assertFalse(hasattr(meta, "GetCacheMetaDetail"))

        register = meta.RegisterInstance(RegisterInstanceRequest(
            trace_id="debug_meta_grpc",
            instance_group="default",
            instance_id="debug_meta_instance",
            block_size=128,
            model_deployment=ModelDeployment(
                model_name="test_model", dtype="FP8", use_mla=False,
                tp_size=1, dp_size=1, pp_size=1,
            ),
            location_spec_infos=[LocationSpecInfo(name="tp0", size=1024)],
        ), timeout=5)
        self.assertEqual(register.header.status.code, 1, register)

        response = debug.GetCacheMetaDetail(GetCacheMetaDetailRequest(
            trace_id="debug_meta_grpc",
            instance_id="debug_meta_instance",
            block_keys=[999],
            detail_level=1,
        ), timeout=5)
        self.assertEqual(response.header.status.code, 1, response)
        self.assertEqual(len(response.items), 1)
        self.assertEqual(response.items[0].block_key, 999)
        self.assertEqual(response.items[0].status.code, 1)
        self.assertEqual(response.items[0].locations[0].status, 0)


if __name__ == "__main__":
    unittest.main()
