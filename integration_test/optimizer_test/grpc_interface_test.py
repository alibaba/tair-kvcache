"""gRPC integration tests for the Online Optimizer service."""

import grpc
from google.protobuf.json_format import MessageToDict, ParseDict

from kv_cache_manager.protocol.protobuf.optimizer_service_pb2 import (
    CreateInstanceGroupRequest,
    UpdateInstanceGroupRequest,
    RemoveInstanceGroupRequest,
    GetInstanceGroupRequest,
    ListInstanceGroupsRequest,
    OptimizerRegisterInstanceRequest,
    OptimizerRemoveInstanceRequest,
    OptimizerGetInstanceRequest,
    TraceQueryRequest,
    OptimizerListInstancesRequest,
    OptimizerResetStatsRequest,
)
from kv_cache_manager.protocol.protobuf.optimizer_service_pb2_grpc import OptimizerServiceStub

import integration_test.optimizer_test.optimizer_interface_cases as cases


class OptimizerServiceGrpcClient(cases.OptimizerServiceClientBase):
    """gRPC client for Optimizer Service."""

    DEFAULT_TIMEOUT = 5  # seconds

    def __init__(self, address: str, timeout: int = None):
        self._address = address
        self._channel = grpc.insecure_channel(self._address)
        self._stub = OptimizerServiceStub(self._channel)
        self._timeout = timeout if timeout is not None else self.DEFAULT_TIMEOUT

    def _to_proto(self, proto_class, data):
        return ParseDict(data, proto_class())

    def _to_dict(self, proto):
        return MessageToDict(proto, including_default_value_fields=True,
                             preserving_proto_field_name=True)

    def _call(self, method_name, request_class, data, check_response=True):
        request = self._to_proto(request_class, data)
        method = getattr(self._stub, method_name)
        response = method(request, timeout=self._timeout)
        response_dict = self._to_dict(response)
        if check_response:
            header = response_dict.get('header', {})
            status = header.get('status', {})
            if status.get('code') != "OK":
                msg = status.get('message', 'unknown error')
                raise AssertionError(f"{method_name} failed: {msg}")
        return response_dict

    def create_instance_group(self, data, check_response=True):
        return self._call("CreateInstanceGroup", CreateInstanceGroupRequest,
                          data, check_response)

    def update_instance_group(self, data, check_response=True):
        return self._call("UpdateInstanceGroup", UpdateInstanceGroupRequest,
                          data, check_response)

    def remove_instance_group(self, data, check_response=True):
        return self._call("RemoveInstanceGroup", RemoveInstanceGroupRequest,
                          data, check_response)

    def get_instance_group(self, data, check_response=True):
        return self._call("GetInstanceGroup", GetInstanceGroupRequest,
                          data, check_response)

    def list_instance_groups(self, data, check_response=True):
        return self._call("ListInstanceGroups", ListInstanceGroupsRequest,
                          data, check_response)

    def register_instance(self, data, check_response=True):
        return self._call("RegisterInstance", OptimizerRegisterInstanceRequest,
                          data, check_response)

    def remove_instance(self, data, check_response=True):
        return self._call("RemoveInstance", OptimizerRemoveInstanceRequest,
                          data, check_response)

    def get_instance(self, data, check_response=True):
        return self._call("GetInstance", OptimizerGetInstanceRequest,
                          data, check_response)

    def trace_query(self, data, check_response=True):
        return self._call("TraceQuery", TraceQueryRequest,
                          data, check_response)

    def list_instances(self, data, check_response=True):
        return self._call("ListInstances", OptimizerListInstancesRequest,
                          data, check_response)

    def reset_stats(self, data, check_response=True):
        return self._call("ResetStats", OptimizerResetStatsRequest,
                          data, check_response)

    def close(self):
        if self._channel:
            self._channel.close()


class OptimizerServiceGrpcTest(cases.OptimizerServiceTestCases):
    """gRPC version of the optimizer service integration tests."""

    def _create_client(self):
        rpc_port = self._server_manager.rpc_port
        address = f"127.0.0.1:{rpc_port}"
        return OptimizerServiceGrpcClient(address)


if __name__ == "__main__":
    import unittest
    unittest.main()
