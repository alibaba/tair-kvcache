import grpc
from google.protobuf.json_format import MessageToDict
from google.protobuf.json_format import ParseDict

# Import protobuf classes for gRPC client
from kv_cache_manager.protocol.protobuf.meta_service_pb2 import (
    RegisterInstanceRequest,
    GetInstanceInfoRequest,
    GetCacheLocationRequest,
    StartWriteCacheRequest,
    FinishWriteCacheRequest,
    RemoveCacheRequest,
    TrimCacheRequest,
    ModelDeployment,
    BlockMask,
    BoolMasksType,
    CommonResponse,
    GetInstanceInfoResponse,
    GetCacheLocationResponse,
    StartWriteCacheResponse,
)
from kv_cache_manager.protocol.protobuf.meta_service_pb2_grpc import MetaServiceStub
import integration_test.meta_service.meta_interface_cases as cases


class MetaServiceGrpcClient(cases.MetaServiceClientBase):
    """gRPC client for MetaService API endpoints"""

    def __init__(self, address):
        self._address = address
        self._channel = grpc.insecure_channel(self._address)
        self._stub = MetaServiceStub(self._channel)

    def _convert_dict_to_proto(self, proto_class, data):
        """Convert a dictionary to a protobuf message"""
        return ParseDict(data, proto_class())

    def _convert_proto_to_dict(self, proto):
        """Convert a protobuf message to a dictionary"""
        # Convert protobuf message to dict using MessageToDict
        return MessageToDict(proto, including_default_value_fields=True, preserving_proto_field_name=True)

    def register_instance(self, data, check_response=True):
        """Register an instance with the service"""
        request = self._convert_dict_to_proto(RegisterInstanceRequest, data)
        response = self._stub.RegisterInstance(request)
        response_dict = self._convert_proto_to_dict(response)
        if check_response:
            if response_dict['header']['status']['code'] != "OK":
                raise AssertionError(
                    f"Request to register_instance failed with error: {response_dict['header']['status']['message']}")
        return response_dict

    def get_instance_info(self, data, check_response=True):
        """Get information about a registered instance"""
        request = self._convert_dict_to_proto(GetInstanceInfoRequest, data)
        response = self._stub.GetInstanceInfo(request)
        response_dict = self._convert_proto_to_dict(response)
        if check_response:
            if response_dict['header']['status']['code'] != "OK":
                raise AssertionError(
                    f"Request to get_instance_info failed with error: {response_dict['header']['status']['message']}")
        return response_dict

    def get_cache_location(self, data, check_response=True):
        """Get cache location for specified block keys"""
        request = self._convert_dict_to_proto(GetCacheLocationRequest, data)
        response = self._stub.GetCacheLocation(request)
        response_dict = self._convert_proto_to_dict(response)
        if check_response:
            if response_dict['header']['status']['code'] != "OK":
                raise AssertionError(
                    f"Request to get_cache_location failed with error: {response_dict['header']['status']['message']}")
        return response_dict

    def start_write_cache(self, data, check_response=True):
        """Start writing cache data"""
        request = self._convert_dict_to_proto(StartWriteCacheRequest, data)
        response = self._stub.StartWriteCache(request)
        response_dict = self._convert_proto_to_dict(response)
        if check_response:
            if response_dict['header']['status']['code'] != "OK":
                raise AssertionError(
                    f"Request to start_write_cache failed with error: {response_dict['header']['status']['message']}")
        return response_dict

    def finish_write_cache(self, data, check_response=True):
        """Finish writing cache data"""
        request = self._convert_dict_to_proto(FinishWriteCacheRequest, data)
        response = self._stub.FinishWriteCache(request)
        response_dict = self._convert_proto_to_dict(response)
        if check_response:
            if response_dict['header']['status']['code'] != "OK":
                raise AssertionError(
                    f"Request to finish_write_cache failed with error: {response_dict['header']['status']['message']}")
        return response_dict

    def remove_cache(self, data, check_response=True):
        """Remove cache data for specified block keys"""
        request = self._convert_dict_to_proto(RemoveCacheRequest, data)
        response = self._stub.RemoveCache(request)
        response_dict = self._convert_proto_to_dict(response)
        if check_response:
            if response_dict['header']['status']['code'] != "OK":
                raise AssertionError(
                    f"Request to remove_cache failed with error: {response_dict['header']['status']['message']}")
        return response_dict

    def trim_cache(self, data, check_response=True):
        """Trim cache data based on specified strategy"""
        request = self._convert_dict_to_proto(TrimCacheRequest, data)
        response = self._stub.TrimCache(request)
        response_dict = self._convert_proto_to_dict(response)
        if check_response:
            if response_dict['header']['status']['code'] != "OK":
                raise AssertionError(
                    f"Request to trim_cache failed with error: {response_dict['header']['status']['message']}")
        return response_dict

    def close(self):
        """Close the gRPC channel"""
        if self._channel:
            self._channel.close()


class MetaServiceGrpcTest(cases.MetaServiceTestBase):
    """gRPC version of the MetaService tests"""

    def _get_manager_client(self):
        self._rpc_port = self.worker_manager.get_worker(0).env.rpc_port
        self._rpc_address = "localhost:%d" % self._rpc_port
        return MetaServiceGrpcClient(self._rpc_address)


if __name__ == "__main__":
    import unittest
    unittest.main()
