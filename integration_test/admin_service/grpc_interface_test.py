import grpc
from google.protobuf.json_format import MessageToDict, ParseDict

from kv_cache_manager.protocol.protobuf.admin_service_pb2 import (
    AddStorageRequest,
    EnableStorageRequest,
    DisableStorageRequest,
    RemoveStorageRequest,
    UpdateStorageRequest,
    ListStorageRequest,
    CreateInstanceGroupRequest,
    UpdateInstanceGroupRequest,
    RemoveInstanceGroupRequest,
    GetInstanceGroupRequest,
    GetCacheMetaRequest,
    RemoveCacheRequest,
    RegisterInstanceRequest,
    RemoveInstanceRequest,
    GetInstanceInfoRequest,
    ListInstanceInfoRequest,
    AddAccountRequest,
    DeleteAccountRequest,
    ListAccountRequest,
    GenConfigSnapshotRequest,
    LoadConfigSnapshotRequest,
    GetMetricsRequest,
    CheckHealthRequest,
    CheckHealthResponse,
    GetManagerClusterInfoRequest,
    GetManagerClusterInfoResponse,
    LeaderDemoteRequest,
    GetLeaderElectorConfigRequest,
    GetLeaderElectorConfigResponse,
    UpdateLeaderElectorConfigRequest,
)
from kv_cache_manager.protocol.protobuf.admin_service_pb2_grpc import AdminServiceStub

import integration_test.admin_service.admin_interface_cases as cases


class AdminServiceGrpcClient(cases.AdminServiceClientBase):
    def __init__(self, address: str):
        self._address = address
        self._channel = grpc.insecure_channel(self._address)
        self._stub = AdminServiceStub(self._channel)

    def _convert_dict_to_proto(self, proto_class, data):
        return ParseDict(data, proto_class())

    def _convert_proto_to_dict(self, proto):
        return MessageToDict(proto, including_default_value_fields=True, preserving_proto_field_name=True)

    def add_storage(self, data, check_response=True):
        request = self._convert_dict_to_proto(AddStorageRequest, data)
        response = self._stub.AddStorage(request)
        response_dict = self._convert_proto_to_dict(response)
        if check_response:
            if response_dict['header']['status']['code'] != "OK":
                raise AssertionError(f"add_storage failed: {response_dict['header']['status']['message']}")
        return response_dict

    def enable_storage(self, data, check_response=True):
        request = self._convert_dict_to_proto(EnableStorageRequest, data)
        response = self._stub.EnableStorage(request)
        response_dict = self._convert_proto_to_dict(response)
        if check_response:
            if response_dict['header']['status']['code'] != "OK":
                raise AssertionError(f"enable_storage failed: {response_dict['header']['status']['message']}")
        return response_dict

    def disable_storage(self, data, check_response=True):
        request = self._convert_dict_to_proto(DisableStorageRequest, data)
        response = self._stub.DisableStorage(request)
        response_dict = self._convert_proto_to_dict(response)
        if check_response:
            if response_dict['header']['status']['code'] != "OK":
                raise AssertionError(f"disable_storage failed: {response_dict['header']['status']['message']}")
        return response_dict

    def remove_storage(self, data, check_response=True):
        request = self._convert_dict_to_proto(RemoveStorageRequest, data)
        response = self._stub.RemoveStorage(request)
        response_dict = self._convert_proto_to_dict(response)
        if check_response:
            if response_dict['header']['status']['code'] != "OK":
                raise AssertionError(f"remove_storage failed: {response_dict['header']['status']['message']}")
        return response_dict

    def update_storage(self, data, check_response=True):
        request = self._convert_dict_to_proto(UpdateStorageRequest, data)
        response = self._stub.UpdateStorage(request)
        response_dict = self._convert_proto_to_dict(response)
        if check_response:
            if response_dict['header']['status']['code'] != "OK":
                raise AssertionError(f"update_storage failed: {response_dict['header']['status']['message']}")
        return response_dict

    def list_storage(self, data, check_response=True):
        request = self._convert_dict_to_proto(ListStorageRequest, data)
        response = self._stub.ListStorage(request)
        response_dict = self._convert_proto_to_dict(response)
        if check_response:
            header = response_dict.get('header')
            if not header or header['status']['code'] != "OK":
                msg = header['status']['message'] if header else "no header"
                raise AssertionError(f"list_storage failed: {msg}")
        return response_dict

    def create_instance_group(self, data, check_response=True):
        request = self._convert_dict_to_proto(CreateInstanceGroupRequest, data)
        response = self._stub.CreateInstanceGroup(request)
        response_dict = self._convert_proto_to_dict(response)
        if check_response:
            if response_dict['header']['status']['code'] != "OK":
                raise AssertionError(f"create_instance_group failed: {response_dict['header']['status']['message']}")
        return response_dict

    def update_instance_group(self, data, check_response=True):
        request = self._convert_dict_to_proto(UpdateInstanceGroupRequest, data)
        response = self._stub.UpdateInstanceGroup(request)
        response_dict = self._convert_proto_to_dict(response)
        if check_response:
            if response_dict['header']['status']['code'] != "OK":
                raise AssertionError(f"update_instance_group failed: {response_dict['header']['status']['message']}")
        return response_dict

    def remove_instance_group(self, data, check_response=True):
        request = self._convert_dict_to_proto(RemoveInstanceGroupRequest, data)
        response = self._stub.RemoveInstanceGroup(request)
        response_dict = self._convert_proto_to_dict(response)
        if check_response:
            if response_dict['header']['status']['code'] != "OK":
                raise AssertionError(f"remove_instance_group failed: {response_dict['header']['status']['message']}")
        return response_dict

    def get_instance_group(self, data, check_response=True):
        request = self._convert_dict_to_proto(GetInstanceGroupRequest, data)
        response = self._stub.GetInstanceGroup(request)
        response_dict = self._convert_proto_to_dict(response)
        if check_response:
            header = response_dict.get('header')
            if not header or header['status']['code'] != "OK":
                msg = header['status']['message'] if header else "no header"
                raise AssertionError(f"get_instance_group failed: {msg}")
        return response_dict

    def get_cache_meta(self, data, check_response=True):
        request = self._convert_dict_to_proto(GetCacheMetaRequest, data)
        response = self._stub.GetCacheMeta(request)
        response_dict = self._convert_proto_to_dict(response)
        if check_response:
            header = response_dict.get('header')
            if not header or header['status']['code'] != "OK":
                msg = header['status']['message'] if header else "no header"
                raise AssertionError(f"get_cache_meta failed: {msg}")
        return response_dict

    def remove_cache(self, data, check_response=True):
        request = self._convert_dict_to_proto(RemoveCacheRequest, data)
        response = self._stub.RemoveCache(request)
        response_dict = self._convert_proto_to_dict(response)
        if check_response:
            header = response_dict.get('header')
            if not header or header['status']['code'] != "OK":
                msg = header['status']['message'] if header else "no header"
                raise AssertionError(f"remove_cache failed: {msg}")
        return response_dict

    def register_instance(self, data, check_response=True):
        request = self._convert_dict_to_proto(RegisterInstanceRequest, data)
        response = self._stub.RegisterInstance(request)
        response_dict = self._convert_proto_to_dict(response)
        if check_response:
            if response_dict['header']['status']['code'] != "OK":
                raise AssertionError(f"register_instance failed: {response_dict['header']['status']['message']}")
        return response_dict

    def remove_instance(self, data, check_response=True):
        request = self._convert_dict_to_proto(RemoveInstanceRequest, data)
        response = self._stub.RemoveInstance(request)
        response_dict = self._convert_proto_to_dict(response)
        if check_response:
            if response_dict['header']['status']['code'] != "OK":
                raise AssertionError(f"remove_instance failed: {response_dict['header']['status']['message']}")
        return response_dict

    def get_instance_info(self, data, check_response=True):
        request = self._convert_dict_to_proto(GetInstanceInfoRequest, data)
        response = self._stub.GetInstanceInfo(request)
        response_dict = self._convert_proto_to_dict(response)
        if check_response:
            header = response_dict.get('header')
            if not header or header['status']['code'] != "OK":
                msg = header['status']['message'] if header else "no header"
                raise AssertionError(f"get_instance_info failed: {msg}")
        return response_dict

    def list_instance_info(self, data, check_response=True):
        request = self._convert_dict_to_proto(ListInstanceInfoRequest, data)
        response = self._stub.ListInstanceInfo(request)
        response_dict = self._convert_proto_to_dict(response)
        if check_response:
            header = response_dict.get('header')
            if not header or header['status']['code'] != "OK":
                msg = header['status']['message'] if header else "no header"
                raise AssertionError(f"list_instance_info failed: {msg}")
        return response_dict

    def add_account(self, data, check_response=True):
        request = self._convert_dict_to_proto(AddAccountRequest, data)
        response = self._stub.AddAccount(request)
        response_dict = self._convert_proto_to_dict(response)
        if check_response:
            if response_dict['header']['status']['code'] != "OK":
                raise AssertionError(f"add_account failed: {response_dict['header']['status']['message']}")
        return response_dict

    def delete_account(self, data, check_response=True):
        request = self._convert_dict_to_proto(DeleteAccountRequest, data)
        response = self._stub.DeleteAccount(request)
        response_dict = self._convert_proto_to_dict(response)
        if check_response:
            if response_dict['header']['status']['code'] != "OK":
                raise AssertionError(f"delete_account failed: {response_dict['header']['status']['message']}")
        return response_dict

    def list_account(self, data, check_response=True):
        request = self._convert_dict_to_proto(ListAccountRequest, data)
        response = self._stub.ListAccount(request)
        response_dict = self._convert_proto_to_dict(response)
        return response_dict

    def gen_config_snapshot(self, data, check_response=True):
        request = self._convert_dict_to_proto(GenConfigSnapshotRequest, data)
        response = self._stub.GenConfigSnapshot(request)
        response_dict = self._convert_proto_to_dict(response)
        header = response_dict.get('header')
        if check_response and (not header or header['status']['code'] != "OK"):
            msg = header['status']['message'] if header else "no header"
            raise AssertionError(f"gen_config_snapshot failed: {msg}")
        return response_dict

    def load_config_snapshot(self, data, check_response=True):
        request = self._convert_dict_to_proto(LoadConfigSnapshotRequest, data)
        response = self._stub.LoadConfigSnapshot(request)
        response_dict = self._convert_proto_to_dict(response)
        if check_response:
            if response_dict['header']['status']['code'] != "OK":
                raise AssertionError(f"load_config_snapshot failed: {response_dict['header']['status']['message']}")
        return response_dict

    def get_metrics(self, data, check_response=True):
        request = self._convert_dict_to_proto(GetMetricsRequest, data)
        response = self._stub.GetMetrics(request)
        response_dict = self._convert_proto_to_dict(response)
        header = response_dict.get('header')
        if check_response and (not header or header['status']['code'] != "OK"):
            msg = header['status']['message'] if header else "no header"
            raise AssertionError(f"get_metrics failed: {msg}")
        return response_dict

    def check_health(self, data, check_response=True):
        request = self._convert_dict_to_proto(CheckHealthRequest, data)
        response = self._stub.CheckHealth(request)
        response_dict = self._convert_proto_to_dict(response)
        if check_response:
            header = response_dict.get('header')
            if not header or header['status']['code'] != "OK":
                msg = header['status']['message'] if header else "no header"
                raise AssertionError(f"check_health failed: {msg}")
        return response_dict

    def get_manager_cluster_info(self, data, check_response=True):
        request = self._convert_dict_to_proto(GetManagerClusterInfoRequest, data)
        response = self._stub.GetManagerClusterInfo(request)
        response_dict = self._convert_proto_to_dict(response)
        if check_response:
            header = response_dict.get('header')
            if not header or header['status']['code'] != "OK":
                msg = header['status']['message'] if header else "no header"
                raise AssertionError(f"get_manager_cluster_info failed: {msg}")
        return response_dict

    def leader_demote(self, data, check_response=True):
        request = self._convert_dict_to_proto(LeaderDemoteRequest, data)
        response = self._stub.LeaderDemote(request)
        response_dict = self._convert_proto_to_dict(response)
        if check_response:
            if response_dict['header']['status']['code'] != "OK":
                raise AssertionError(f"leader_demote failed: {response_dict['header']['status']['message']}")
        return response_dict

    def get_leader_elector_config(self, data, check_response=True):
        request = self._convert_dict_to_proto(GetLeaderElectorConfigRequest, data)
        response = self._stub.GetLeaderElectorConfig(request)
        response_dict = self._convert_proto_to_dict(response)
        if check_response:
            header = response_dict.get('header')
            if not header or header['status']['code'] != "OK":
                msg = header['status']['message'] if header else "no header"
                raise AssertionError(f"get_leader_elector_config failed: {msg}")
        return response_dict

    def update_leader_elector_config(self, data, check_response=True):
        request = self._convert_dict_to_proto(UpdateLeaderElectorConfigRequest, data)
        response = self._stub.UpdateLeaderElectorConfig(request)
        response_dict = self._convert_proto_to_dict(response)
        if check_response:
            if response_dict['header']['status']['code'] != "OK":
                raise AssertionError(f"update_leader_elector_config failed: {response_dict['header']['status']['message']}")
        return response_dict

    def close(self):
        if self._channel:
            self._channel.close()


class AdminServiceGrpcTest(cases.AdminServiceTestBase):
    def _get_manager_client(self):
        self._rpc_port = self.worker_manager.get_worker(0).env.admin_rpc_port
        self._rpc_address = f"localhost:{self._rpc_port}"
        return AdminServiceGrpcClient(self._rpc_address)

class AdminServiceGrpcLeaderElectionTest(cases.AdminServiceLeaderElectionTest):
    """gRPC版本的选主测试"""

    def _get_manager_client(self, worker_id: int):
        worker = self.worker_manager.get_worker(worker_id)
        rpc_port = worker.env.admin_rpc_port
        rpc_address = f"localhost:{rpc_port}"
        return AdminServiceGrpcClient(rpc_address)


if __name__ == "__main__":
    import unittest
    unittest.main()
