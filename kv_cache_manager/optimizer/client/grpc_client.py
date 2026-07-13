"""gRPC client SDK for Online Optimizer.

This module is intended for external users of Online Optimizer.  It exposes
both complete proto-level RPC methods and convenience helpers used by the
benchmark tool. ``OptimizerGrpcClient`` implements the shared
``OptimizerClientBase`` interface defined in ``base.py``, so it is
interchangeable with ``OptimizerHttpClient``.
"""

import time
from typing import Iterable, Optional, Sequence
from urllib.parse import urlparse

import grpc
from google.protobuf.json_format import MessageToDict, ParseDict

from kv_cache_manager.protocol.protobuf import optimizer_service_pb2 as pb2
from kv_cache_manager.protocol.protobuf.optimizer_service_pb2_grpc import OptimizerServiceStub

from .base import (
    OptimizerClientBase,
    OptimizerClientConfig,
    OptimizerClientInitParams,
)


_TRANSIENT_GRPC_CODES = frozenset([
    grpc.StatusCode.UNAVAILABLE,
    grpc.StatusCode.DEADLINE_EXCEEDED,
    grpc.StatusCode.RESOURCE_EXHAUSTED,
])


class OptimizerGrpcClient(OptimizerClientBase):
    """Synchronous gRPC client for ``OptimizerService``.

    The client follows the existing KVCacheManager client style:
    - ``Create(config, init_params)`` is the primary construction entry.
    - ``close`` and context manager are supported for resource cleanup.
    - proto request methods cover all RPCs, while convenience helpers keep
      common Online Optimizer calls compact.
    """

    @staticmethod
    def Create(config=None, init_params: Optional[OptimizerClientInitParams] = None):
        return OptimizerGrpcClient(config=config, init_params=init_params)

    def __init__(
        self,
        config=None,
        init_params: Optional[OptimizerClientInitParams] = None,
    ):
        client_config = self._parse_config(config)
        self._init_params = init_params or OptimizerClientInitParams()
        self._address = self._normalize_address(client_config.address)
        self._timeout = client_config.timeout
        self._max_retries = max(0, int(client_config.max_retries))
        self._retry_backoff_seconds = max(0.0, float(client_config.retry_backoff_seconds))
        self._wait_for_ready = client_config.wait_for_ready
        self._channel = grpc.insecure_channel(self._address, options=client_config.options)
        self._stub = OptimizerServiceStub(self._channel)

    @property
    def address(self) -> str:
        return self._address

    @staticmethod
    def _normalize_address(address: str) -> str:
        if address.startswith("http://") or address.startswith("https://"):
            parsed = urlparse(address)
            return parsed.netloc
        return address

    @staticmethod
    def _to_proto(proto_class, data):
        if isinstance(data, proto_class):
            return data
        return ParseDict(data, proto_class())

    @staticmethod
    def to_dict(response) -> dict:
        return MessageToDict(
            response,
            including_default_value_fields=True,
            preserving_proto_field_name=True,
        )

    def _call_rpc(
        self,
        method_name: str,
        request,
        check_response: bool = True,
        ok_codes: Sequence[int] = (pb2.OK,),
        timeout: Optional[float] = None,
    ):
        method = getattr(self._stub, method_name)
        call_timeout = self._timeout if timeout is None else timeout
        last_error = None
        for attempt in range(self._max_retries + 1):
            try:
                response = method(
                    request,
                    timeout=call_timeout,
                    wait_for_ready=self._wait_for_ready,
                )
                if check_response:
                    self._check_response(method_name, response, ok_codes)
                return response
            except grpc.RpcError as exc:
                last_error = exc
                if exc.code() not in _TRANSIENT_GRPC_CODES or attempt >= self._max_retries:
                    raise
                time.sleep(self._retry_backoff_seconds * (2 ** attempt))
        raise last_error

    def create_instance_group_rpc(
        self,
        request: pb2.CreateInstanceGroupRequest,
        check_response: bool = True,
        allow_duplicate: bool = False,
        timeout: Optional[float] = None,
    ):
        ok_codes = (pb2.OK, pb2.DUPLICATE_ENTITY) if allow_duplicate else (pb2.OK,)
        return self._call_rpc("CreateInstanceGroup", request, check_response, ok_codes, timeout)

    def update_instance_group_rpc(
        self,
        request: pb2.UpdateInstanceGroupRequest,
        check_response: bool = True,
        timeout: Optional[float] = None,
    ):
        return self._call_rpc("UpdateInstanceGroup", request, check_response, timeout=timeout)

    def remove_instance_group_rpc(
        self,
        request: pb2.RemoveInstanceGroupRequest,
        check_response: bool = True,
        timeout: Optional[float] = None,
    ):
        return self._call_rpc("RemoveInstanceGroup", request, check_response, timeout=timeout)

    def get_instance_group_rpc(
        self,
        request: pb2.GetInstanceGroupRequest,
        check_response: bool = True,
        timeout: Optional[float] = None,
    ):
        return self._call_rpc("GetInstanceGroup", request, check_response, timeout=timeout)

    def list_instance_groups_rpc(
        self,
        request: pb2.ListInstanceGroupsRequest,
        check_response: bool = True,
        timeout: Optional[float] = None,
    ):
        return self._call_rpc("ListInstanceGroups", request, check_response, timeout=timeout)

    def register_instance_rpc(
        self,
        request: pb2.OptimizerRegisterInstanceRequest,
        check_response: bool = True,
        allow_duplicate: bool = False,
        timeout: Optional[float] = None,
    ):
        ok_codes = (pb2.OK, pb2.DUPLICATE_ENTITY) if allow_duplicate else (pb2.OK,)
        return self._call_rpc("RegisterInstance", request, check_response, ok_codes, timeout)

    def remove_instance_rpc(
        self,
        request: pb2.OptimizerRemoveInstanceRequest,
        check_response: bool = True,
        timeout: Optional[float] = None,
    ):
        return self._call_rpc("RemoveInstance", request, check_response, timeout=timeout)

    def get_instance_rpc(
        self,
        request: pb2.OptimizerGetInstanceRequest,
        check_response: bool = True,
        timeout: Optional[float] = None,
    ):
        return self._call_rpc("GetInstance", request, check_response, timeout=timeout)

    def list_instances_rpc(
        self,
        request: pb2.OptimizerListInstancesRequest,
        check_response: bool = True,
        timeout: Optional[float] = None,
    ):
        return self._call_rpc("ListInstances", request, check_response, timeout=timeout)

    def trace_query_rpc(
        self,
        request: pb2.TraceQueryRequest,
        check_response: bool = True,
        timeout: Optional[float] = None,
    ):
        return self._call_rpc("TraceQuery", request, check_response, timeout=timeout)

    def reset_stats_rpc(
        self,
        request: pb2.OptimizerResetStatsRequest,
        check_response: bool = True,
        timeout: Optional[float] = None,
    ):
        return self._call_rpc("ResetStats", request, check_response, timeout=timeout)

    def create_instance_group_from_dict(self, data: dict, check_response: bool = True):
        return self.create_instance_group_rpc(
            self._to_proto(pb2.CreateInstanceGroupRequest, data), check_response)

    def update_instance_group_from_dict(self, data: dict, check_response: bool = True):
        return self.update_instance_group_rpc(
            self._to_proto(pb2.UpdateInstanceGroupRequest, data), check_response)

    def remove_instance_group_from_dict(self, data: dict, check_response: bool = True):
        return self.remove_instance_group_rpc(
            self._to_proto(pb2.RemoveInstanceGroupRequest, data), check_response)

    def get_instance_group_from_dict(self, data: dict, check_response: bool = True):
        return self.get_instance_group_rpc(
            self._to_proto(pb2.GetInstanceGroupRequest, data), check_response)

    def list_instance_groups_from_dict(self, data: dict, check_response: bool = True):
        return self.list_instance_groups_rpc(
            self._to_proto(pb2.ListInstanceGroupsRequest, data), check_response)

    def register_instance_from_dict(self, data: dict, check_response: bool = True):
        return self.register_instance_rpc(
            self._to_proto(pb2.OptimizerRegisterInstanceRequest, data), check_response)

    def remove_instance_from_dict(self, data: dict, check_response: bool = True):
        return self.remove_instance_rpc(
            self._to_proto(pb2.OptimizerRemoveInstanceRequest, data), check_response)

    def get_instance_from_dict(self, data: dict, check_response: bool = True):
        return self.get_instance_rpc(
            self._to_proto(pb2.OptimizerGetInstanceRequest, data), check_response)

    def list_instances_from_dict(self, data: dict, check_response: bool = True):
        return self.list_instances_rpc(
            self._to_proto(pb2.OptimizerListInstancesRequest, data), check_response)

    def trace_query_from_dict(self, data: dict, check_response: bool = True):
        return self.trace_query_rpc(
            self._to_proto(pb2.TraceQueryRequest, data), check_response)

    def reset_stats_from_dict(self, data: dict, check_response: bool = True):
        return self.reset_stats_rpc(
            self._to_proto(pb2.OptimizerResetStatsRequest, data), check_response)

    def create_instance_group(
        self,
        name: str,
        capacity_gb,
        eviction_policy=pb2.OPTIMIZER_EVICTION_POLICY_LRU,
        ttl_seconds: int = 0,
        shared_group_quota: bool = False,
        enable_theoretical_max_cache: bool = False,
        allow_duplicate: bool = True,
        trace_id: Optional[str] = None,
    ):
        caps = list(capacity_gb) if isinstance(capacity_gb, (list, tuple)) else [capacity_gb]
        request = pb2.CreateInstanceGroupRequest(
            trace_id=trace_id or self.new_trace_id(),
            instance_group=pb2.OptimizerInstanceGroupProto(
                name=name,
                eviction_policy=self._eviction_policy(eviction_policy),
                capacity_gb=caps,
                shared_group_quota=shared_group_quota,
                ttl_seconds=ttl_seconds,
                enable_theoretical_max_cache=enable_theoretical_max_cache,
            ),
        )
        return self.create_instance_group_rpc(request, allow_duplicate=allow_duplicate)

    def update_instance_group(
        self,
        name: str,
        capacity_gb,
        eviction_policy=pb2.OPTIMIZER_EVICTION_POLICY_LRU,
        ttl_seconds: int = 0,
        shared_group_quota: bool = False,
        enable_theoretical_max_cache: bool = False,
        trace_id: Optional[str] = None,
    ):
        caps = list(capacity_gb) if isinstance(capacity_gb, (list, tuple)) else [capacity_gb]
        request = pb2.UpdateInstanceGroupRequest(
            trace_id=trace_id or self.new_trace_id(),
            instance_group=pb2.OptimizerInstanceGroupProto(
                name=name,
                eviction_policy=self._eviction_policy(eviction_policy),
                capacity_gb=caps,
                shared_group_quota=shared_group_quota,
                ttl_seconds=ttl_seconds,
                enable_theoretical_max_cache=enable_theoretical_max_cache,
            ),
        )
        return self.update_instance_group_rpc(request)

    def remove_instance_group(self, name: str, trace_id: Optional[str] = None):
        request = pb2.RemoveInstanceGroupRequest(trace_id=trace_id or self.new_trace_id(), name=name)
        return self.remove_instance_group_rpc(request)

    def get_instance_group(self, name: str, trace_id: Optional[str] = None):
        request = pb2.GetInstanceGroupRequest(trace_id=trace_id or self.new_trace_id(), name=name)
        return self.get_instance_group_rpc(request)

    def list_instance_groups(self, trace_id: Optional[str] = None):
        request = pb2.ListInstanceGroupsRequest(trace_id=trace_id or self.new_trace_id())
        return self.list_instance_groups_rpc(request)

    def register_instance(
        self,
        instance_group: str,
        instance_id: str,
        block_size: int,
        block_bytes: int = 0,
        location_spec_infos: Optional[Sequence[pb2.LocationSpecInfo]] = None,
        location_spec_groups: Optional[Sequence[pb2.LocationSpecGroup]] = None,
        optimizer_state_info: Optional[pb2.OptimizerStateInfo] = None,
        linear_step: int = 0,
        allow_duplicate: bool = True,
        trace_id: Optional[str] = None,
    ):
        spec_size = block_bytes if block_bytes and block_bytes > 0 else 1
        specs = list(location_spec_infos) if location_spec_infos is not None else [
            pb2.LocationSpecInfo(name="default", size=spec_size)]
        groups = list(location_spec_groups) if location_spec_groups is not None else [
            pb2.LocationSpecGroup(name="full", spec_names=[specs[0].name])]
        state = optimizer_state_info or pb2.OptimizerStateInfo(
            full_location_spec_group_name=groups[0].name)
        request = pb2.OptimizerRegisterInstanceRequest(
            trace_id=trace_id or self.new_trace_id(),
            instance_group=instance_group,
            instance_id=instance_id,
            block_size=block_size,
            location_spec_infos=specs,
            location_spec_groups=groups,
            optimizer_state_info=state,
            linear_step=linear_step,
        )
        return self.register_instance_rpc(request, allow_duplicate=allow_duplicate)

    def remove_instance(self, instance_id: str, trace_id: Optional[str] = None):
        request = pb2.OptimizerRemoveInstanceRequest(
            trace_id=trace_id or self.new_trace_id(), instance_id=instance_id)
        return self.remove_instance_rpc(request)

    def get_instance(self, instance_id: str, trace_id: Optional[str] = None):
        request = pb2.OptimizerGetInstanceRequest(
            trace_id=trace_id or self.new_trace_id(), instance_id=instance_id)
        return self.get_instance_rpc(request)

    def list_instances(self, instance_group: str = "", trace_id: Optional[str] = None):
        request = pb2.OptimizerListInstancesRequest(
            trace_id=trace_id or self.new_trace_id(), instance_group=instance_group)
        return self.list_instances_rpc(request)

    def trace_query(
        self,
        instance_id: str,
        block_keys: Iterable[int],
        token_ids: Optional[Iterable[int]] = None,
        trace_id: Optional[str] = None,
    ):
        request = pb2.TraceQueryRequest(
            trace_id=trace_id or self.new_trace_id(),
            instance_id=instance_id,
            block_keys=list(block_keys),
            token_ids=list(token_ids or []),
        )
        return self.trace_query_rpc(request)

    def reset_stats(self, instance_id: str, trace_id: Optional[str] = None):
        request = pb2.OptimizerResetStatsRequest(
            trace_id=trace_id or self.new_trace_id(), instance_id=instance_id)
        return self.reset_stats_rpc(request)

    def close(self):
        if self._channel:
            self._channel.close()
            self._channel = None
