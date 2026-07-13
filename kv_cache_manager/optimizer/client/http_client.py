"""HTTP client SDK for Online Optimizer.

``OptimizerHttpClient`` implements the shared ``OptimizerClientBase``
interface defined in ``base.py``, so it is interchangeable with
``OptimizerGrpcClient``. Requests are still built as protobuf messages (reusing
the same generated types as the gRPC client) and serialized to JSON via
``google.protobuf.json_format``; responses are parsed back from JSON into the
*same* protobuf response message types via ``ParseDict``, so callers get
identical response objects regardless of transport.
"""

from typing import Iterable, Optional, Sequence

import requests
from google.protobuf.json_format import MessageToDict, ParseDict
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from kv_cache_manager.protocol.protobuf import optimizer_service_pb2 as pb2

from .base import (
    OptimizerClientBase,
    OptimizerClientConfig,
    OptimizerClientInitParams,
)


_RETRYABLE_STATUS_CODES = frozenset([502, 503, 504])


class OptimizerHttpClient(OptimizerClientBase):
    """Synchronous HTTP (REST) client for ``OptimizerService``.

    Mirrors ``OptimizerGrpcClient``'s construction style and RPC surface:
    - ``Create(config, init_params)`` is the primary construction entry.
    - ``close`` and context manager are supported for resource cleanup.
    - Every RPC returns the same protobuf response message type as the gRPC
      client (parsed from the REST JSON body via ``ParseDict``).
    """

    API_PREFIX = "/api/optimizer"

    @staticmethod
    def Create(config=None, init_params: Optional[OptimizerClientInitParams] = None):
        return OptimizerHttpClient(config=config, init_params=init_params)

    def __init__(
        self,
        config=None,
        init_params: Optional[OptimizerClientInitParams] = None,
    ):
        client_config = self._parse_config(config)
        self._init_params = init_params or OptimizerClientInitParams()
        self._base_url = self._normalize_address(client_config.address)
        self._timeout = (client_config.connection_timeout, client_config.timeout)
        self._max_retries = max(0, int(client_config.max_retries))
        self._retry_backoff_seconds = max(0.0, float(client_config.retry_backoff_seconds))
        self._session = self._build_session()

    @property
    def address(self) -> str:
        return self._base_url

    @staticmethod
    def _normalize_address(address: str) -> str:
        if not address.startswith("http://") and not address.startswith("https://"):
            address = f"http://{address}"
        return address.rstrip("/")

    def _build_session(self) -> requests.Session:
        session = requests.Session()
        retry_strategy = Retry(
            total=self._max_retries,
            backoff_factor=self._retry_backoff_seconds,
            status_forcelist=_RETRYABLE_STATUS_CODES,
            allowed_methods=frozenset(["POST"]),
        )
        adapter = HTTPAdapter(
            pool_connections=32,
            pool_maxsize=64,
            max_retries=retry_strategy,
        )
        session.mount("http://", adapter)
        session.mount("https://", adapter)
        session.headers.update({"Content-Type": "application/json"})
        return session

    @staticmethod
    def _request_to_dict(request) -> dict:
        return MessageToDict(
            request,
            including_default_value_fields=True,
            preserving_proto_field_name=True,
        )

    def _post(self, path: str, request, response_class):
        url = f"{self._base_url}{self.API_PREFIX}/{path}"
        payload = self._request_to_dict(request)
        response = self._session.post(url, json=payload, timeout=self._timeout)
        response.raise_for_status()
        return ParseDict(response.json(), response_class())

    def _call(self, operation: str, path: str, request, response_class,
               ok_codes: Sequence[int] = (pb2.OK,), check_response: bool = True):
        response = self._post(path, request, response_class)
        if check_response:
            self._check_response(operation, response, ok_codes)
        return response

    # ------------------------------------------------------------------
    # InstanceGroup CRUD
    # ------------------------------------------------------------------
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
        ok_codes = (pb2.OK, pb2.DUPLICATE_ENTITY) if allow_duplicate else (pb2.OK,)
        return self._call("CreateInstanceGroup", "createInstanceGroup", request,
                           pb2.CommonResponse, ok_codes)

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
        return self._call("UpdateInstanceGroup", "updateInstanceGroup", request, pb2.CommonResponse)

    def remove_instance_group(self, name: str, trace_id: Optional[str] = None):
        request = pb2.RemoveInstanceGroupRequest(trace_id=trace_id or self.new_trace_id(), name=name)
        return self._call("RemoveInstanceGroup", "removeInstanceGroup", request, pb2.CommonResponse)

    def get_instance_group(self, name: str, trace_id: Optional[str] = None):
        request = pb2.GetInstanceGroupRequest(trace_id=trace_id or self.new_trace_id(), name=name)
        return self._call("GetInstanceGroup", "getInstanceGroup", request, pb2.GetInstanceGroupResponse)

    def list_instance_groups(self, trace_id: Optional[str] = None):
        request = pb2.ListInstanceGroupsRequest(trace_id=trace_id or self.new_trace_id())
        return self._call("ListInstanceGroups", "listInstanceGroups", request, pb2.ListInstanceGroupsResponse)

    # ------------------------------------------------------------------
    # Instance management
    # ------------------------------------------------------------------
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
        specs = list(location_spec_infos or []) or [
            pb2.LocationSpecInfo(name="default", size=spec_size)]
        groups = list(location_spec_groups or []) or [
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
        ok_codes = (pb2.OK, pb2.DUPLICATE_ENTITY) if allow_duplicate else (pb2.OK,)
        return self._call("RegisterInstance", "registerInstance", request,
                           pb2.OptimizerRegisterInstanceResponse, ok_codes)

    def remove_instance(self, instance_id: str, trace_id: Optional[str] = None):
        request = pb2.OptimizerRemoveInstanceRequest(
            trace_id=trace_id or self.new_trace_id(), instance_id=instance_id)
        return self._call("RemoveInstance", "removeInstance", request,
                           pb2.OptimizerRemoveInstanceResponse)

    def get_instance(self, instance_id: str, trace_id: Optional[str] = None):
        request = pb2.OptimizerGetInstanceRequest(
            trace_id=trace_id or self.new_trace_id(), instance_id=instance_id)
        return self._call("GetInstance", "getInstance", request, pb2.OptimizerGetInstanceResponse)

    def list_instances(self, instance_group: str = "", trace_id: Optional[str] = None):
        request = pb2.OptimizerListInstancesRequest(
            trace_id=trace_id or self.new_trace_id(), instance_group=instance_group)
        return self._call("ListInstances", "listInstances", request,
                           pb2.OptimizerListInstancesResponse)

    # ------------------------------------------------------------------
    # TraceQuery / ResetStats
    # ------------------------------------------------------------------
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
        return self._call("TraceQuery", "traceQuery", request, pb2.TraceQueryResponse)

    def reset_stats(self, instance_id: str, trace_id: Optional[str] = None):
        request = pb2.OptimizerResetStatsRequest(
            trace_id=trace_id or self.new_trace_id(), instance_id=instance_id)
        return self._call("ResetStats", "resetStats", request, pb2.OptimizerResetStatsResponse)

    def close(self):
        if self._session:
            self._session.close()
            self._session = None
