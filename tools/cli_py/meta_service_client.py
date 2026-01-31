import asyncio
import functools
import logging
import os
import sys
import socket
from typing import Any, AsyncGenerator, Optional

import grpc
from grpc import StatusCode

from kv_cache_manager.protocol.protobuf.meta_service_pb2 import (
    CommonResponse,
    RegisterInstanceRequest,
    GetInstanceInfoRequest,
    GetInstanceInfoResponse,
    GetCacheLocationRequest,
    StartWriteCacheRequest,
    FinishWriteCacheRequest,
)

from kv_cache_manager.protocol.protobuf.meta_service_pb2_grpc import MetaServiceStub


class MetaServiceClient(object):
    def __init__(self, address: Optional[str] = None):
        self._addresses = address
        if not self._addresses:
            self._addresses = f"localhost:39681"
        self._channel = grpc.insecure_channel(self._addresses)
        logging.info(f"MetaServiceClient connect to addresses: {self._addresses}")

    '''
        # AddStorageRequest构造示例
        request = RegisterInstanceRequest(
            trace_id="trace-123456",
            instance_group="default_group",
            instance_id="instance_001",
            block_size=1024,
            model_deployment=ModelDeployment(
                model_name="my_model",
                dtype="FP8",
                use_mla=True,
                tp_size=8,
                dp_size=2,
                lora_name="lora_hash_123",
                pp_size=3,
                extra="extra_info",
                user_data="custom_user_data",
            ),
            location_spec_infos=[
                LocationSpecInfo(name="tp0", size=1024)
            ]
        )
    '''

    def register_instance(self, request: RegisterInstanceRequest) -> CommonResponse:
        stub = MetaServiceStub(self._channel)
        response = stub.RegisterInstance(request)
        print("[RegisterInstanceResponse]:", str(response))
        return response

    def get_instance_info(self, request: Optional[GetInstanceInfoRequest] = None) -> GetInstanceInfoResponse:
        stub = MetaServiceStub(self._channel)
        response = stub.GetInstanceInfo(request)
        print("[GetInstanceInfoResponse]:", str(response))
        return response
