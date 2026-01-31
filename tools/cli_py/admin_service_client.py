import asyncio
import functools
import logging
import os
import sys
import socket
from typing import Any, AsyncGenerator, Optional

import grpc
from grpc import StatusCode

from kv_cache_manager.protocol.protobuf.admin_service_pb2 import (
    CommonResponse,
    Storage,
    MooncakeStorageSpec,
    AddStorageRequest,
    EnableStorageRequest,
    DisableStorageRequest,
    RemoveStorageRequest,
    UpdateStorageRequest,
    ListStorageRequest,
    ListStorageResponse,
)

from kv_cache_manager.protocol.protobuf.admin_service_pb2_grpc import AdminServiceStub


class AdminServiceClient(object):
    def __init__(self, address: Optional[str] = None):
        self._addresses = address
        if not self._addresses:
            self._addresses = f"localhost:39681"
        self._channel = grpc.insecure_channel(self._addresses)
        # TODO 创建到服务器的连接
        logging.info(f"DebugServiceClient connect to addresses: {self._addresses}")

    '''
        # AddStorageRequest构造示例
        mooncake = MooncakeStorageSpec(
            global_unique_name="unique_3fs_name",
            local_hostname="127.0.0.1:1111",
            metadata_connstring="/3fs/stage/3fs",
            protocol="tcp",
        )
        storage = Storage(
            mooncake=mooncake
        )
        request = AddStorageRequest(
            trace_id="trace-123456",
            storage=storage
        )
    '''

    def add_storage(self, request: AddStorageRequest) -> CommonResponse:
        stub = AdminServiceStub(self._channel)
        response = stub.AddStorage(request)
        print("[AddStorageReponse]:", str(response))
        return response

    def list_storage(self, request: Optional[ListStorageRequest] = None) -> ListStorageResponse:
        stub = AdminServiceStub(self._channel)
        if request is None:
            request = ListStorageRequest(trace_id="trace-123456")
        response = stub.ListStorage(request)
        print("[ListStorageReponse]:", str(response))
        return response
