import asyncio
import functools
import logging
import os
import sys
import socket
from typing import Any, AsyncGenerator, Optional

import grpc
from grpc import StatusCode

from kv_cache_manager.protocol.protobuf.debug_service_pb2 import (
    PingRequest,
    PingResponse,
)

from kv_cache_manager.protocol.protobuf.debug_service_pb2_grpc import DebugServiceStub


class KvCacheManagerRpcClient(object):
    def __init__(self, address: Optional[str] = None):
        self._addresses = address
        if not self._addresses:
            self._addresses = f"localhost:39681"
        self._channel = grpc.insecure_channel(self._addresses)
        # TODO 创建到服务器的连接
        logging.info(f"client connect to rpc addresses: {self._addresses}")

    def ping(self, message: Optional[str] = None) -> Optional[str]:
        stub = DebugServiceStub(self._channel)
        if message is None:
            message = "ping from" + socket.gethostbyname(socket.gethostname())
        request = PingRequest(message=message)
        response = stub.Ping(request)
        print("[ping reponse]:", str(response))


if __name__ == '__main__':
    KvCacheManagerRpcClient().ping()
