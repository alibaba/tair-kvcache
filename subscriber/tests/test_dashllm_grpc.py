from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock

from subscriber.engine.dashllm_grpc import DashllmGrpcClient
from subscriber.proto import kv_cache_group_metadata_pb2


class _FakeGrpcChannel:
    def __init__(self, response: object) -> None:
        self._response = response
        self.path: str | None = None
        self.requests: list[object] = []
        self.timeouts: list[float] = []
        self.close = AsyncMock()

    def unary_unary(self, path: str, **_: Any) -> Any:
        self.path = path

        async def invoke(request: object, *, timeout: float) -> object:
            self.requests.append(request)
            self.timeouts.append(timeout)
            return self._response

        return invoke


async def test_metadata_requests_reuse_one_channel(mocker: Any) -> None:
    response = kv_cache_group_metadata_pb2.KvCacheGroupListPB()
    channel = _FakeGrpcChannel(response)
    insecure_channel = mocker.patch("grpc.aio.insecure_channel", return_value=channel)
    client = DashllmGrpcClient("127.0.0.1:18002")

    assert await client.get_kv_cache_group_metadata(1.0) is response
    assert await client.get_kv_cache_group_metadata(2.0) is response

    insecure_channel.assert_called_once_with("127.0.0.1:18002")
    assert channel.path == "/RpcService/GetKvCacheGroupsMetadata"
    assert len(channel.requests) == 2
    assert channel.timeouts == [1.0, 2.0]


async def test_close_releases_created_channel_once(mocker: Any) -> None:
    response = kv_cache_group_metadata_pb2.KvCacheGroupListPB()
    channel = _FakeGrpcChannel(response)
    mocker.patch("grpc.aio.insecure_channel", return_value=channel)
    client = DashllmGrpcClient("127.0.0.1:18002")

    await client.get_kv_cache_group_metadata(1.0)
    await client.close()
    await client.close()

    channel.close.assert_awaited_once()
