"""Reusable gRPC client for local DashLLM control-plane RPCs."""

from __future__ import annotations

from typing import cast

import grpc

from subscriber.proto import (
    kv_cache_group_metadata_pb2,
    kv_cache_group_metadata_pb2_grpc,
)


class DashllmGrpcClient:
    """Own one asyncio-loop-local DashLLM gRPC channel and its stubs."""

    def __init__(self, target: str) -> None:
        self._target = target
        self._channel: grpc.aio.Channel | None = None
        self._rpc_service: kv_cache_group_metadata_pb2_grpc.RpcServiceStub | None = None

    def _get_rpc_service(self) -> kv_cache_group_metadata_pb2_grpc.RpcServiceStub:
        if self._rpc_service is None:
            self._channel = grpc.aio.insecure_channel(self._target)
            self._rpc_service = kv_cache_group_metadata_pb2_grpc.RpcServiceStub(
                self._channel
            )
        return self._rpc_service

    async def get_kv_cache_group_metadata(self, timeout_s: float) -> object:
        """Fetch the vLLM KV cache group metadata response."""

        response = await self._get_rpc_service().GetKvCacheGroupsMetadata(
            kv_cache_group_metadata_pb2.KvCacheGroupsRequestPB(), timeout=timeout_s
        )
        return cast(object, response)

    async def close(self) -> None:
        """Release the channel when the owning adapter stops."""

        channel = self._channel
        self._channel = None
        self._rpc_service = None
        if channel is not None:
            await channel.close()
