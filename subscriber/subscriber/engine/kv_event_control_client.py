"""Same-Pod UDS client for DashLLM KV-event bootstrap and snapshots."""

from __future__ import annotations

from collections.abc import Callable
from typing import cast

import grpc

from subscriber.engine.grpc_options import CHANNEL_OPTIONS
from subscriber.proto import engine_service_rpc_pb2, engine_service_rpc_pb2_grpc


class DashllmKvEventControlClient:
    """Own the local control channel independently from remote worker status."""

    def __init__(self, uds_path: str) -> None:
        self._target = self._to_uds_target(uds_path)
        self._channel: grpc.aio.Channel | None = None
        self._service: engine_service_rpc_pb2_grpc.KvEventControlServiceStub | None = (
            None
        )

    @staticmethod
    def _to_uds_target(path: str) -> str:
        if path.startswith("unix:"):
            return path
        if not path.startswith("/"):
            raise ValueError("DashLLM KV event control UDS path must be absolute")
        return f"unix://{path}"

    def _get_service(
        self,
    ) -> engine_service_rpc_pb2_grpc.KvEventControlServiceStub:
        if self._service is None:
            self._channel = grpc.aio.insecure_channel(
                self._target, options=CHANNEL_OPTIONS
            )
            stub_factory = cast(
                Callable[
                    [grpc.aio.Channel],
                    engine_service_rpc_pb2_grpc.KvEventControlServiceStub,
                ],
                engine_service_rpc_pb2_grpc.KvEventControlServiceStub,
            )
            self._service = stub_factory(self._channel)
        return self._service

    async def get_kv_event_bootstrap_info(
        self, timeout_s: float
    ) -> engine_service_rpc_pb2.KvEventBootstrapInfoPB:
        """Fetch engine-owned event transport and cache schema."""

        response = await self._get_service().GetKvEventBootstrapInfo(
            engine_service_rpc_pb2.KvEventBootstrapInfoRequestPB(),
            timeout=timeout_s,
        )
        return cast(engine_service_rpc_pb2.KvEventBootstrapInfoPB, response)

    async def get_all_kv_cache_blocks(self, timeout_s: float) -> object:
        """Fetch one versioned full KV cache snapshot."""

        response = await self._get_service().GetAllKvCacheBlocks(
            engine_service_rpc_pb2.KvCacheBlocksRequestPB(), timeout=timeout_s
        )
        return cast(object, response)

    async def close(self) -> None:
        """Release the local control channel."""

        channel = self._channel
        self._channel = None
        self._service = None
        if channel is not None:
            await channel.close()
