"""Remote TCP client for the FlexLB-compatible DashLLM worker-status RPC."""

from __future__ import annotations

from collections.abc import Callable
from typing import cast

import grpc

from subscriber.engine.grpc_options import CHANNEL_OPTIONS
from subscriber.proto import engine_service_rpc_pb2, engine_service_rpc_pb2_grpc


class DashllmWorkerStatusClient:
    """Own the remote status channel and its monotonic request cursors."""

    def __init__(self, target: str) -> None:
        self._target = target
        self._channel: grpc.aio.Channel | None = None
        self._service: engine_service_rpc_pb2_grpc.RpcServiceStub | None = None
        self._status_version = -1
        self._latest_cache_version = 0
        self._latest_finished_version = -1

    def _get_service(self) -> engine_service_rpc_pb2_grpc.RpcServiceStub:
        if self._service is None:
            self._channel = grpc.aio.insecure_channel(
                self._target, options=CHANNEL_OPTIONS
            )
            stub_factory = cast(
                Callable[
                    [grpc.aio.Channel],
                    engine_service_rpc_pb2_grpc.RpcServiceStub,
                ],
                engine_service_rpc_pb2_grpc.RpcServiceStub,
            )
            self._service = stub_factory(self._channel)
        return self._service

    async def get_worker_status(
        self, timeout_s: float
    ) -> engine_service_rpc_pb2.WorkerStatusPB:
        """Probe liveness and advance the accepted finished cursor."""

        request = engine_service_rpc_pb2.StatusVersionPB(
            latest_cache_version=self._latest_cache_version,
            latest_finished_version=self._latest_finished_version,
        )
        response = await self._get_service().GetWorkerStatus(request, timeout=timeout_s)
        typed_response = cast(engine_service_rpc_pb2.WorkerStatusPB, response)
        if (
            typed_response.status_version > 0
            and typed_response.status_version > self._status_version
        ):
            self._status_version = typed_response.status_version
            self._latest_finished_version = typed_response.latest_finished_version
        return typed_response

    async def close(self) -> None:
        """Release the remote status channel."""

        channel = self._channel
        self._channel = None
        self._service = None
        if channel is not None:
            await channel.close()
