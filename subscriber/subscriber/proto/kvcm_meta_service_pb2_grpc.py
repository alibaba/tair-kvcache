"""Hand-maintained gRPC stubs for KVCM MetaService."""

from __future__ import annotations

import warnings

import grpc

from subscriber.proto import kvcm_meta_service_pb2 as _pb2

GRPC_GENERATED_VERSION = "1.68.1"
GRPC_VERSION = grpc.__version__
_version_not_supported = False

try:
    from grpc._utilities import first_version_is_lower

    _version_not_supported = first_version_is_lower(
        GRPC_VERSION, GRPC_GENERATED_VERSION
    )
except ImportError:
    _version_not_supported = True

if _version_not_supported:
    raise RuntimeError(
        f"The grpc package installed is at version {GRPC_VERSION},"
        + " but subscriber/proto/kvcm_meta_service_pb2_grpc.py depends on"
        + f" grpcio>={GRPC_GENERATED_VERSION}."
    )


class MetaServiceStub:
    """Client stub for kv_cache_manager.proto.meta.MetaService."""

    def __init__(self, channel: grpc.Channel) -> None:
        self.RegisterInstance = channel.unary_unary(
            "/kv_cache_manager.proto.meta.MetaService/RegisterInstance",
            request_serializer=_pb2.RegisterInstanceRequest.SerializeToString,
            response_deserializer=_pb2.RegisterInstanceResponse.FromString,
            _registered_method=True,
        )
        self.GetClusterInfo = channel.unary_unary(
            "/kv_cache_manager.proto.meta.MetaService/GetClusterInfo",
            request_serializer=_pb2.GetClusterInfoRequest.SerializeToString,
            response_deserializer=_pb2.GetClusterInfoResponse.FromString,
            _registered_method=True,
        )
        self.ReportEvent = channel.unary_unary(
            "/kv_cache_manager.proto.meta.MetaService/ReportEvent",
            request_serializer=_pb2.ReportEventRequest.SerializeToString,
            response_deserializer=_pb2.ReportEventResponse.FromString,
            _registered_method=True,
        )


class MetaServiceServicer:
    """Server base class for tests."""

    def RegisterInstance(self, request, context):
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details("Method not implemented!")
        raise NotImplementedError("Method not implemented!")

    def GetClusterInfo(self, request, context):
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details("Method not implemented!")
        raise NotImplementedError("Method not implemented!")

    def ReportEvent(self, request, context):
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details("Method not implemented!")
        raise NotImplementedError("Method not implemented!")


def add_MetaServiceServicer_to_server(
    servicer: MetaServiceServicer,
    server: grpc.Server,
) -> None:
    rpc_method_handlers = {
        "RegisterInstance": grpc.unary_unary_rpc_method_handler(
            servicer.RegisterInstance,
            request_deserializer=_pb2.RegisterInstanceRequest.FromString,
            response_serializer=_pb2.RegisterInstanceResponse.SerializeToString,
        ),
        "GetClusterInfo": grpc.unary_unary_rpc_method_handler(
            servicer.GetClusterInfo,
            request_deserializer=_pb2.GetClusterInfoRequest.FromString,
            response_serializer=_pb2.GetClusterInfoResponse.SerializeToString,
        ),
        "ReportEvent": grpc.unary_unary_rpc_method_handler(
            servicer.ReportEvent,
            request_deserializer=_pb2.ReportEventRequest.FromString,
            response_serializer=_pb2.ReportEventResponse.SerializeToString,
        ),
    }
    generic_handler = grpc.method_handlers_generic_handler(
        "kv_cache_manager.proto.meta.MetaService",
        rpc_method_handlers,
    )
    server.add_generic_rpc_handlers((generic_handler,))
    server.add_registered_method_handlers(
        "kv_cache_manager.proto.meta.MetaService",
        rpc_method_handlers,
    )


class MetaService:
    """Experimental static call helpers matching grpc_tools output."""

    @staticmethod
    def RegisterInstance(
        request,
        target,
        options=(),
        channel_credentials=None,
        call_credentials=None,
        insecure=False,
        compression=None,
        wait_for_ready=None,
        timeout=None,
        metadata=None,
    ):
        warnings.warn(
            "MetaService static methods are experimental",
            FutureWarning,
            stacklevel=2,
        )
        return grpc.experimental.unary_unary(
            request,
            target,
            "/kv_cache_manager.proto.meta.MetaService/RegisterInstance",
            _pb2.RegisterInstanceRequest.SerializeToString,
            _pb2.RegisterInstanceResponse.FromString,
            options,
            channel_credentials,
            insecure,
            call_credentials,
            compression,
            wait_for_ready,
            timeout,
            metadata,
            _registered_method=True,
        )

    @staticmethod
    def GetClusterInfo(
        request,
        target,
        options=(),
        channel_credentials=None,
        call_credentials=None,
        insecure=False,
        compression=None,
        wait_for_ready=None,
        timeout=None,
        metadata=None,
    ):
        warnings.warn(
            "MetaService static methods are experimental",
            FutureWarning,
            stacklevel=2,
        )
        return grpc.experimental.unary_unary(
            request,
            target,
            "/kv_cache_manager.proto.meta.MetaService/GetClusterInfo",
            _pb2.GetClusterInfoRequest.SerializeToString,
            _pb2.GetClusterInfoResponse.FromString,
            options,
            channel_credentials,
            insecure,
            call_credentials,
            compression,
            wait_for_ready,
            timeout,
            metadata,
            _registered_method=True,
        )

    @staticmethod
    def ReportEvent(
        request,
        target,
        options=(),
        channel_credentials=None,
        call_credentials=None,
        insecure=False,
        compression=None,
        wait_for_ready=None,
        timeout=None,
        metadata=None,
    ):
        warnings.warn(
            "MetaService static methods are experimental",
            FutureWarning,
            stacklevel=2,
        )
        return grpc.experimental.unary_unary(
            request,
            target,
            "/kv_cache_manager.proto.meta.MetaService/ReportEvent",
            _pb2.ReportEventRequest.SerializeToString,
            _pb2.ReportEventResponse.FromString,
            options,
            channel_credentials,
            insecure,
            call_credentials,
            compression,
            wait_for_ready,
            timeout,
            metadata,
            _registered_method=True,
        )
