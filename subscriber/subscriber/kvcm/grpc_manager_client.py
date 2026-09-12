"""KVCM manager gRPC client with service discovery and leader discovery."""

from __future__ import annotations

import asyncio
import random
import time
from dataclasses import dataclass
from typing import Any, cast
from urllib.parse import urlsplit

import grpc

from subscriber import logger
from subscriber.kvcm.base import AbstractKvCacheManagerClient
from subscriber.kvcm.errors import (
    KvcmResponseRejectedError,
    KvcmUnavailableError,
    ReportEventTransportDiagnostics,
    attach_report_event_transport_diagnostics,
    report_event_transport_diagnostics,
)
from subscriber.kvcm.service_discovery import (
    ServiceDiscovery,
    create_service_discovery,
)
from subscriber.proto import kvcm_meta_service_pb2, kvcm_meta_service_pb2_grpc

_DISCOVERY_STEP = "kvcm_discovery"
_REQUEST_STEP = "kvcm_request"
_CHANNEL_OPTIONS: list[tuple[str, int]] = [
    ("grpc.max_receive_message_length", 8 * 1024 * 1024),
    ("grpc.keepalive_time_ms", 2_000),
    ("grpc.keepalive_timeout_ms", 10_000),
    ("grpc.keepalive_permit_without_calls", 1),
    ("grpc.http2.max_pings_without_data", 0),
    ("grpc.enable_retries", 0),
    ("grpc.initial_reconnect_backoff_ms", 100),
    ("grpc.max_reconnect_backoff_ms", 1_000),
    ("grpc.tcp_receive_buffer_size", 512 * 1024),
]


@dataclass(frozen=True)
class _RequestResult:
    """One logical manager request and its retry and RPC-attempt diagnostics."""

    response: Any
    retry_count: int
    grpc_call_ms: float


@dataclass(frozen=True)
class _CallResult:
    """One completed RPC attempt and the time spent awaiting that RPC."""

    response: Any
    grpc_call_ms: float


class GrpcKvCacheManagerClient(AbstractKvCacheManagerClient):
    """gRPC manager client with optional service-discovery and leader refresh."""

    def __init__(
        self,
        base_url: str,
        *,
        instance_id: str = "",
        auto_discover_leader: bool = True,
        leader_retry_count: int = 1,
        leader_retry_base_interval_seconds: float = 0.005,
        discovery_refresh_interval_seconds: int = 30,
        min_discover_interval_seconds: float = 1.0,
        request_timeout_seconds: float = 5.0,
    ) -> None:
        super().__init__()
        self._configured_base_url = base_url
        self._request_timeout_seconds = float(request_timeout_seconds)
        if self._request_timeout_seconds <= 0:
            raise ValueError("request_timeout_seconds must be positive")

        self._instance_id = instance_id
        self._auto_discover_leader = auto_discover_leader
        self._leader_retry_count = leader_retry_count
        self._leader_retry_base_interval = leader_retry_base_interval_seconds
        self._discovery_refresh_interval = discovery_refresh_interval_seconds
        self._min_discover_interval = min_discover_interval_seconds

        self._service_discovery: ServiceDiscovery | None = None
        self._target = _base_url_to_target(base_url)
        self._discovery_target = self._target
        self._channel: grpc.aio.Channel | None = None
        self._stub: kvcm_meta_service_pb2_grpc.MetaServiceStub | None = None
        self._inflight_calls: dict[
            grpc.aio.Channel, set[grpc.aio.UnaryUnaryCall[Any, Any]]
        ] = {}
        self._raw_report_event: Any | None = None
        self._leader_lock = asyncio.Lock()
        self._refresh_event = asyncio.Event()
        self._closed = False
        self._last_discover_time: float = 0.0
        self._refresh_task: asyncio.Task[None] | None = None
        self._started = False

    async def start(self) -> None:
        """Start service discovery, leader refresh, and the initial channel."""
        if self._started:
            return
        if self._target is None:
            self._service_discovery = create_service_discovery(
                self._configured_base_url
            )
            if self._service_discovery is None:
                raise ValueError(
                    f"Invalid service discovery address: {self._configured_base_url}"
                )
            await self._service_discovery.start()
            ep = self._service_discovery.get_one_endpoint()
            if ep is not None:
                self._set_target(ep.host)
                self._discovery_target = ep.host
                logger.info(
                    "service discovery resolved manager endpoint",
                    step=_DISCOVERY_STEP,
                    tags={
                        "discovery_type": self._service_discovery.get_type(),
                        "target": self._target,
                    },
                )
            else:
                logger.warning(
                    "service discovery returned no endpoints; "
                    "waiting for background refresh",
                    step=_DISCOVERY_STEP,
                    tags={"discovery_url": self._configured_base_url},
                )

        if self._target is not None:
            self._get_stub()
        if self._auto_discover_leader:
            try:
                if await self.is_ready():
                    await self._discover_leader()
            except Exception as exc:
                logger.warning(
                    "initial leader discovery failed; keeping base target",
                    step=_DISCOVERY_STEP,
                    tags={
                        "target": self._target or "",
                        "error": exc.__class__.__name__,
                        "message": str(exc),
                    },
                )
            self._refresh_task = asyncio.create_task(
                self._leader_refresh_loop(),
                name="kvcm-grpc-leader-refresh",
            )
        self._started = True

    async def is_ready(self) -> bool:
        """Return whether requests have a usable gRPC target."""

        if self._target:
            return True
        if self._service_discovery is None:
            return False
        endpoint = self._service_discovery.get_one_endpoint()
        if endpoint is None:
            return False
        async with self._leader_lock:
            if not self._target:
                self._set_target(endpoint.host)
                logger.info(
                    "service discovery recovered manager endpoint",
                    step=_DISCOVERY_STEP,
                    tags={
                        "discovery_type": self._service_discovery.get_type(),
                        "target": self._target,
                    },
                )
        return True

    async def register_instance(
        self, data: dict[str, Any], check_response: bool = True
    ) -> dict[str, Any]:
        """Register an instance through the protobuf MetaService RPC."""
        request = _register_instance_request_from_dict(data)
        result = await self._request("RegisterInstance", request, check_response)
        return _register_instance_response_to_dict(
            cast(kvcm_meta_service_pb2.RegisterInstanceResponse, result.response)
        )

    async def report_event(
        self, data: dict[str, Any], check_response: bool = True
    ) -> dict[str, Any]:
        """Report lifecycle or KV events and attach local gRPC diagnostics.

        The returned dict and classified KVCM errors carry final wire bytes,
        raw encoder duration, and cumulative RPC-attempt duration for the
        domain client's asynchronous telemetry. A raw ``AioRpcError`` remains
        raw for direct transport callers; its diagnostics are available through
        :func:`subscriber.kvcm.errors.report_event_transport_diagnostics`.
        """
        wire_encode_started_at = time.monotonic()
        request = _report_event_request_to_wire_bytes(data)
        wire_encode_ms = (time.monotonic() - wire_encode_started_at) * 1000
        try:
            result = await self._request("ReportEvent", request, check_response)
        except KvcmResponseRejectedError as exc:
            exc.request_bytes = len(request)
            exc.wire_encode_ms = wire_encode_ms
            raise
        except KvcmUnavailableError as exc:
            exc.request_bytes = len(request)
            exc.wire_encode_ms = wire_encode_ms
            if exc.grpc_call_ms is None:
                exc.grpc_call_ms = 0.0
            raise
        except grpc.aio.AioRpcError as exc:
            diagnostics = report_event_transport_diagnostics(exc)
            attach_report_event_transport_diagnostics(
                exc,
                ReportEventTransportDiagnostics(
                    request_bytes=len(request),
                    wire_encode_ms=wire_encode_ms,
                    grpc_call_ms=diagnostics.grpc_call_ms,
                ),
            )
            raise
        response = _report_event_response_to_dict(
            cast(kvcm_meta_service_pb2.ReportEventResponse, result.response)
        )
        response["_subscriber_retry_count"] = result.retry_count
        response["_subscriber_request_bytes"] = len(request)
        response["_subscriber_wire_encode_ms"] = wire_encode_ms
        response["_subscriber_grpc_call_ms"] = result.grpc_call_ms
        return response

    async def close(self) -> None:
        """Stop refresh work and immediately release transport resources."""
        self._closed = True
        self._refresh_event.set()
        if self._refresh_task is not None:
            self._refresh_task.cancel()
            try:
                await self._refresh_task
            except asyncio.CancelledError:
                pass
            self._refresh_task = None
        await self._close_channel()
        if self._service_discovery is not None:
            await self._service_discovery.close()

    def _set_target(self, target: str | None) -> None:
        if target == self._target:
            return
        self._target = target
        self._stub = None
        self._raw_report_event = None

    async def _switch_target(self, target: str) -> None:
        if target == self._target:
            return
        previous = self._target
        previous_channel = self._detach_channel()
        self._set_target(target)
        self._get_stub()
        if previous_channel is not None:
            await self._wait_for_channel_calls(previous_channel)
            await previous_channel.close()
        logger.info(
            "leader discovery switched manager endpoint",
            step=_DISCOVERY_STEP,
            tags={"previous_target": previous or "", "target": target},
        )

    async def _close_channel(self) -> None:
        channel = self._detach_channel()
        if channel is not None:
            await channel.close()

    def _detach_channel(self) -> grpc.aio.Channel | None:
        """Remove the active channel so new RPCs use the current target."""
        channel = self._channel
        self._channel = None
        self._stub = None
        self._raw_report_event = None
        return channel

    async def _wait_for_channel_calls(self, channel: grpc.aio.Channel) -> None:
        """Let deadline-bounded RPCs finish before retiring their channel."""
        calls = tuple(self._inflight_calls.get(channel, ()))
        if calls:
            await asyncio.gather(*calls, return_exceptions=True)

    def _get_stub(self) -> kvcm_meta_service_pb2_grpc.MetaServiceStub:
        if not self._target:
            raise KvcmUnavailableError("KVCM gRPC target is unavailable")
        if self._stub is None:
            self._channel = grpc.aio.insecure_channel(
                self._target,
                options=_CHANNEL_OPTIONS,
            )
            self._stub = kvcm_meta_service_pb2_grpc.MetaServiceStub(
                cast(grpc.Channel, self._channel)
            )
            self._raw_report_event = self._channel.unary_unary(
                "/kv_cache_manager.proto.meta.MetaService/ReportEvent",
                request_serializer=lambda request: request,
                response_deserializer=(
                    kvcm_meta_service_pb2.ReportEventResponse.FromString
                ),
            )
        return self._stub

    def _resolve_discovery_target(self) -> str | None:
        if self._service_discovery is not None:
            ep = self._service_discovery.get_one_endpoint()
            if ep is not None:
                return ep.host
        return self._discovery_target or self._target

    async def _discover_leader(self) -> bool:
        snapshot = self._target
        async with self._leader_lock:
            if self._target != snapshot:
                return True
            return await self._do_discover_leader()

    async def _do_discover_leader(self) -> bool:
        target = self._resolve_discovery_target()
        if not target:
            return False
        original_target = self._target
        leader_discovered = False
        if target != self._target:
            await self._switch_target(target)
        try:
            request = kvcm_meta_service_pb2.GetClusterInfoRequest(
                trace_id=f"leader_discovery_{time.monotonic()}",
                instance_id=self._instance_id,
            )
            call_result = await self._call(
                "GetClusterInfo", request, notify_refresh=False
            )
            response = cast(
                kvcm_meta_service_pb2.GetClusterInfoResponse,
                call_result.response,
            )
            data = _get_cluster_info_response_to_dict(response)
            if _get_status_code(data) != "OK":
                status = data.get("header", {}).get("status", {})
                logger.warning(
                    "leader discovery returned error",
                    step=_DISCOVERY_STEP,
                    tags={
                        "target": target,
                        "kvcm_status_code": status.get("code", "Unknown"),
                        "message": status.get("message", ""),
                    },
                )
                return False
            leader_ep = data.get("leader_endpoint")
            if (
                not isinstance(leader_ep, dict)
                or not leader_ep.get("host")
                or not leader_ep.get("meta_rpc_port")
            ):
                logger.warning(
                    "leader discovery response missing leader endpoint",
                    step=_DISCOVERY_STEP,
                    tags={"target": target},
                )
                return False
            leader_discovered = True
            await self._switch_target(
                f"{leader_ep['host']}:{leader_ep['meta_rpc_port']}"
            )
            return True
        except Exception as exc:
            logger.warning(
                "leader discovery request failed",
                step=_DISCOVERY_STEP,
                tags={
                    "target": target,
                    "error": exc.__class__.__name__,
                    "message": str(exc),
                },
            )
            return False
        finally:
            if (
                not leader_discovered
                and original_target
                and self._target == target
                and target != original_target
            ):
                await self._switch_target(original_target)
            self._last_discover_time = time.monotonic()

    async def _leader_refresh_loop(self) -> None:
        while not self._closed:
            try:
                await asyncio.wait_for(
                    self._refresh_event.wait(),
                    timeout=self._discovery_refresh_interval,
                )
            except TimeoutError:
                pass
            self._refresh_event.clear()
            if self._closed:
                break
            remaining = self._min_discover_interval - (
                time.monotonic() - self._last_discover_time
            )
            if remaining > 0:
                await asyncio.sleep(remaining)
                if self._closed:
                    break
            try:
                await self._discover_leader()
            except Exception as exc:
                logger.warning(
                    "background leader refresh failed",
                    step=_DISCOVERY_STEP,
                    tags={"error": exc.__class__.__name__, "message": str(exc)},
                )

    def _notify_leader_refresh(self) -> None:
        if self._auto_discover_leader:
            self._refresh_event.set()

    async def _request(
        self,
        method_name: str,
        request: Any,
        check_response: bool,
    ) -> _RequestResult:
        retries_left = self._leader_retry_count if self._auto_discover_leader else 0
        retry_count = 0
        grpc_call_ms = 0.0
        while True:
            try:
                call_result = await self._call(
                    method_name,
                    request,
                    attempt=retry_count + 1,
                )
            except grpc.aio.AioRpcError as exc:
                diagnostics = report_event_transport_diagnostics(exc)
                if diagnostics.grpc_call_ms is not None:
                    grpc_call_ms += diagnostics.grpc_call_ms
                if not retry_count:
                    attach_report_event_transport_diagnostics(
                        exc,
                        ReportEventTransportDiagnostics(grpc_call_ms=grpc_call_ms),
                    )
                    raise
                raise KvcmUnavailableError(
                    str(exc),
                    status_code=f"GRPC_{exc.code().name}",
                    reason="transport",
                    retry_count=retry_count,
                    grpc_call_ms=grpc_call_ms,
                ) from exc
            else:
                response = call_result.response
                grpc_call_ms += call_result.grpc_call_ms
            payload = _response_to_dict(response)
            if (
                self._auto_discover_leader
                and _get_status_code(payload) == "SERVER_NOT_LEADER"
            ):
                if retries_left > 0:
                    retries_left -= 1
                    retry_count += 1
                    sleep_time = (
                        self._leader_retry_base_interval * retry_count
                        + random.uniform(0, self._leader_retry_base_interval)
                    )
                    logger.warning(
                        "kvcm request returned SERVER_NOT_LEADER; retrying",
                        step=_REQUEST_STEP,
                        tags={
                            "method": method_name,
                            "attempt": retry_count,
                            "retry_count": retry_count,
                            "kvcm_status_code": "SERVER_NOT_LEADER",
                            "retry_delay_seconds": round(sleep_time, 3),
                            "retries_left": retries_left,
                        },
                    )
                    await asyncio.sleep(sleep_time)
                    await self._discover_leader()
                    continue
                if retries_left <= 0:
                    logger.error(
                        "kvcm leader discovery retries exhausted",
                        step=_REQUEST_STEP,
                        tags={"method": method_name},
                    )
                    raise KvcmUnavailableError(
                        f"KVCM leader unavailable for {method_name}: "
                        "SERVER_NOT_LEADER and leader discovery exhausted",
                        status_code="SERVER_NOT_LEADER",
                        reason="leader_retry_exhausted",
                        retry_count=retry_count,
                        grpc_call_ms=grpc_call_ms,
                    )
            if check_response:
                status = payload.get("header", {}).get("status", {})
                if status.get("code") != "OK":
                    item_results = payload.get("item_results")
                    item_results_detail = (
                        f"; item_results={item_results!r}" if item_results else ""
                    )
                    raise KvcmResponseRejectedError(
                        f"KVCM /grpc/{method_name} failed: "
                        f"{status.get('code')} {status.get('message')}"
                        f"{item_results_detail}",
                        status_code=_get_status_code(payload) or "UNKNOWN",
                        retry_count=retry_count,
                        grpc_call_ms=grpc_call_ms,
                    )
            return _RequestResult(
                response=response,
                retry_count=retry_count,
                grpc_call_ms=grpc_call_ms,
            )

    async def _call(
        self,
        method_name: str,
        request: Any,
        *,
        notify_refresh: bool = True,
        attempt: int = 1,
    ) -> _CallResult:
        """Execute one RPC attempt with channel bookkeeping and transport logs.

        The returned duration covers only ``await call``. Stub/channel setup,
        inflight-set maintenance, and logging stay outside that measurement so
        ``grpc_call_ms`` remains comparable with downstream RPC latency.
        """

        target = self._target or ""
        request_bytes = len(request) if isinstance(request, bytes) else None
        try:
            if method_name == "ReportEvent" and isinstance(request, bytes):
                self._get_stub()
                method = self._raw_report_event
                if method is None:
                    raise KvcmUnavailableError(
                        "KVCM gRPC ReportEvent method is unavailable"
                    )
            else:
                method = getattr(self._get_stub(), method_name)
            channel = self._channel
            if channel is None:
                raise RuntimeError("KVCM gRPC channel was not initialized")
            call = method(
                request,
                timeout=self._request_timeout_seconds,
                wait_for_ready=False,
            )
            active_calls = self._inflight_calls.setdefault(channel, set())
            active_calls.add(call)
            try:
                rpc_started_at = time.monotonic()
                response = await call
            except grpc.aio.AioRpcError as exc:
                attach_report_event_transport_diagnostics(
                    exc,
                    ReportEventTransportDiagnostics(
                        grpc_call_ms=(time.monotonic() - rpc_started_at) * 1000
                    ),
                )
                raise
            finally:
                active_calls.discard(call)
                if not active_calls:
                    self._inflight_calls.pop(channel, None)
            grpc_call_ms = (time.monotonic() - rpc_started_at) * 1000
            if logger.is_debug_enabled():
                completed_tags: dict[str, object] = {
                    "target": target,
                    "method": method_name,
                    "attempt": attempt,
                    "grpc_status": "OK",
                    "grpc_call_ms": round(grpc_call_ms, 3),
                }
                if request_bytes is not None:
                    completed_tags["request_bytes"] = request_bytes
                logger.debug(
                    "kvcm grpc request completed",
                    step=_REQUEST_STEP,
                    tags=completed_tags,
                )
            return _CallResult(response=response, grpc_call_ms=grpc_call_ms)
        except asyncio.CancelledError:
            raise
        except grpc.aio.AioRpcError as exc:
            if notify_refresh:
                self._notify_leader_refresh()
            code = exc.code()
            message = exc.details() or str(exc)
            error_tags: dict[str, object] = {
                "target": target,
                "method": method_name,
                "attempt": attempt,
                "grpc_status": code.name,
                "grpc_call_ms": round(
                    report_event_transport_diagnostics(exc).grpc_call_ms or 0.0,
                    3,
                ),
            }
            if request_bytes is not None:
                error_tags["request_bytes"] = request_bytes
            if code == grpc.StatusCode.DEADLINE_EXCEEDED:
                error_tags["timeout_seconds"] = self._request_timeout_seconds
                logger.warning(
                    "kvcm grpc request timed out",
                    step=_REQUEST_STEP,
                    tags=error_tags,
                )
            elif code == grpc.StatusCode.UNAVAILABLE:
                error_tags["message"] = message
                logger.warning(
                    "kvcm grpc request connection failed; notifying leader refresh",
                    step=_REQUEST_STEP,
                    tags=error_tags,
                )
            else:
                error_tags["message"] = message
                logger.warning(
                    "kvcm grpc request failed",
                    step=_REQUEST_STEP,
                    tags=error_tags,
                )
            raise


def _base_url_to_target(base_url: str) -> str | None:
    """Return a direct gRPC target, or ``None`` to request service discovery.

    ``start()`` treats ``None`` as an internal sentinel: it passes the configured
    address to ``create_service_discovery()``. Bare targets and ``http(s)`` or
    ``grpc`` URLs are direct targets; ``static`` and ``spectrum`` URLs therefore
    return ``None``. An empty or unsupported discovery address is rejected by
    ``start()`` rather than being used as a gRPC target.
    """
    if not base_url:
        return None
    if "://" not in base_url:
        return base_url.rstrip("/")
    parsed = urlsplit(base_url)
    if parsed.scheme in {"http", "https", "grpc"} and parsed.netloc:
        return parsed.netloc.rstrip("/")
    if parsed.scheme:
        return None
    return base_url.rstrip("/")


def _enum_value(enum_wrapper: Any, value: object, *, default_name: str | None) -> int:
    if isinstance(value, int) and not isinstance(value, bool):
        if value in enum_wrapper.DESCRIPTOR.values_by_number:
            return value
    if isinstance(value, str) and value:
        values = enum_wrapper.DESCRIPTOR.values_by_name
        if value in values:
            return cast(int, values[value].number)
    if default_name is not None:
        return cast(int, enum_wrapper.Value(default_name))
    raise ValueError(f"invalid {enum_wrapper.DESCRIPTOR.full_name} value: {value!r}")


def _event_type_value(value: object) -> int:
    if (
        isinstance(value, int)
        and not isinstance(value, bool)
        and value
        not in kvcm_meta_service_pb2.ReportEventType.DESCRIPTOR.values_by_number
    ):
        raise ValueError(f"invalid ReportEventType value: {value!r}")
    return _enum_value(
        kvcm_meta_service_pb2.ReportEventType,
        value,
        default_name="EVENT_UNSPECIFIED",
    )


def _enum_name(enum_wrapper: Any, value: int) -> str:
    try:
        return cast(str, enum_wrapper.Name(value))
    except ValueError:
        return str(value)


def _dict_list(value: object) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, dict)]


def _register_instance_request_from_dict(
    data: dict[str, Any],
) -> kvcm_meta_service_pb2.RegisterInstanceRequest:
    model = data.get("model_deployment")
    model_data = model if isinstance(model, dict) else {}
    return kvcm_meta_service_pb2.RegisterInstanceRequest(
        trace_id=str(data.get("trace_id", "")),
        instance_group=str(data.get("instance_group", "")),
        instance_id=str(data.get("instance_id", "")),
        block_size=int(data.get("block_size", 0) or 0),
        location_spec_infos=[
            kvcm_meta_service_pb2.LocationSpecInfo(
                name=str(item.get("name", "")),
                size=int(item.get("size", 0) or 0),
            )
            for item in _dict_list(data.get("location_spec_infos"))
        ],
        model_deployment=kvcm_meta_service_pb2.ModelDeployment(
            model_name=str(model_data.get("model_name", "")),
            dtype=str(model_data.get("dtype", "")),
            use_mla=bool(model_data.get("use_mla", False)),
            tp_size=int(model_data.get("tp_size", 0) or 0),
            dp_size=int(model_data.get("dp_size", 0) or 0),
            lora_name=str(model_data.get("lora_name", "")),
            pp_size=int(model_data.get("pp_size", 0) or 0),
            extra=str(model_data.get("extra", "")),
            user_data=str(model_data.get("user_data", "")),
            use_eagle_pop=bool(model_data.get("use_eagle_pop", False)),
        ),
        location_spec_groups=[
            kvcm_meta_service_pb2.LocationSpecGroup(
                name=str(item.get("name", "")),
                spec_names=[
                    str(spec_name)
                    for spec_name in item.get("spec_names", [])
                    if isinstance(spec_name, str)
                ],
            )
            for item in _dict_list(data.get("location_spec_groups"))
        ],
        default_query_type=cast(
            kvcm_meta_service_pb2.QueryType,
            _enum_value(
                kvcm_meta_service_pb2.QueryType,
                data.get("default_query_type"),
                default_name="QT_UNSPECIFIED",
            ),
        ),
    )


def _location_specs_from_list(specs: object) -> list[Any]:
    return [
        kvcm_meta_service_pb2.LocationSpec(
            name=str(item.get("name", "")),
            uri=str(item.get("uri", "")),
        )
        for item in _dict_list(specs)
    ]


_WIRE_VARINT = 0
_WIRE_LEN = 2


def _write_varint(out: bytearray, value: int) -> None:
    value = int(value)
    while value > 0x7F:
        out.append((value & 0x7F) | 0x80)
        value >>= 7
    out.append(value)


def _varint_size(value: int) -> int:
    value = int(value)
    size = 1
    while value > 0x7F:
        size += 1
        value >>= 7
    return size


def _write_key(out: bytearray, field_number: int, wire_type: int) -> None:
    _write_varint(out, (field_number << 3) | wire_type)


def _key_size(field_number: int, wire_type: int) -> int:
    return _varint_size((field_number << 3) | wire_type)


def _write_string(out: bytearray, field_number: int, value: object) -> None:
    if value is None:
        return
    encoded = str(value).encode()
    if not encoded:
        return
    _write_encoded_string(out, field_number, encoded)


def _write_encoded_string(out: bytearray, field_number: int, encoded: bytes) -> None:
    _write_key(out, field_number, _WIRE_LEN)
    _write_varint(out, len(encoded))
    out.extend(encoded)


def _encoded_string_size(field_number: int, encoded: bytes) -> int:
    if not encoded:
        return 0
    return (
        _key_size(field_number, _WIRE_LEN) + _varint_size(len(encoded)) + len(encoded)
    )


def _repeated_encoded_string_size(field_number: int, encoded: bytes) -> int:
    return (
        _key_size(field_number, _WIRE_LEN) + _varint_size(len(encoded)) + len(encoded)
    )


def _write_enum(out: bytearray, field_number: int, value: int) -> None:
    if value == 0:
        return
    _write_key(out, field_number, _WIRE_VARINT)
    _write_varint(out, value)


def _write_message(out: bytearray, field_number: int, payload: bytes) -> None:
    _write_key(out, field_number, _WIRE_LEN)
    _write_varint(out, len(payload))
    out.extend(payload)


def _message_size(field_number: int, payload_len: int) -> int:
    return _key_size(field_number, _WIRE_LEN) + _varint_size(payload_len) + payload_len


def _location_spec_to_wire_bytes(data: dict[str, Any]) -> bytes:
    out = bytearray()
    _write_string(out, 1, data.get("name", ""))
    _write_string(out, 2, data.get("uri", ""))
    return bytes(out)


def _location_specs_to_wire_messages(
    out: bytearray, field_number: int, specs: object
) -> None:
    for item in _dict_list(specs):
        _write_message(out, field_number, _location_spec_to_wire_bytes(item))


def _node_register_to_wire_bytes(data: dict[str, Any]) -> bytes:
    out = bytearray()
    for medium in data.get("mediums", []):
        if isinstance(medium, str):
            _write_encoded_string(out, 1, medium.encode())
    return bytes(out)


def _block_add_to_wire_bytes(data: dict[str, Any]) -> bytes:
    out = bytearray()
    _write_string(out, 1, data.get("block_key", ""))
    _write_string(out, 2, data.get("uri", ""))
    _write_string(out, 3, data.get("medium", ""))
    _location_specs_to_wire_messages(out, 4, data.get("specs"))
    return bytes(out)


def _block_delete_to_wire_bytes(data: dict[str, Any]) -> bytes:
    out = bytearray()
    _write_string(out, 1, data.get("block_key", ""))
    _write_string(out, 2, data.get("medium", ""))
    for name in data.get("spec_names", []):
        if isinstance(name, str):
            _write_encoded_string(out, 3, name.encode())
    return bytes(out)


def _heartbeat_to_wire_bytes(data: dict[str, Any]) -> bytes:
    out = bytearray()
    system_status = data.get("system_status")
    status_map = system_status if isinstance(system_status, dict) else {}
    for key, value in status_map.items():
        entry = bytearray()
        _write_string(entry, 1, key)
        _write_string(entry, 2, value)
        _write_message(out, 1, bytes(entry))
    return bytes(out)


def _block_snapshot_to_wire_bytes(data: dict[str, Any]) -> bytes:
    out = bytearray()
    _write_string(out, 1, data.get("medium", ""))
    spec_cache: dict[tuple[str, str], bytes] = {}
    blocks = data.get("blocks")
    if not isinstance(blocks, list):
        return bytes(out)
    for item in blocks:
        if not isinstance(item, dict):
            continue
        block_key = str(item.get("block_key", "")).encode()
        medium = str(item.get("medium", "")).encode()
        specs_len = 0
        spec_payloads: list[bytes] = []
        specs = item.get("specs")
        if isinstance(specs, list):
            for spec in specs:
                if isinstance(spec, dict):
                    payload = _location_spec_to_cached_wire_bytes(spec, spec_cache)
                    spec_payloads.append(payload)
                    specs_len += _message_size(3, len(payload))
        payload_len = (
            _encoded_string_size(1, block_key)
            + _encoded_string_size(2, medium)
            + specs_len
        )
        out.append(0x12)
        _write_varint(out, payload_len)
        if block_key:
            out.append(0x0A)
            _write_varint(out, len(block_key))
            out.extend(block_key)
        if medium:
            out.append(0x12)
            _write_varint(out, len(medium))
            out.extend(medium)
        for spec_payload in spec_payloads:
            out.append(0x1A)
            _write_varint(out, len(spec_payload))
            out.extend(spec_payload)
    return bytes(out)


def _location_spec_to_cached_wire_bytes(
    data: dict[str, Any], cache: dict[tuple[str, str], bytes]
) -> bytes:
    name = str(data.get("name", ""))
    uri = str(data.get("uri", ""))
    key = (name, uri)
    cached = cache.get(key)
    if cached is not None:
        return cached
    out = bytearray()
    _write_string(out, 1, name)
    _write_string(out, 2, uri)
    payload = bytes(out)
    cache[key] = payload
    return payload


def _write_block_add_event_message(
    out: bytearray,
    field_number: int,
    data: dict[str, Any],
    spec_cache: dict[tuple[str, str], bytes],
) -> None:
    payload = data.get("block_add")
    block_add = payload if isinstance(payload, dict) else {}
    block_key = str(block_add.get("block_key", "")).encode()
    uri = str(block_add.get("uri", "")).encode()
    medium = str(block_add.get("medium", "")).encode()
    specs_len = 0
    spec_payloads: list[bytes] = []
    specs = block_add.get("specs")
    if isinstance(specs, list):
        for spec in specs:
            if isinstance(spec, dict):
                spec_payload = _location_spec_to_cached_wire_bytes(spec, spec_cache)
                spec_payloads.append(spec_payload)
                specs_len += _message_size(4, len(spec_payload))
    block_add_len = (
        _encoded_string_size(1, block_key)
        + _encoded_string_size(2, uri)
        + _encoded_string_size(3, medium)
        + specs_len
    )
    event_type_len = _key_size(1, _WIRE_VARINT) + _varint_size(
        kvcm_meta_service_pb2.EVENT_BLOCK_ADD
    )
    event_len = event_type_len + _message_size(3, block_add_len)
    _write_key(out, field_number, _WIRE_LEN)
    _write_varint(out, event_len)
    out.append(0x08)
    _write_varint(out, kvcm_meta_service_pb2.EVENT_BLOCK_ADD)
    out.append(0x1A)
    _write_varint(out, block_add_len)
    if block_key:
        out.append(0x0A)
        _write_varint(out, len(block_key))
        out.extend(block_key)
    if uri:
        out.append(0x12)
        _write_varint(out, len(uri))
        out.extend(uri)
    if medium:
        out.append(0x1A)
        _write_varint(out, len(medium))
        out.extend(medium)
    for spec_payload in spec_payloads:
        out.append(0x22)
        _write_varint(out, len(spec_payload))
        out.extend(spec_payload)


def _write_block_delete_event_message(
    out: bytearray, field_number: int, data: dict[str, Any]
) -> None:
    payload = data.get("block_delete")
    block_delete = payload if isinstance(payload, dict) else {}
    block_key = str(block_delete.get("block_key", "")).encode()
    medium = str(block_delete.get("medium", "")).encode()
    spec_names_len = 0
    encoded_spec_names: list[bytes] = []
    spec_names = block_delete.get("spec_names")
    if isinstance(spec_names, list):
        for name in spec_names:
            if isinstance(name, str):
                encoded = name.encode()
                encoded_spec_names.append(encoded)
                spec_names_len += _repeated_encoded_string_size(3, encoded)
    block_delete_len = (
        _encoded_string_size(1, block_key)
        + _encoded_string_size(2, medium)
        + spec_names_len
    )
    event_type_len = _key_size(1, _WIRE_VARINT) + _varint_size(
        kvcm_meta_service_pb2.EVENT_BLOCK_DELETE
    )
    event_len = event_type_len + _message_size(4, block_delete_len)
    _write_key(out, field_number, _WIRE_LEN)
    _write_varint(out, event_len)
    out.append(0x08)
    _write_varint(out, kvcm_meta_service_pb2.EVENT_BLOCK_DELETE)
    out.append(0x22)
    _write_varint(out, block_delete_len)
    if block_key:
        out.append(0x0A)
        _write_varint(out, len(block_key))
        out.extend(block_key)
    if medium:
        out.append(0x12)
        _write_varint(out, len(medium))
        out.extend(medium)
    for encoded in encoded_spec_names:
        out.append(0x1A)
        _write_varint(out, len(encoded))
        out.extend(encoded)


def _event_item_to_wire_bytes(data: dict[str, Any]) -> bytes:
    event_type = data.get("event_type")
    event_type_number = _event_type_value(event_type)
    out = bytearray()
    _write_enum(out, 1, event_type_number)
    if event_type == "EVENT_NODE_REGISTER":
        payload = data.get("node_register")
        node_register = payload if isinstance(payload, dict) else {}
        _write_message(out, 2, _node_register_to_wire_bytes(node_register))
    elif event_type == "EVENT_BLOCK_ADD":
        payload = data.get("block_add")
        block_add = payload if isinstance(payload, dict) else {}
        _write_message(out, 3, _block_add_to_wire_bytes(block_add))
    elif event_type == "EVENT_BLOCK_DELETE":
        payload = data.get("block_delete")
        block_delete = payload if isinstance(payload, dict) else {}
        _write_message(out, 4, _block_delete_to_wire_bytes(block_delete))
    elif event_type == "EVENT_HOST_DOWN":
        _write_message(out, 5, b"")
    elif event_type == "EVENT_HEARTBEAT":
        payload = data.get("heartbeat")
        heartbeat = payload if isinstance(payload, dict) else {}
        _write_message(out, 6, _heartbeat_to_wire_bytes(heartbeat))
    elif event_type == "EVENT_BLOCK_SNAPSHOT":
        payload = data.get("block_snapshot")
        block_snapshot = payload if isinstance(payload, dict) else {}
        _write_message(out, 7, _block_snapshot_to_wire_bytes(block_snapshot))
    return bytes(out)


def _report_event_request_to_wire_bytes(data: dict[str, Any]) -> bytes:
    """Encode a ReportEvent request without building protobuf message objects.

    The output must be wire-equivalent to ``_report_event_request_from_dict`` for
    accepted input. This path writes KVCM's fixed field layout directly so large
    snapshots avoid per-block Python protobuf allocations; update it together
    with the protobuf schema and its equivalence tests.
    """
    out = bytearray()
    _write_string(out, 1, data.get("trace_id", ""))
    _write_string(out, 2, data.get("instance_id", ""))
    _write_string(out, 3, data.get("host_ip_port", ""))
    spec_cache: dict[tuple[str, str], bytes] = {}
    for item in _dict_list(data.get("events")):
        event_type = item.get("event_type")
        if event_type == "EVENT_BLOCK_ADD":
            _write_block_add_event_message(out, 4, item, spec_cache)
        elif event_type == "EVENT_BLOCK_DELETE":
            _write_block_delete_event_message(out, 4, item)
        else:
            _write_message(out, 4, _event_item_to_wire_bytes(item))
    _write_enum(
        out,
        5,
        _enum_value(
            kvcm_meta_service_pb2.StorageType,
            data.get("storage_type"),
            default_name=None,
        ),
    )
    return bytes(out)


def _event_item_from_dict(data: dict[str, Any]) -> kvcm_meta_service_pb2.EventItem:
    event_type = data.get("event_type")
    event_type_number = _event_type_value(event_type)
    common: dict[str, Any] = {
        "event_type": cast(kvcm_meta_service_pb2.ReportEventType, event_type_number)
    }
    if event_type == "EVENT_NODE_REGISTER":
        payload = data.get("node_register")
        node_register = payload if isinstance(payload, dict) else {}
        return kvcm_meta_service_pb2.EventItem(
            **common,
            node_register=kvcm_meta_service_pb2.NodeRegisterEventParams(
                mediums=[
                    str(medium)
                    for medium in node_register.get("mediums", [])
                    if isinstance(medium, str)
                ]
            ),
        )
    if event_type == "EVENT_BLOCK_ADD":
        payload = data.get("block_add")
        block_add = payload if isinstance(payload, dict) else {}
        return kvcm_meta_service_pb2.EventItem(
            **common,
            block_add=kvcm_meta_service_pb2.BlockAddEventParams(
                block_key=str(block_add.get("block_key", "")),
                uri=str(block_add.get("uri", "")),
                medium=str(block_add.get("medium", "")),
                specs=_location_specs_from_list(block_add.get("specs")),
            ),
        )
    if event_type == "EVENT_BLOCK_DELETE":
        payload = data.get("block_delete")
        block_delete = payload if isinstance(payload, dict) else {}
        return kvcm_meta_service_pb2.EventItem(
            **common,
            block_delete=kvcm_meta_service_pb2.BlockDeleteEventParams(
                block_key=str(block_delete.get("block_key", "")),
                medium=str(block_delete.get("medium", "")),
                spec_names=[
                    str(name)
                    for name in block_delete.get("spec_names", [])
                    if isinstance(name, str)
                ],
            ),
        )
    if event_type == "EVENT_HOST_DOWN":
        return kvcm_meta_service_pb2.EventItem(
            **common,
            host_down=kvcm_meta_service_pb2.HostDownEventParams(),
        )
    if event_type == "EVENT_HEARTBEAT":
        payload = data.get("heartbeat")
        heartbeat = payload if isinstance(payload, dict) else {}
        system_status = heartbeat.get("system_status")
        status_map = system_status if isinstance(system_status, dict) else {}
        return kvcm_meta_service_pb2.EventItem(
            **common,
            heartbeat=kvcm_meta_service_pb2.HeartbeatEventParams(
                system_status={
                    str(key): str(value) for key, value in status_map.items()
                }
            ),
        )
    if event_type == "EVENT_BLOCK_SNAPSHOT":
        payload = data.get("block_snapshot")
        block_snapshot = payload if isinstance(payload, dict) else {}
        return kvcm_meta_service_pb2.EventItem(
            **common,
            block_snapshot=kvcm_meta_service_pb2.BlockSnapshotEventParams(
                medium=str(block_snapshot.get("medium", "")),
                blocks=[
                    kvcm_meta_service_pb2.BlockSnapshotItem(
                        block_key=str(item.get("block_key", "")),
                        medium=str(item.get("medium", "")),
                        specs=_location_specs_from_list(item.get("specs")),
                    )
                    for item in _dict_list(block_snapshot.get("blocks"))
                ],
            ),
        )
    return kvcm_meta_service_pb2.EventItem(**common)


def _report_event_request_from_dict(
    data: dict[str, Any],
) -> kvcm_meta_service_pb2.ReportEventRequest:
    return kvcm_meta_service_pb2.ReportEventRequest(
        trace_id=str(data.get("trace_id", "")),
        instance_id=str(data.get("instance_id", "")),
        host_ip_port=str(data.get("host_ip_port", "")),
        events=[_event_item_from_dict(item) for item in _dict_list(data.get("events"))],
        storage_type=cast(
            kvcm_meta_service_pb2.StorageType,
            _enum_value(
                kvcm_meta_service_pb2.StorageType,
                data.get("storage_type"),
                default_name=None,
            ),
        ),
    )


def _status_to_dict(header: Any) -> dict[str, Any]:
    return {
        "status": {
            "code": _enum_name(kvcm_meta_service_pb2.ErrorCode, header.status.code),
            "message": header.status.message,
        },
        "request_id": header.request_id,
        "tracer_result": header.tracer_result,
    }


def _register_instance_response_to_dict(
    response: kvcm_meta_service_pb2.RegisterInstanceResponse,
) -> dict[str, Any]:
    return {
        "header": _status_to_dict(response.header),
        "storage_configs": response.storage_configs,
        "extra_info": response.extra_info,
    }


def _report_event_response_to_dict(
    response: kvcm_meta_service_pb2.ReportEventResponse,
) -> dict[str, Any]:
    return {
        "header": _status_to_dict(response.header),
        "item_results": [
            _enum_name(kvcm_meta_service_pb2.ErrorCode, item)
            for item in response.item_results
        ],
        "committed_snapshot_version": response.committed_snapshot_version,
        "retry_after_ms": response.retry_after_ms,
        "snapshot_required": response.snapshot_required,
        "extra_info": response.extra_info,
    }


def _get_cluster_info_response_to_dict(
    response: kvcm_meta_service_pb2.GetClusterInfoResponse,
) -> dict[str, Any]:
    return {
        "header": _status_to_dict(response.header),
        "self_node_id": response.self_node_id,
        "leader_node_id": response.leader_node_id,
        "leader_endpoint": {
            "node_id": response.leader_endpoint.node_id,
            "host": response.leader_endpoint.host,
            "meta_rpc_port": response.leader_endpoint.meta_rpc_port,
            "meta_http_port": response.leader_endpoint.meta_http_port,
            "custom_info": response.leader_endpoint.custom_info,
        },
    }


def _response_to_dict(response: Any) -> dict[str, Any]:
    if isinstance(response, kvcm_meta_service_pb2.RegisterInstanceResponse):
        return _register_instance_response_to_dict(response)
    if isinstance(response, kvcm_meta_service_pb2.ReportEventResponse):
        return _report_event_response_to_dict(response)
    if isinstance(response, kvcm_meta_service_pb2.GetClusterInfoResponse):
        return _get_cluster_info_response_to_dict(response)
    return {}


def _get_status_code(response_data: dict[str, Any]) -> str | None:
    code = response_data.get("header", {}).get("status", {}).get("code")
    return code if isinstance(code, str) else None
