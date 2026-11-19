from __future__ import annotations

import asyncio
import json
import os
import time
from collections.abc import Callable, Mapping
from typing import Any, cast

import grpc

from subscriber import logger
from subscriber.config import SubscriberConfig
from subscriber.engine.metadata import KvCacheDescriptor, MetadataProtocolError
from subscriber.kvcm.base import AbstractKvCacheManagerClient
from subscriber.kvcm.enum import KvcmReportEventType
from subscriber.kvcm.errors import (
    KvcmReportError,
    KvcmReportRejectedError,
    KvcmResponseRejectedError,
    KvcmUnavailableError,
    report_event_transport_diagnostics,
)
from subscriber.kvcm.event_payload import (
    build_merged_snapshot_blocks,
    expand_report_events,
    source_counts,
    split_report_event_requests,
)
from subscriber.kvcm.kinds import effective_attention_type_categories
from subscriber.kvcm.manager_client import HttpKvCacheManagerClient
from subscriber.metrics import (
    BatchTelemetry,
    report_heartbeat,
    report_registration_recovery,
    report_registration_transition,
)
from subscriber.types import BlockSnapshot, KvCacheGroupSpec, KVEventBatch
from subscriber.utils.network import resolve_host_ip_port

# Legacy compatibility for fake tests and older HTTP transports. New transports
# raise KvcmResponseRejectedError for rejected KVCM responses.
_REPORT_REJECTED_PREFIX = "KVCM /api/reportEvent failed:"

# While unregistered with no available KVCM endpoint, repeat a summary warning
# at most this often.
_UNREGISTERED_WARN_INTERVAL_S = 30.0
_SNAPSHOT_SIGNAL_WARN_INTERVAL_S = 30.0


def _report_status_code(response: dict[str, Any]) -> str:
    status = response.get("header", {}).get("status", {})
    code = status.get("code") if isinstance(status, dict) else None
    return code if isinstance(code, str) and code else "UNKNOWN"


def _response_retry_count(response: dict[str, Any]) -> int:
    retry_count = response.get("_subscriber_retry_count")
    if isinstance(retry_count, int) and not isinstance(retry_count, bool):
        return max(0, retry_count)
    return 0


def _transport_gauges(
    *,
    request_bytes: object,
    wire_encode_ms: object,
    grpc_call_ms: object,
) -> dict[str, float]:
    """Return valid gRPC ReportEvent diagnostics as metric observations."""

    gauges: dict[str, float] = {}
    for value, metric_name in (
        (request_bytes, "kvcm_report_event_request_bytes"),
        (wire_encode_ms, "kvcm_report_event_wire_encode_ms"),
        (grpc_call_ms, "kvcm_report_event_grpc_call_ms"),
    ):
        if (
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and value >= 0
        ):
            gauges[metric_name] = float(value)
    return gauges


def _response_transport_gauges(response: dict[str, Any]) -> dict[str, float]:
    """Return valid gRPC ReportEvent diagnostics from a transport response."""

    return _transport_gauges(
        request_bytes=response.get("_subscriber_request_bytes"),
        wire_encode_ms=response.get("_subscriber_wire_encode_ms"),
        grpc_call_ms=response.get("_subscriber_grpc_call_ms"),
    )


def _get_engine_config_from_env() -> dict[str, Any]:
    raw = os.environ.get("DS_LLM_ENGINE_CONFIG", "")
    if not raw:
        return {}
    try:
        parsed = json.loads(raw)
        if not isinstance(parsed, dict):
            logger.warning(
                "DS_LLM_ENGINE_CONFIG must decode to a JSON object",
                step="kvcm_client_init",
                tags={"parsed_type": type(parsed).__name__},
            )
            return {}
        return parsed
    except (json.JSONDecodeError, TypeError):
        logger.warning(
            "failed to parse DS_LLM_ENGINE_CONFIG",
            step="kvcm_client_init",
            exc_info=True,
        )
        return {}


class KvcmClient:
    """Async boundary for forwarding KV event batches to kvcm.

    ``on_snapshot_required`` is invoked from the heartbeat loop when a
    heartbeat response carries ``snapshot_required=True`` — kvcm signals this
    after a restart that lost its view of this instance's kvcache, so the
    engine adapter can trigger an immediate full snapshot. The callback must
    be synchronous and non-blocking (``Event.set()`` plus logging only); its
    exceptions are swallowed and the signal is retried on the next heartbeat.
    The incremental-send path handles ``snapshot_required`` separately via the
    ``report_kv_events`` return value (wired in ``forwarding.py``).

    TODO: Define restart and idempotent-close semantics if a future caller needs
    to reuse a client after ``close()``. The current lifecycle is one
    ``start()`` followed by one ``close()``.
    """

    def __init__(
        self,
        config: SubscriberConfig,
        *,
        medium_mapper: Callable[[str | None], str],
        storage_type: str,
        supported_mediums: list[str],
        descriptor: KvCacheDescriptor,
        manager_client: AbstractKvCacheManagerClient | None = None,
        on_snapshot_required: Callable[[], None] | None = None,
    ) -> None:
        self._config = config
        self._kind_categories = effective_attention_type_categories(
            config.extra_attention_types
        )
        self._medium_mapper = medium_mapper
        self._storage_type = storage_type
        self._supported_mediums = supported_mediums
        self._descriptor = descriptor
        self._group_by_idx: dict[int, KvCacheGroupSpec] | None = (
            {spec.group_idx: spec for spec in descriptor.groups}
            # An empty metadata tuple is a valid topology; it registers the
            # same default location spec as the no-metadata path because KVCM
            # rejects empty location_spec_infos.
            if descriptor.groups
            else None
        )
        self._manager_client: AbstractKvCacheManagerClient = (
            manager_client or self._create_manager_client()
        )
        self._host_ip_port_value: str | None = None
        self._spec_cache: dict[tuple[str, int | None], list[dict[str, str]]] = {}
        self._heartbeat_task: asyncio.Task[None] | None = None
        self._engine_config: dict[str, Any] = _get_engine_config_from_env()
        self._engine_config_fallback_warned = False
        self._payload_fallback_logged: set[int] = set()
        self._on_snapshot_required = on_snapshot_required
        # Dedup latch: True once on_snapshot_required has fired for the
        # current streak of snapshot_required=True heartbeat responses.
        # Re-armed when the response stops requesting a snapshot or after a
        # successful re-registration (see _handle_heartbeat_snapshot_required).
        self._snapshot_signal_delivered = False
        self._snapshot_signal_failure_count = 0
        self._last_snapshot_signal_warn_s = 0.0
        self._registered = False
        self._started = False

    # ------------------------------------------------------------------
    # Registration state
    # ------------------------------------------------------------------

    def _set_registration_state(self, registered: bool) -> None:
        """Transition registration state, deduplicating equal values."""
        if self._registered == registered:
            return
        from_state = "registered" if self._registered else "unregistered"
        to_state = "registered" if registered else "unregistered"
        self._registered = registered
        report_registration_transition(from_state, to_state)
        logger.info(
            "kvcm registration state changed",
            step="kvcm_register",
            tags={"registered": registered},
        )

    @property
    def is_registered(self) -> bool:
        """Whether the client is currently registered with KVCM."""
        return self._registered

    def _base_url(self) -> str:
        if not self._config.kvcm_base_url:
            raise ValueError("kvcm_base_url is required")
        return self._config.kvcm_base_url

    def _create_manager_client(self) -> AbstractKvCacheManagerClient:
        if self._config.kvcm_protocol == "http":
            return HttpKvCacheManagerClient(
                self._base_url(),
                request_timeout_seconds=self._config.kvcm_request_timeout_s,
            )
        if self._config.kvcm_protocol == "grpc":
            from subscriber.kvcm.grpc_manager_client import GrpcKvCacheManagerClient

            return GrpcKvCacheManagerClient(
                self._base_url(),
                request_timeout_seconds=self._config.kvcm_request_timeout_s,
            )
        raise ValueError(f"unsupported kvcm_protocol: {self._config.kvcm_protocol}")

    def _instance_group(self) -> str:
        return self._config.kvcm_instance_group

    def _instance_id(self) -> str:
        deployment = self._require_deployment_name()
        return f"{deployment}_{self._effective_block_size()}"

    @staticmethod
    def _require_deployment_name() -> str:
        """Return the unique deployment identity, rejecting a blank value.

        A blank SPECTRUM_DEPLOYMENT_NAME would produce a degenerate
        instance_id such as ``_16`` shared by every replica missing the
        variable, breaking the cross-instance KVCache isolation invariant.
        """

        deployment = os.environ.get("SPECTRUM_DEPLOYMENT_NAME", "").strip()
        if not deployment:
            raise ValueError(
                "SPECTRUM_DEPLOYMENT_NAME must be set to a unique deployment "
                "identity; a blank value would share one KVCM instance_id "
                "across unrelated engine instances"
            )
        return deployment

    def _host_ip_port(self) -> str:
        if self._host_ip_port_value is not None:
            return self._host_ip_port_value
        raise RuntimeError("kvcm client host identity has not been resolved")

    def _trace_id(self, operation: str) -> str:
        return f"subscriber_{operation}_{time.monotonic_ns()}"

    @staticmethod
    def _config_int(cfg: dict[str, Any], key: str, default: int = 1) -> int:
        raw = cfg.get(key)
        if isinstance(raw, bool) or not isinstance(raw, int) or raw <= 0:
            return default
        return raw

    def _effective_block_size(self) -> int:
        """Return the runtime block size used for KVCM instance identity.

        The engine metadata describes the cache topology that is actually
        running, which can differ from ``DS_LLM_ENGINE_CONFIG`` after runtime
        calculation. When groups have different token spans, use their minimum
        so the one KVCM instance identity is compatible with every group. The
        environment value is only a fallback for engines without metadata.
        """

        if self._group_by_idx:
            block_sizes = {
                idx: spec.block_size for idx, spec in self._group_by_idx.items()
            }
            effective = min(block_sizes.values())
            if len(set(block_sizes.values())) > 1:
                logger.warning(
                    "group metadata contains heterogeneous block_sizes; using min",
                    step="kvcm_register",
                    tags={
                        "block_sizes": block_sizes,
                        "effective_block_size": effective,
                    },
                )
            return effective
        block_size = self._config_int(self._engine_config, "block_size")
        if not self._engine_config_fallback_warned:
            logger.warning(
                "kv cache metadata unavailable; using engine config "
                "block_size fallback",
                step="kvcm_register",
                tags={"block_size": block_size},
            )
            self._engine_config_fallback_warned = True
        return block_size

    def _fallback_location_spec_name(self) -> str:
        return f"vllm_{self._effective_block_size()}"

    @staticmethod
    def _group_location_spec_name(
        spec: KvCacheGroupSpec,
        kind_categories: Mapping[str, str],
    ) -> str:
        """Derive one group-aware location-spec name or reject its kind."""

        category = kind_categories.get(spec.kind)
        if category is None:
            raise MetadataProtocolError(
                f"cannot classify component kind {spec.kind!r} for KVCM"
            )
        return f"{category}{spec.group_idx}"

    @classmethod
    def validate_descriptor_location_specs(
        cls,
        config: SubscriberConfig,
        descriptor: KvCacheDescriptor,
    ) -> None:
        """Validate bootstrap-derived names without constructing a manager client."""

        kind_categories = effective_attention_type_categories(
            config.extra_attention_types
        )
        for spec in descriptor.groups:
            cls._group_location_spec_name(spec, kind_categories)

    def _location_spec_name(self, group_idx: int | None = None) -> str:
        if self._group_by_idx is None:
            return self._fallback_location_spec_name()
        if group_idx is None:
            raise MetadataProtocolError(
                "event is missing component identity for group-aware KVCM"
            )
        spec = self._group_by_idx.get(group_idx)
        if spec is None:
            raise MetadataProtocolError(
                f"group_idx {group_idx} is not present in KV cache bootstrap"
            )
        return self._group_location_spec_name(spec, self._kind_categories)

    def _location_specs(
        self,
    ) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
        """Build ``location_spec_infos`` and ``location_spec_groups``.

        With group metadata, each group becomes its own location spec named
        ``{category}{group_idx}`` sized by that group's complete payload bytes,
        and each spec gets its own single-member location spec group of
        the same name. Without metadata a single ``default`` group is used,
        matching pre-hybrid engines.
        """

        if self._group_by_idx is None:
            block_size = self._effective_block_size()
            name = self._location_spec_name()
            infos: list[dict[str, object]] = [{"name": name, "size": block_size}]
            groups: list[dict[str, object]] = [
                {"name": "default", "spec_names": [name]}
            ]
            return infos, groups

        infos = []
        groups = []
        for group_idx in sorted(self._group_by_idx):
            spec = self._group_by_idx[group_idx]
            name = self._location_spec_name(spec.group_idx)
            payload_size = spec.group_payload_size_bytes
            if payload_size is None:
                payload_size = spec.block_size
                if spec.group_idx not in self._payload_fallback_logged:
                    self._payload_fallback_logged.add(spec.group_idx)
                    logger.info(
                        "Engine component payload size unavailable; using block size",
                        step="kvcm_register",
                        tags={
                            "component_id": spec.group_idx,
                            "block_size": spec.block_size,
                        },
                    )
            infos.append({"name": name, "size": payload_size})
            groups.append({"name": name, "spec_names": [name]})
        return infos, groups

    def validate_location_specs(self) -> None:
        """Validate bootstrap-derived KVCM specs without starting the client."""

        self._location_specs()

    def _register_instance_request(self) -> dict[str, object]:
        cfg = self._engine_config
        block_size = self._effective_block_size()
        location_spec_infos, location_spec_groups = self._location_specs()
        tp_size = self._config_int(cfg, "tensor_parallel_size")
        dp_size = self._config_int(cfg, "data_parallel_size")
        pp_size = self._config_int(cfg, "pipeline_parallel_size")
        return {
            "trace_id": self._trace_id("register_instance"),
            "instance_group": self._instance_group(),
            "instance_id": self._instance_id(),
            "block_size": block_size,
            "location_spec_infos": location_spec_infos,
            "model_deployment": {
                "model_name": "default",
                "dtype": "bytes",
                "use_mla": False,
                "tp_size": tp_size,
                "dp_size": dp_size,
                "lora_name": "",
                "pp_size": pp_size,
                "extra": "",
                "user_data": "",
                "use_eagle_pop": self._descriptor.use_eagle_pop,
            },
            "location_spec_groups": location_spec_groups,
            "default_query_type": self._config.kvcm_query_type,
        }

    def _node_register_event(self) -> dict[str, object]:
        return {
            "event_type": KvcmReportEventType.NODE_REGISTER,
            "node_register": {"mediums": self._supported_mediums},
        }

    def _heartbeat_event(self) -> dict[str, object]:
        return {
            "event_type": KvcmReportEventType.HEARTBEAT,
            "heartbeat": {"system_status": {}},
        }

    def _report_event_request(
        self,
        events: list[dict[str, object]],
        *,
        operation: str = "report_event",
        trace_id: str | None = None,
    ) -> dict[str, object]:
        return {
            "trace_id": trace_id if trace_id is not None else self._trace_id(operation),
            "instance_id": self._instance_id(),
            "host_ip_port": self._host_ip_port(),
            "events": events,
            "storage_type": self._storage_type,
        }

    def _block_specs(
        self, medium: str, group_idx: int | None = None
    ) -> list[dict[str, str]]:
        key = (medium, group_idx)
        cached = self._spec_cache.get(key)
        if cached is not None:
            return cached
        spec = [
            {
                "name": self._location_spec_name(group_idx),
                "uri": (
                    f"{self._config.engine_type}://{self._host_ip_port()}/{medium}"
                ),
            }
        ]
        self._spec_cache[key] = spec
        return spec

    def _block_spec_names(self, group_idx: int | None = None) -> list[str]:
        return [self._location_spec_name(group_idx)]

    def _report_events_for_batches(
        self, batches: list[KVEventBatch]
    ) -> list[dict[str, object]]:
        return expand_report_events(
            batches,
            medium_mapper=self._medium_mapper,
            block_specs=self._block_specs,
            block_spec_names=self._block_spec_names,
        )

    @staticmethod
    def _metadata_protocol_rejection(
        error: MetadataProtocolError,
    ) -> KvcmReportRejectedError:
        """Classify a local descriptor/event mismatch without issuing an RPC."""

        return KvcmReportRejectedError(
            f"KVCM metadata protocol error: {error}",
            status_code="METADATA_PROTOCOL",
            reason="metadata_protocol",
        )

    async def start(self) -> None:
        self._require_deployment_name()
        # host_port is guaranteed non-None by SubscriberConfig.validate().
        self._host_ip_port_value = await resolve_host_ip_port(
            cast(int, self._config.host_port)
        )
        await self._manager_client.start()
        self._started = True
        if await self._manager_is_ready():
            await self._register_and_report_node()
        else:
            logger.warning(
                "kvcm has no available endpoint; starting in not-ready state",
                step="kvcm_register",
            )
        self._heartbeat_task = asyncio.create_task(self._heartbeat_loop())

    async def _manager_is_ready(self) -> bool:
        if self._manager_client is None:
            return False
        return await self._manager_client.is_ready()

    async def _register_instance(self) -> dict[str, object]:
        if not self._started or self._manager_client is None:
            raise RuntimeError("kvcm client has not been started")
        request = self._register_instance_request()
        if logger.is_debug_enabled():
            logger.debug(
                "kvcm register_instance request",
                step="kvcm_register",
                tags={"request": request},
            )
        await self._manager_client.register_instance(request)
        return request

    async def _register_and_report_node(self) -> None:
        """Perform the two-step kvcm registration: the register_instance RPC
        followed by the NODE_REGISTER event report. Registration state is
        transitioned via ``_set_registration_state`` only after both steps
        succeed. Success also re-arms the heartbeat snapshot signal
        (``_rearm_snapshot_signal``): after re-registration kvcm's view
        of this instance is fresh, so a previously serviced
        ``snapshot_required`` must be able to fire again."""

        try:
            await self._register_instance()
        except Exception as exc:
            logger.warning(
                "kvcm register_instance failed (%s: %s)",
                type(exc).__name__,
                exc,
                step="kvcm_register",
                tags={"phase": "register_instance"},
                exc_info=exc,
            )
            return
        try:
            await self._report_events([self._node_register_event()])
        except Exception as exc:
            logger.warning(
                "kvcm node_register report failed (%s: %s)",
                type(exc).__name__,
                exc,
                step="kvcm_register",
                tags={"phase": "node_register"},
                exc_info=exc,
            )
            return
        self._rearm_snapshot_signal()
        self._set_registration_state(True)

    def _rearm_snapshot_signal(self) -> None:
        """Reset the snapshot-signal streak state (dedup latch and failure
        count) so the next ``snapshot_required=True`` response starts a fresh
        streak with an immediate first-failure warning."""
        self._snapshot_signal_delivered = False
        self._snapshot_signal_failure_count = 0

    def _handle_heartbeat_snapshot_required(self, response: dict[str, Any]) -> None:
        """Fire ``_on_snapshot_required`` at most once per consecutive streak
        of ``snapshot_required=True`` heartbeat responses.

        The dedup flag (``_snapshot_signal_delivered``) resets when a
        response no longer requests a snapshot (strict ``is True``; missing or
        non-bool values count as not requested) and after a successful
        re-registration (``_register_and_report_node``). If the callback
        raises, the flag stays unset so the next heartbeat retries; the
        failure warning is rate-limited. Best-effort: never raises into the
        heartbeat loop.
        """
        if self._on_snapshot_required is None:
            return
        if response.get("snapshot_required") is not True:
            self._rearm_snapshot_signal()
            return
        if self._snapshot_signal_delivered:
            return
        try:
            self._on_snapshot_required()
        except Exception as exc:
            self._snapshot_signal_failure_count += 1
            now_s = time.monotonic()
            if (
                self._snapshot_signal_failure_count == 1
                or now_s - self._last_snapshot_signal_warn_s
                >= _SNAPSHOT_SIGNAL_WARN_INTERVAL_S
            ):
                self._last_snapshot_signal_warn_s = now_s
                logger.warning(
                    "failed to request immediate snapshot from kvcm heartbeat; "
                    "continuing heartbeat",
                    step="snapshot_signal",
                    tags={
                        "error": exc.__class__.__name__,
                        "message": str(exc),
                        "failure_count": self._snapshot_signal_failure_count,
                    },
                    exc_info=exc,
                )
            return
        self._snapshot_signal_failure_count = 0
        self._snapshot_signal_delivered = True

    async def _heartbeat_loop(self) -> None:
        not_ready_since_s: float | None = None
        last_not_ready_warn_s = 0.0
        while True:
            await asyncio.sleep(self._config.kvcm_heartbeat_interval_s)
            if not self._registered:
                if not await self._manager_is_ready():
                    now_s = time.monotonic()
                    if not_ready_since_s is None:
                        not_ready_since_s = now_s
                        last_not_ready_warn_s = now_s
                    elif now_s - last_not_ready_warn_s >= _UNREGISTERED_WARN_INTERVAL_S:
                        last_not_ready_warn_s = now_s
                        logger.warning(
                            "kvcm still unregistered; no available endpoint",
                            step="kvcm_register",
                            tags={
                                "unregistered_for_s": round(
                                    now_s - not_ready_since_s, 1
                                ),
                            },
                        )
                    continue
                not_ready_since_s = None
                await self._register_and_report_node()
                if not self._registered:
                    continue
                report_registration_recovery()
                logger.info(
                    "kvcm registration recovered",
                    step="kvcm_register",
                )
            not_ready_since_s = None

            try:
                response = await self._report_events([self._heartbeat_event()])
                self._handle_heartbeat_snapshot_required(response)
                report_heartbeat(success=True)
            except Exception as exc:
                self._set_registration_state(False)
                report_heartbeat(success=False)
                logger.warning(
                    "kvcm heartbeat report failed (%s: %s)",
                    type(exc).__name__,
                    exc,
                    step="kvcm_heartbeat",
                    exc_info=exc,
                )

    async def _report_events(self, events: list[dict[str, object]]) -> dict[str, Any]:
        """Send registration or heartbeat events outside the forwarding seam."""

        if not self._started or self._manager_client is None:
            raise RuntimeError("kvcm client has not been started")
        return await self._manager_client.report_event(
            self._report_event_request(events)
        )

    async def _report_events_checked(
        self, request: dict[str, object], epoch: int
    ) -> dict[str, Any]:
        """Send a prepared ReportEvent request and classify transport failures.

        Transports raise ``KvcmResponseRejectedError`` for non-OK responses
        (rejected reports); those become :class:`KvcmReportRejectedError`. Any
        other exception is a retryable transport failure and becomes
        :class:`KvcmUnavailableError`. A response carrying partial
        ``item_results`` is also rejected. Registration state is deliberately
        NOT changed here; the heartbeat loop owns reconnection and
        re-registration independently.
        """

        try:
            response = await self._manager_client.report_event(
                request, check_response=True
            )
        except asyncio.CancelledError:
            raise
        except KvcmResponseRejectedError as exc:
            raise KvcmReportRejectedError(
                str(exc),
                status_code=exc.status_code,
                reason="rejected",
                retry_count=exc.retry_count,
                request_bytes=exc.request_bytes,
                wire_encode_ms=exc.wire_encode_ms,
                grpc_call_ms=exc.grpc_call_ms,
            ) from exc
        except KvcmUnavailableError as exc:
            raise KvcmUnavailableError(
                str(exc),
                status_code=exc.status_code,
                reason=exc.reason,
                retry_count=exc.retry_count,
                request_bytes=exc.request_bytes,
                wire_encode_ms=exc.wire_encode_ms,
                grpc_call_ms=exc.grpc_call_ms,
            ) from exc
        except grpc.aio.AioRpcError as exc:
            diagnostics = report_event_transport_diagnostics(exc)
            raise KvcmUnavailableError(
                str(exc),
                status_code=f"GRPC_{exc.code().name}",
                reason="transport",
                request_bytes=diagnostics.request_bytes,
                wire_encode_ms=diagnostics.wire_encode_ms,
                grpc_call_ms=diagnostics.grpc_call_ms,
            ) from exc
        except Exception as exc:
            if _REPORT_REJECTED_PREFIX in str(exc):
                raise KvcmReportRejectedError(
                    str(exc),
                    reason="rejected",
                ) from exc
            raise KvcmUnavailableError(str(exc), reason="transport") from exc
        item_results = (
            response.get("item_results") if isinstance(response, dict) else None
        )
        if item_results:
            trace_id = request.get("trace_id")
            logger.warning(
                "kvcm report_event returned partial item results",
                step="kvcm_send",
                tags={
                    "epoch": epoch,
                    "trace_id": trace_id,
                    "item_results": item_results,
                },
            )
            raise KvcmReportRejectedError(
                f"kvcm report_event returned partial item_results: {item_results!r}",
                status_code="ITEM_RESULTS",
                reason="partial_results",
                retry_count=_response_retry_count(response),
                request_bytes=response.get("_subscriber_request_bytes"),
                wire_encode_ms=response.get("_subscriber_wire_encode_ms"),
                grpc_call_ms=response.get("_subscriber_grpc_call_ms"),
            )
        return response

    def _ensure_send_ready(self) -> None:
        """Raise if the client is not in a state that can send reports."""

        if not self._started or self._manager_client is None:
            raise RuntimeError("kvcm client has not been started")
        if not self._registered:
            raise KvcmUnavailableError("kvcm client is not ready")

    async def _send_report_and_record_result(
        self,
        request: dict[str, object],
        epoch: int,
        *,
        log_message: str,
        extra_tags: dict[str, object],
        send_started_at: float,
        request_telemetry: BatchTelemetry | None = None,
        additional_request_gauges: Mapping[str, float] | None = None,
    ) -> dict[str, Any]:
        """Send one logical ReportEvent and record its terminal telemetry.

        This is the forwarding seam's one owner for final request/result
        observations: request/failure/retry counters, logical call duration,
        gRPC transport diagnostics, and optional path-specific gauges. It only
        appends to ``BatchTelemetry``; ``MetricsReporter`` performs backend I/O
        later. ``extra_tags`` are log-only path diagnostics, not metric labels.
        """

        report_event_started_at = time.monotonic()
        status_code = "UNKNOWN"
        failure_reason: str | None = None
        retry_count = 0
        all_request_gauges = dict(additional_request_gauges or {})
        try:
            response = await self._report_events_checked(request, epoch)
            status_code = _report_status_code(response)
            retry_count = _response_retry_count(response)
            all_request_gauges.update(_response_transport_gauges(response))
            return response
        except KvcmReportError as exc:
            status_code = exc.status_code
            failure_reason = exc.reason
            retry_count = exc.retry_count
            all_request_gauges.update(
                _transport_gauges(
                    request_bytes=exc.request_bytes,
                    wire_encode_ms=exc.wire_encode_ms,
                    grpc_call_ms=exc.grpc_call_ms,
                )
            )
            raise
        finally:
            report_event_call_ms = (time.monotonic() - report_event_started_at) * 1000
            if request_telemetry is not None:
                request_telemetry.count(
                    "kvcm_report_event_request_count",
                    1,
                    tags={"status_code": status_code},
                )
                if failure_reason is not None:
                    request_telemetry.count(
                        "kvcm_report_event_failure_count",
                        1,
                        tags={
                            "status_code": status_code,
                            "reason": failure_reason,
                        },
                    )
                if retry_count:
                    request_telemetry.count(
                        "kvcm_report_event_retry_count",
                        retry_count,
                        tags={"reason": "SERVER_NOT_LEADER"},
                    )
                request_telemetry.gauge(
                    "kvcm_report_event_call_ms",
                    report_event_call_ms,
                    tags={"status_code": status_code},
                )
                for name, value in all_request_gauges.items():
                    request_telemetry.gauge(
                        name,
                        value,
                        tags={"status_code": status_code},
                    )
            if logger.is_debug_enabled():
                tags: dict[str, object] = {"epoch": epoch}
                trace_id = request.get("trace_id")
                if trace_id is not None:
                    tags["trace_id"] = trace_id
                tags.update(extra_tags)
                tags["status_code"] = status_code
                tags["kvcm_report_event_call_ms"] = round(report_event_call_ms, 3)
                tags["kvcm_send_total_ms"] = round(
                    (time.monotonic() - send_started_at) * 1000, 3
                )
                logger.debug(log_message, step="kvcm_send", tags=tags)

    async def report_kv_events(
        self,
        batches: list[KVEventBatch],
        epoch: int,
        telemetries: list[BatchTelemetry] | None = None,
        trace_id: str | None = None,
        *,
        reregister_after_host_down: bool = True,
    ) -> bool:
        """Report incremental KV event batches to kvcm.

        Returns ``True`` when the KVCM response indicates a full snapshot is
        required (``snapshot_required`` field is boolean ``True``). Missing or
        non-boolean values (e.g. from older KVCM versions) return ``False``.

        When ``telemetries`` is provided, marks two sequential stages on each:
        ``expand`` (payload serialization) and ``kvcm_send`` (KVCM ReportEvent
        调用). This mirrors the stage-marking contract of
        :meth:`report_snapshot`.

        Engine-originated ``AllBlocksCleared`` means that the cache was reset
        while the engine remains available. KVCM represents the reset as
        ``EVENT_HOST_DOWN``, which removes the node, so the default path sends
        a standalone ``EVENT_NODE_REGISTER`` immediately afterwards. Health
        and shutdown callers pass ``reregister_after_host_down=False`` because
        their host-down is terminal for the current engine epoch.
        """

        self._ensure_send_ready()
        send_started_at = time.monotonic()
        try:
            events = self._report_events_for_batches(batches)
        except MetadataProtocolError as exc:
            if telemetries:
                for telemetry in telemetries:
                    telemetry.mark("expand")
            raise self._metadata_protocol_rejection(exc) from exc
        request_telemetry = telemetries[0] if telemetries else None
        if telemetries:
            for t in telemetries:
                t.mark("expand")
        event_expand_ms = (time.monotonic() - send_started_at) * 1000
        if not events:
            if logger.is_debug_enabled():
                logger.debug(
                    "kvcm send skipped because batch group has no reportable events",
                    step="kvcm_send",
                    tags={"epoch": epoch},
                )
            return False
        request_event_groups = split_report_event_requests(events)
        if reregister_after_host_down:
            normalized_groups: list[list[dict[str, object]]] = []
            for request_events in request_event_groups:
                normalized_groups.append(request_events)
                if request_events[0].get("event_type") == KvcmReportEventType.HOST_DOWN:
                    normalized_groups.append([self._node_register_event()])
            request_event_groups = normalized_groups
        if request_telemetry is not None:
            request_telemetry.gauge(
                "kvcm_report_event_count",
                sum(len(group) for group in request_event_groups),
            )
        counts = source_counts(batches)
        snapshot_required = False
        try:
            for request_part_index, request_events in enumerate(request_event_groups):
                request = self._report_event_request(
                    request_events,
                    trace_id=trace_id,
                )
                response = await self._send_report_and_record_result(
                    request,
                    epoch,
                    log_message="kvcm report_event timing",
                    extra_tags={
                        "source_batch_count": counts.batch_count,
                        "source_event_count": counts.event_count,
                        "report_event_count": len(request_events),
                        "request_part_index": request_part_index,
                        "request_part_count": len(request_event_groups),
                        "event_expand_ms": round(event_expand_ms, 3),
                    },
                    send_started_at=time.monotonic(),
                    request_telemetry=request_telemetry,
                )
                snapshot_required = (
                    response.get("snapshot_required") is True or snapshot_required
                )
        finally:
            if telemetries:
                for t in telemetries:
                    t.mark("kvcm_send")
        return snapshot_required

    async def report_snapshot(
        self,
        batches: list[KVEventBatch],
        epoch: int,
        telemetry: BatchTelemetry | None = None,
        trace_id: str | None = None,
    ) -> None:
        """Report a full KV cache block snapshot to kvcm.

        Contract mirrors :meth:`report_kv_events`: gated on registration, lossy error
        classification (rejected vs unavailable), partial ``item_results``
        rejection, and debug timing diagnostics. All ``BlockSnapshot`` events in
        ``batches`` are flattened into a single ``EVENT_BLOCK_SNAPSHOT`` report;
        when no snapshot block is present the report is skipped entirely.

        When ``telemetry`` is provided, marks two sequential stages:
        ``expand`` (fused collect + merge) and ``kvcm_send`` (KVCM ReportEvent
        调用).
        """

        self._ensure_send_ready()
        send_started_at = time.monotonic()
        snapshots = [
            event
            for batch in batches
            for event in batch.events
            if isinstance(event, BlockSnapshot)
        ]
        if not snapshots:
            if logger.is_debug_enabled():
                logger.debug(
                    "kvcm snapshot send skipped because batch has no snapshot blocks",
                    step="kvcm_send",
                    tags={"epoch": epoch},
                )
            return
        try:
            merged = build_merged_snapshot_blocks(
                batches,
                medium_mapper=self._medium_mapper,
                block_specs=self._block_specs,
            )
        except MetadataProtocolError as exc:
            if telemetry is not None:
                telemetry.mark("expand")
            raise self._metadata_protocol_rejection(exc) from exc
        if telemetry is not None:
            telemetry.mark("expand")
        snapshot_block_count = sum(len(snapshot.items) for snapshot in snapshots)
        event: dict[str, object] = {
            "event_type": KvcmReportEventType.BLOCK_SNAPSHOT,
            "block_snapshot": {"blocks": merged},
        }
        request = self._report_event_request(
            [event], operation="report_snapshot", trace_id=trace_id
        )
        counts = source_counts(batches)
        event_expand_ms = (time.monotonic() - send_started_at) * 1000
        try:
            await self._send_report_and_record_result(
                request,
                epoch,
                log_message="kvcm snapshot report_event timing",
                extra_tags={
                    "source_batch_count": counts.batch_count,
                    "source_event_count": counts.event_count,
                    "snapshot_block_count": snapshot_block_count,
                    "event_expand_ms": round(event_expand_ms, 3),
                },
                send_started_at=send_started_at,
                request_telemetry=telemetry,
                additional_request_gauges={
                    "kvcm_snapshot_source_block_count": float(snapshot_block_count),
                    "kvcm_snapshot_merged_block_count": float(len(merged)),
                },
            )
        finally:
            if telemetry is not None:
                telemetry.mark("kvcm_send")

    async def close(self) -> None:
        if self._heartbeat_task is not None:
            self._heartbeat_task.cancel()
            try:
                await self._heartbeat_task
            except asyncio.CancelledError:
                pass
            self._heartbeat_task = None
        if self._manager_client is not None:
            await self._manager_client.close()
        self._set_registration_state(False)
        self._started = False
