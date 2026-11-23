"""Typed errors raised when reporting KV events to KVCM.

The forwarding loop catches :class:`KvcmReportError` to drop the affected
batch and continue consuming. This is intentionally lossy so a KVCM
control-plane failure cannot backpressure or stop engine serving. KVCM
availability is not a forwarding gate input; the KVCM heartbeat loop owns
re-registration independently.
"""

from __future__ import annotations

from dataclasses import dataclass

_REPORT_EVENT_TRANSPORT_DIAGNOSTICS_ATTRIBUTE = (
    "_subscriber_report_event_transport_diagnostics"
)


@dataclass(frozen=True)
class ReportEventTransportDiagnostics:
    """Local gRPC ReportEvent diagnostics, independent of response semantics.

    A raw transport exception cannot be changed to a domain error without
    breaking direct gRPC callers. In that narrow case the gRPC adapter stores
    this immutable value on the exception; callers use
    :func:`report_event_transport_diagnostics` instead of depending on an
    adapter-private attribute name.
    """

    request_bytes: int | None = None
    wire_encode_ms: float | None = None
    grpc_call_ms: float | None = None


def attach_report_event_transport_diagnostics(
    error: BaseException,
    diagnostics: ReportEventTransportDiagnostics,
) -> None:
    """Attach the one internal diagnostics value to a raw transport error."""

    setattr(error, _REPORT_EVENT_TRANSPORT_DIAGNOSTICS_ATTRIBUTE, diagnostics)


def report_event_transport_diagnostics(
    error: BaseException,
) -> ReportEventTransportDiagnostics:
    """Return ReportEvent diagnostics attached to a raw transport error.

    Errors from a transport that does not provide gRPC diagnostics return an
    empty value so consumers can preserve their existing fallback behavior.
    """

    value = getattr(error, _REPORT_EVENT_TRANSPORT_DIAGNOSTICS_ATTRIBUTE, None)
    if isinstance(value, ReportEventTransportDiagnostics):
        return value
    return ReportEventTransportDiagnostics()


class KvcmReportError(RuntimeError):
    """Base error for KVCM event reporting failures."""

    def __init__(
        self,
        message: str,
        *,
        status_code: str = "UNKNOWN",
        reason: str = "unknown",
        retry_count: int = 0,
        request_bytes: int | None = None,
        wire_encode_ms: float | None = None,
        grpc_call_ms: float | None = None,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.reason = reason
        self.retry_count = max(0, retry_count)
        self.request_bytes = request_bytes
        self.wire_encode_ms = wire_encode_ms
        self.grpc_call_ms = grpc_call_ms


class KvcmUnavailableError(KvcmReportError):
    """KVCM is not registered or a retryable transport failure occurred.

    Registration state is deliberately NOT changed here: the KVCM heartbeat
    loop owns reconnection and re-registration.
    """


class KvcmReportRejectedError(KvcmReportError):
    """KVCM rejected the report (non-retryable for this batch).

    A rejected/partial report is dropped and observed without changing
    registration.
    """


class KvcmResponseRejectedError(RuntimeError):
    """Non-OK KVCM response before domain-level error classification."""

    def __init__(
        self,
        message: str,
        *,
        status_code: str = "UNKNOWN",
        retry_count: int = 0,
        request_bytes: int | None = None,
        wire_encode_ms: float | None = None,
        grpc_call_ms: float | None = None,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.retry_count = max(0, retry_count)
        self.request_bytes = request_bytes
        self.wire_encode_ms = wire_encode_ms
        self.grpc_call_ms = grpc_call_ms
