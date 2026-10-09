"""Ordered subscriber-state reporting to DashServing.

``DashservingStateReporter`` owns the subscriber's local phase
(starting/active/inactive/failed), a strictly monotonic ``seq_id`` allocator,
and the active-heartbeat task. Every logical report (startup active, each
heartbeat, terminal failed/inactive) consumes a fresh sequence number under a
single short-held ``asyncio.Lock``; the lock is released before any HTTP await.

Ordering authority is DashServing's ``seq_id`` comparison, not a network-duration
local lock: overlapping POSTs are linearized server-side, so a delayed lower-seq
active cannot resurrect state after a higher-seq inactive.

Unsupported topology (fail-safe, see plan §1.2): sequence-only ordering assumes a
single reporter per DashServing process lifetime. An independent subscriber
restart would reset ``seq_id`` to 1 and be rejected as stale by the surviving
DashServing. That case must add ``session_id + seq_id`` or a persisted monotonic
counter; until then it is treated as a loud invariant violation, never silent
success. TODO(independent-restart): add session identity before supporting it.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Awaitable, Callable
from enum import Enum

import httpx

from subscriber import logger
from subscriber.config import SubscriberConfig
from subscriber.metrics import (
    report_dashserving_heartbeat,
    report_dashserving_state_report,
    report_subscriber_phase_transition,
)

_STEP = "state_report"

# After the first heartbeat failure warning, repeat a summary at most this often.
_FAILURE_SUMMARY_INTERVAL_S = 30.0


class SubscriberPhase(Enum):
    """Local subscriber phase reported to DashServing."""

    STARTING = "starting"
    ACTIVE = "active"
    INACTIVE = "inactive"
    FAILED = "failed"


class StateReportError(RuntimeError):
    """A logical state report was not accepted by DashServing."""


class DashservingStateReporter:
    """Report subscriber phase to DashServing with monotonic sequence ordering.

    The reporter serializes sequence allocation and local phase mutation under one
    ``asyncio.Lock`` but never holds it across an HTTP call. ``start_heartbeat``
    creates and stores its own task; ``stop_heartbeat`` cancels and awaits it.
    Callers must not assign ``_heartbeat_task`` directly.
    """

    def __init__(
        self,
        config: SubscriberConfig,
        *,
        http_client: httpx.AsyncClient | None = None,
        clock: Callable[[], float] | None = None,
        sleep: Callable[[float], Awaitable[None]] | None = None,
    ) -> None:
        self._state_url = config.subscriber_state_url
        self._heartbeat_interval_s = config.subscriber_heartbeat_interval_s
        self._request_timeout_s = config.subscriber_state_request_timeout_s
        self._shutdown_timeout_s = config.subscriber_shutdown_report_timeout_s

        self._http_client = http_client or httpx.AsyncClient()
        self._owns_client = http_client is None
        self._clock = clock or time.monotonic
        self._sleep = sleep or asyncio.sleep

        self._lock = asyncio.Lock()
        self._next_seq_id = 1
        self._phase = SubscriberPhase.STARTING
        self._terminal = False
        self._heartbeat_task: asyncio.Task[None] | None = None
        self._closed = False

        # Heartbeat failure suppression bookkeeping (touched only by the single
        # heartbeat task / direct calls; not shared across concurrent producers).
        self._consecutive_failures = 0
        self._first_failure_at: float | None = None
        self._last_summary_at: float | None = None

    @property
    def phase(self) -> SubscriberPhase:
        """Current local phase."""

        return self._phase

    @property
    def next_seq_id(self) -> int:
        """Sequence number that the next logical report will consume."""

        return self._next_seq_id

    async def report_active(self) -> None:
        """Report ``active`` and await an accepted 2xx acknowledgement.

        Raises ``httpx`` transport errors, ``StateReportError`` for a non-2xx or
        malformed response, or for a logical rejection (``accepted=false``). A
        rejection is a failed report, never a silent success, so startup can rely
        on a clean return as "active confirmed".
        """

        async with self._lock:
            if self._terminal:
                raise StateReportError(
                    "cannot report active: reporter is terminal "
                    f"(phase={self._phase.value})"
                )
            seq = self._next_seq_id
            self._next_seq_id += 1

        # The lock is released before the HTTP await; DashServing's seq_id
        # comparison, not a network-duration lock, linearizes overlapping reports.
        started_at = time.monotonic()
        try:
            ack = await self._post("active", seq, self._request_timeout_s)
            self._require_accepted(ack, "active", seq)
        except Exception:
            self._record_state_report(
                "active", "error", (time.monotonic() - started_at) * 1000
            )
            raise
        self._record_state_report(
            "active", "accepted", (time.monotonic() - started_at) * 1000
        )

        # Transition only on an accepted acknowledgement, and only STARTING->ACTIVE.
        # A newer terminal transition (inactive/failed) may have landed while this
        # report was in flight; guarding on STARTING ensures a late accepted ack
        # never rolls back the local phase.
        async with self._lock:
            if self._phase is SubscriberPhase.STARTING:
                self._phase = SubscriberPhase.ACTIVE
                report_subscriber_phase_transition("starting", "active")
                logger.info(
                    "subscriber phase transition",
                    step=_STEP,
                    tags={
                        "from": SubscriberPhase.STARTING.value,
                        "to": "active",
                        "seq_id": seq,
                    },
                )

        logger.debug(
            "state report accepted",
            step=_STEP,
            tags={"state": "active", "seq_id": seq, "last_seq_id": ack.last_seq_id},
        )

    async def report_failed(self, reason: str) -> None:
        """Report terminal ``failed``; legal only while locally starting.

        Best-effort: a transport failure is logged and swallowed so it cannot mask
        the original fatal startup error.
        """

        async with self._lock:
            if self._phase is not SubscriberPhase.STARTING:
                raise StateReportError(
                    "report_failed is only allowed while starting "
                    f"(phase={self._phase.value})"
                )
            seq = self._next_seq_id
            self._next_seq_id += 1
            self._phase = SubscriberPhase.FAILED
            self._terminal = True

        report_subscriber_phase_transition("starting", "failed")
        logger.info(
            "subscriber phase transition",
            step=_STEP,
            tags={
                "from": SubscriberPhase.STARTING.value,
                "to": "failed",
                "seq_id": seq,
                "reason": reason,
            },
        )
        await self._best_effort_post("failed", seq, self._request_timeout_s)

    def start_heartbeat(self) -> None:
        """Create and store the active-heartbeat task; reject a second start.

        Synchronous and atomic within the event loop (no await point), so the
        ownership check needs no lock. The task is created here; callers never
        assign ``_heartbeat_task``.
        """

        if self._heartbeat_task is not None:
            raise StateReportError("heartbeat already started")
        if self._terminal:
            raise StateReportError(
                "cannot start heartbeat: reporter is terminal "
                f"(phase={self._phase.value})"
            )
        self._heartbeat_task = asyncio.create_task(
            self._heartbeat_loop(), name="subscriber-state-heartbeat"
        )

    async def stop_heartbeat(self) -> None:
        """Cancel and await the heartbeat task, if any. Idempotent."""

        async with self._lock:
            task = self._heartbeat_task
            self._heartbeat_task = None
        if task is None:
            return
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

    async def report_shutdown_inactive(self, reason: str) -> None:
        """Gracefully report terminal ``inactive`` with a bounded best-effort POST.

        Stops the heartbeat first so no racing active heartbeat can take a higher
        sequence than this inactive. Makes the reporter terminal; never raises.
        """

        await self.stop_heartbeat()
        async with self._lock:
            if self._terminal:
                return
            seq = self._next_seq_id
            self._next_seq_id += 1
            previous = self._phase
            self._phase = SubscriberPhase.INACTIVE
            self._terminal = True

        report_subscriber_phase_transition(previous.value, "inactive")
        logger.info(
            "subscriber phase transition",
            step=_STEP,
            tags={
                "from": previous.value,
                "to": "inactive",
                "seq_id": seq,
                "reason": reason,
            },
        )
        await self._best_effort_post("inactive", seq, self._shutdown_timeout_s)

    async def close(self) -> None:
        """Stop the heartbeat and release the owned HTTP client. Idempotent."""

        if self._closed:
            return
        self._closed = True
        await self.stop_heartbeat()
        if self._owns_client:
            await self._http_client.aclose()

    async def _heartbeat_loop(self) -> None:
        try:
            while True:
                await self._sleep(self._heartbeat_interval_s)
                await self._send_heartbeat_once()
        except asyncio.CancelledError:
            logger.debug("state heartbeat cancelled", step=_STEP, tags={})
            raise

    async def _send_heartbeat_once(self) -> None:
        async with self._lock:
            if self._terminal:
                return
            seq = self._next_seq_id
            self._next_seq_id += 1

        started_at = time.monotonic()
        try:
            ack = await self._post("active", seq, self._request_timeout_s)
            self._require_accepted(ack, "active", seq)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            self._record_state_report(
                "active", "error", (time.monotonic() - started_at) * 1000
            )
            self._note_heartbeat_failure(
                error=exc.__class__.__name__, message=str(exc), seq_id=seq
            )
            return

        self._record_state_report(
            "active", "accepted", (time.monotonic() - started_at) * 1000
        )
        self._note_heartbeat_success()
        logger.debug(
            "state heartbeat accepted",
            step=_STEP,
            tags={"state": "active", "seq_id": seq, "last_seq_id": ack.last_seq_id},
        )

    def _note_heartbeat_failure(self, *, error: str, message: str, seq_id: int) -> None:
        report_dashserving_heartbeat(success=False)
        now = self._clock()
        self._consecutive_failures += 1
        if self._first_failure_at is None:
            self._first_failure_at = now
            self._last_summary_at = now
            logger.warning(
                "state heartbeat failed; will retry next interval",
                step=_STEP,
                tags={"error": error, "message": message, "seq_id": seq_id},
            )
            return
        assert self._last_summary_at is not None
        if now - self._last_summary_at >= _FAILURE_SUMMARY_INTERVAL_S:
            self._last_summary_at = now
            logger.warning(
                "state heartbeat still failing",
                step=_STEP,
                tags={
                    "consecutive_failures": self._consecutive_failures,
                    "for_s": round(now - self._first_failure_at, 3),
                    "last_error": error,
                    "last_message": message,
                },
            )

    def _note_heartbeat_success(self) -> None:
        report_dashserving_heartbeat(success=True)
        if self._consecutive_failures > 0:
            logger.info(
                "state heartbeat recovered",
                step=_STEP,
                tags={"previous_consecutive_failures": self._consecutive_failures},
            )
        self._consecutive_failures = 0
        self._first_failure_at = None
        self._last_summary_at = None

    def _record_state_report(self, state: str, result: str, latency_ms: float) -> None:
        """Best-effort metric recording; never raises."""
        report_dashserving_state_report(state, result, latency_ms)

    async def _best_effort_post(self, state: str, seq: int, timeout_s: float) -> None:
        started_at = time.monotonic()
        try:
            ack = await self._post(state, seq, timeout_s)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            self._record_state_report(
                state, "error", (time.monotonic() - started_at) * 1000
            )
            logger.warning(
                "terminal state report failed; DashServing falls back to TTL",
                step=_STEP,
                tags={
                    "state": state,
                    "seq_id": seq,
                    "error": exc.__class__.__name__,
                    "message": str(exc),
                },
            )
            return
        if not ack.accepted:
            self._record_state_report(
                state, "stale", (time.monotonic() - started_at) * 1000
            )
            logger.warning(
                "terminal state report not accepted",
                step=_STEP,
                tags={"state": state, "seq_id": seq, "last_seq_id": ack.last_seq_id},
            )
            return
        self._record_state_report(
            state, "accepted", (time.monotonic() - started_at) * 1000
        )
        logger.debug(
            "state report accepted",
            step=_STEP,
            tags={"state": state, "seq_id": seq, "last_seq_id": ack.last_seq_id},
        )

    def _require_accepted(self, ack: _Ack, state: str, seq: int) -> None:
        if ack.accepted:
            return
        if ack.last_seq_id > seq:
            # A higher sequence already won. Under the single-reporter deployment
            # invariant this is impossible from one process: either an independent
            # restart reset our counter (unsupported, see module TODO) or sequences
            # are corrupted. Loud invariant warning, and a failed logical report.
            logger.warning(
                "state report rejected: server already advanced past our sequence "
                "(invariant violation under single-reporter assumption)",
                step=_STEP,
                tags={"state": state, "seq_id": seq, "last_seq_id": ack.last_seq_id},
            )
        else:
            logger.warning(
                "state report rejected as duplicate/stale",
                step=_STEP,
                tags={"state": state, "seq_id": seq, "last_seq_id": ack.last_seq_id},
            )
        raise StateReportError(
            f"{state} report seq={seq} not accepted "
            f"(server last_seq_id={ack.last_seq_id})"
        )

    async def _post(self, state: str, seq: int, timeout_s: float) -> _Ack:
        response = await self._http_client.post(
            self._state_url,
            json={"state": state, "seq_id": seq},
            timeout=timeout_s,
        )
        response.raise_for_status()
        return _parse_ack(response)


class _Ack:
    """Validated DashServing acknowledgement for one state report."""

    __slots__ = ("accepted", "last_seq_id")

    def __init__(self, accepted: bool, last_seq_id: int) -> None:
        self.accepted = accepted
        self.last_seq_id = last_seq_id


def _parse_ack(response: httpx.Response) -> _Ack:
    try:
        payload = response.json()
    except ValueError as exc:
        raise StateReportError(f"state ack is not valid JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise StateReportError(f"state ack is not a JSON object: {payload!r}")
    accepted = payload.get("accepted")
    last_seq_id = payload.get("last_seq_id")
    if not isinstance(accepted, bool):
        raise StateReportError(f"state ack 'accepted' is not a bool: {payload!r}")
    # bool is a subclass of int; reject it explicitly for last_seq_id.
    if not isinstance(last_seq_id, int) or isinstance(last_seq_id, bool):
        raise StateReportError(f"state ack 'last_seq_id' is not an int: {payload!r}")
    return _Ack(accepted, last_seq_id)
