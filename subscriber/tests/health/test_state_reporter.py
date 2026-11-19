from __future__ import annotations

import asyncio
import json
from collections.abc import Awaitable, Callable

import httpx
import pytest

from subscriber.config import SubscriberConfig
from subscriber.health.state_reporter import (
    DashservingStateReporter,
    StateReportError,
    SubscriberPhase,
)


class FakeClock:
    """Deterministic monotonic clock the test advances manually."""

    def __init__(self, start: float = 1000.0) -> None:
        self.now = start

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


class FakeTransport(httpx.AsyncBaseTransport):
    """Injected fake transport.

    Default behavior echoes an accepted ack carrying the request's own seq_id.
    Tests can queue explicit responses/exceptions, or install an async handler
    (e.g. one that blocks on an event to simulate an in-flight request).
    """

    def __init__(
        self,
        handler: Callable[[httpx.Request], Awaitable[httpx.Response]] | None = None,
    ) -> None:
        self.requests: list[dict[str, object]] = []
        self.raw: list[httpx.Request] = []
        self._queued: list[httpx.Response | Exception] = []
        self._handler = handler

    def queue(self, *items: httpx.Response | Exception) -> None:
        self._queued.extend(items)

    def states(self) -> list[str]:
        return [str(r["state"]) for r in self.requests]

    def seqs(self) -> list[int]:
        return [int(r["seq_id"]) for r in self.requests]

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        await request.aread()
        self.raw.append(request)
        self.requests.append(json.loads(request.content.decode()))
        if self._queued:
            item = self._queued.pop(0)
            if isinstance(item, Exception):
                raise item
            return item
        if self._handler is not None:
            return await self._handler(request)
        seq = self.requests[-1]["seq_id"]
        return httpx.Response(200, json={"accepted": True, "last_seq_id": seq})


def make_reporter(
    transport: FakeTransport | None = None,
    *,
    heartbeat_interval_s: float = 0.01,
    request_timeout_s: float = 1.0,
    shutdown_timeout_s: float = 1.0,
    clock: Callable[[], float] | None = None,
    sleep: Callable[[float], Awaitable[None]] | None = None,
) -> tuple[DashservingStateReporter, FakeTransport]:
    transport = transport or FakeTransport()
    config = SubscriberConfig(
        subscriber_heartbeat_interval_s=heartbeat_interval_s,
        subscriber_state_request_timeout_s=request_timeout_s,
        subscriber_shutdown_report_timeout_s=shutdown_timeout_s,
    )
    client = httpx.AsyncClient(transport=transport)
    reporter = DashservingStateReporter(
        config, http_client=client, clock=clock, sleep=sleep
    )
    return reporter, transport


async def wait_until(predicate: Callable[[], bool], timeout: float = 2.0) -> None:
    async def _poll() -> None:
        while not predicate():
            await asyncio.sleep(0.001)

    await asyncio.wait_for(_poll(), timeout=timeout)


# --- strict increment -------------------------------------------------------


async def test_active_reports_strictly_increment_from_one() -> None:
    reporter, transport = make_reporter()

    await reporter.report_active()
    await reporter.report_active()
    await reporter.report_active()

    assert transport.seqs() == [1, 2, 3]
    assert transport.states() == ["active", "active", "active"]
    assert reporter.next_seq_id == 4
    assert reporter.phase is SubscriberPhase.ACTIVE


async def test_first_active_logs_phase_transition_once(mocker) -> None:
    reporter, _ = make_reporter()
    info = mocker.patch("subscriber.health.state_reporter.logger.info")

    await reporter.report_active()
    await reporter.report_active()

    transition_calls = [
        c for c in info.call_args_list if c.args and "transition" in c.args[0]
    ]
    assert len(transition_calls) == 1
    assert transition_calls[0].kwargs["tags"]["from"] == "starting"
    assert transition_calls[0].kwargs["tags"]["to"] == "active"


# --- timeout gaps -----------------------------------------------------------


async def test_timed_out_sequence_is_never_reused() -> None:
    transport = FakeTransport()
    transport.queue(httpx.ConnectTimeout("boom"))
    reporter, transport = make_reporter(transport)

    with pytest.raises(httpx.ConnectTimeout):
        await reporter.report_active()

    # seq 1 was consumed by the timeout; the retry must use seq 2, not reuse 1.
    assert reporter.next_seq_id == 2
    await reporter.report_active()
    assert transport.seqs() == [1, 2]
    assert reporter.phase is SubscriberPhase.ACTIVE


async def test_non_2xx_response_is_a_failed_report() -> None:
    transport = FakeTransport()
    transport.queue(httpx.Response(500, json={}))
    reporter, transport = make_reporter(transport)

    with pytest.raises(httpx.HTTPStatusError):
        await reporter.report_active()

    assert reporter.next_seq_id == 2


async def test_malformed_ack_is_a_failed_report() -> None:
    transport = FakeTransport()
    transport.queue(httpx.Response(200, json={"accepted": "yes"}))
    reporter, _ = make_reporter(transport)

    with pytest.raises(StateReportError):
        await reporter.report_active()

    # bool last_seq_id must also be rejected (bool is an int subclass).
    transport.queue(httpx.Response(200, json={"accepted": True, "last_seq_id": True}))
    with pytest.raises(StateReportError):
        await reporter.report_active()


# --- duplicate / stale acknowledgement --------------------------------------


async def test_stale_ack_with_higher_server_seq_is_invariant_warning(mocker) -> None:
    transport = FakeTransport()
    transport.queue(httpx.Response(200, json={"accepted": False, "last_seq_id": 99}))
    reporter, _ = make_reporter(transport)
    warning = mocker.patch("subscriber.health.state_reporter.logger.warning")

    with pytest.raises(StateReportError):
        await reporter.report_active()

    assert warning.call_count == 1
    tags = warning.call_args.kwargs["tags"]
    assert tags["seq_id"] == 1
    assert tags["last_seq_id"] == 99
    assert "invariant" in warning.call_args.args[0]


async def test_duplicate_ack_is_rejected_not_success(mocker) -> None:
    transport = FakeTransport()
    transport.queue(httpx.Response(200, json={"accepted": False, "last_seq_id": 1}))
    reporter, _ = make_reporter(transport)
    warning = mocker.patch("subscriber.health.state_reporter.logger.warning")

    with pytest.raises(StateReportError):
        await reporter.report_active()

    assert warning.call_count == 1
    assert "duplicate/stale" in warning.call_args.args[0]


# --- no active after graceful inactive --------------------------------------


async def test_no_active_after_graceful_inactive() -> None:
    reporter, transport = make_reporter()

    await reporter.report_active()
    await reporter.report_shutdown_inactive("graceful shutdown")

    assert reporter.phase is SubscriberPhase.INACTIVE
    with pytest.raises(StateReportError):
        await reporter.report_active()

    # Only the active and the terminal inactive were sent; no active after.
    assert transport.states() == ["active", "inactive"]
    assert transport.seqs() == [1, 2]


async def test_late_active_ack_does_not_roll_back_inactive() -> None:
    entered = asyncio.Event()
    release = asyncio.Event()

    async def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content.decode())
        if body["state"] == "active":
            entered.set()
            await release.wait()
        return httpx.Response(
            200, json={"accepted": True, "last_seq_id": body["seq_id"]}
        )

    transport = FakeTransport(handler=handler)
    reporter, _ = make_reporter(transport)

    active_task = asyncio.create_task(reporter.report_active())
    await wait_until(entered.is_set)

    # A newer local transition lands while the active POST is still in flight.
    await reporter.report_shutdown_inactive("shutdown")
    assert reporter.phase is SubscriberPhase.INACTIVE

    release.set()
    await active_task

    # The late accepted active ack must not roll back the terminal inactive phase.
    assert reporter.phase is SubscriberPhase.INACTIVE


async def test_shutdown_inactive_uses_higher_seq_than_last_heartbeat() -> None:
    reporter, transport = make_reporter(heartbeat_interval_s=0.005)

    await reporter.report_active()
    reporter.start_heartbeat()
    await wait_until(lambda: len(transport.requests) >= 3)
    await reporter.report_shutdown_inactive("shutdown")

    last = transport.requests[-1]
    assert last["state"] == "inactive"
    heartbeat_seqs = [
        int(r["seq_id"]) for r in transport.requests if r["state"] == "active"
    ]
    assert int(last["seq_id"]) > max(heartbeat_seqs)


async def test_shutdown_inactive_is_best_effort_on_transport_failure(mocker) -> None:
    transport = FakeTransport()
    reporter, transport = make_reporter(transport)
    await reporter.report_active()
    transport.queue(httpx.ConnectTimeout("down"))
    warning = mocker.patch("subscriber.health.state_reporter.logger.warning")

    # Must not raise even though the POST fails.
    await reporter.report_shutdown_inactive("shutdown")

    assert reporter.phase is SubscriberPhase.INACTIVE
    assert warning.call_count == 1


async def test_shutdown_inactive_is_idempotent() -> None:
    reporter, transport = make_reporter()
    await reporter.report_active()

    await reporter.report_shutdown_inactive("first")
    await reporter.report_shutdown_inactive("second")

    assert transport.states() == ["active", "inactive"]


# --- report_failed ----------------------------------------------------------


async def test_report_failed_allowed_only_while_starting() -> None:
    reporter, transport = make_reporter()

    await reporter.report_failed("bad metadata")

    assert reporter.phase is SubscriberPhase.FAILED
    assert transport.requests == [{"state": "failed", "seq_id": 1}]
    with pytest.raises(StateReportError):
        await reporter.report_failed("again")


async def test_report_failed_rejected_after_active() -> None:
    reporter, transport = make_reporter()
    await reporter.report_active()

    with pytest.raises(StateReportError):
        await reporter.report_failed("too late")

    assert transport.states() == ["active"]
    assert reporter.phase is SubscriberPhase.ACTIVE


# --- heartbeat task ownership -----------------------------------------------


async def test_start_heartbeat_rejects_second_start() -> None:
    reporter, _ = make_reporter(heartbeat_interval_s=10.0)

    reporter.start_heartbeat()
    try:
        with pytest.raises(StateReportError):
            reporter.start_heartbeat()
    finally:
        await reporter.stop_heartbeat()


async def test_stop_heartbeat_cancels_and_clears_task() -> None:
    reporter, _ = make_reporter(heartbeat_interval_s=10.0)
    reporter.start_heartbeat()
    task = reporter._heartbeat_task
    assert task is not None

    await reporter.stop_heartbeat()

    assert reporter._heartbeat_task is None
    assert task.cancelled()
    # Idempotent.
    await reporter.stop_heartbeat()


async def test_heartbeat_sends_new_seq_active_each_interval() -> None:
    reporter, transport = make_reporter(heartbeat_interval_s=0.005)
    await reporter.report_active()  # seq 1

    reporter.start_heartbeat()
    try:
        await wait_until(lambda: len(transport.requests) >= 4)
    finally:
        await reporter.stop_heartbeat()

    assert all(r["state"] == "active" for r in transport.requests)
    seqs = transport.seqs()
    assert seqs[0] == 1
    # Strictly increasing by one across startup active + heartbeats.
    assert seqs == list(range(1, len(seqs) + 1))


async def test_heartbeat_transient_failure_logged_then_continues(mocker) -> None:
    transport = FakeTransport()
    transport.queue(httpx.ConnectTimeout("blip"))  # first heartbeat fails
    reporter, transport = make_reporter(transport, heartbeat_interval_s=0.005)
    warning = mocker.patch("subscriber.health.state_reporter.logger.warning")
    info = mocker.patch("subscriber.health.state_reporter.logger.info")

    reporter.start_heartbeat()
    try:
        # Wait past the failed heartbeat and a subsequent successful one.
        await wait_until(lambda: len(transport.requests) >= 2)
        await wait_until(
            lambda: any(
                c.args and "recovered" in c.args[0] for c in info.call_args_list
            )
        )
    finally:
        await reporter.stop_heartbeat()

    first_failures = [
        c for c in warning.call_args_list if c.args and "failed" in c.args[0]
    ]
    assert len(first_failures) == 1
    recovered = [c for c in info.call_args_list if c.args and "recovered" in c.args[0]]
    assert len(recovered) == 1


async def test_heartbeat_failure_summaries_are_suppressed_by_clock(mocker) -> None:
    clock = FakeClock()
    reporter, _ = make_reporter(clock=clock)
    warning = mocker.patch("subscriber.health.state_reporter.logger.warning")

    reporter._note_heartbeat_failure(error="E", message="m", seq_id=1)
    reporter._note_heartbeat_failure(error="E", message="m", seq_id=2)
    # Only the first failure warns; the next is suppressed within the window.
    assert warning.call_count == 1

    clock.advance(31.0)
    reporter._note_heartbeat_failure(error="E", message="m", seq_id=3)
    assert warning.call_count == 2
    summary_tags = warning.call_args.kwargs["tags"]
    assert summary_tags["consecutive_failures"] == 3


# --- cancellation during in-flight heartbeat --------------------------------


async def test_cancellation_during_in_flight_heartbeat() -> None:
    entered = asyncio.Event()
    release = asyncio.Event()

    async def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content.decode())
        entered.set()
        await release.wait()
        return httpx.Response(
            200, json={"accepted": True, "last_seq_id": body["seq_id"]}
        )

    transport = FakeTransport(handler=handler)
    reporter, _ = make_reporter(transport, heartbeat_interval_s=0.001)

    reporter.start_heartbeat()
    task = reporter._heartbeat_task
    assert task is not None
    await wait_until(entered.is_set)  # heartbeat POST is now in flight

    # stop_heartbeat must cancel the in-flight request and await it, not hang.
    await asyncio.wait_for(reporter.stop_heartbeat(), timeout=2.0)
    assert task.cancelled()
    assert reporter._heartbeat_task is None
    release.set()


async def test_heartbeat_loop_reraises_cancellation() -> None:
    reporter, _ = make_reporter(heartbeat_interval_s=10.0)
    reporter.start_heartbeat()
    task = reporter._heartbeat_task
    assert task is not None

    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task


# --- unsupported independent-restart fail-safe ------------------------------


async def test_independent_restart_is_loud_not_silent_success(mocker) -> None:
    # A surviving DashServing already holds a higher seq from a previous reporter
    # instance. A freshly restarted reporter begins at seq 1 and is rejected as
    # stale. Sequence-only ordering cannot represent this topology; the reporter
    # must fail loudly rather than report a false active. See module TODO.
    transport = FakeTransport()
    transport.queue(httpx.Response(200, json={"accepted": False, "last_seq_id": 500}))
    reporter, _ = make_reporter(transport)
    warning = mocker.patch("subscriber.health.state_reporter.logger.warning")

    with pytest.raises(StateReportError):
        await reporter.report_active()

    assert reporter.phase is SubscriberPhase.STARTING
    assert warning.call_count == 1
    assert "invariant" in warning.call_args.args[0]


# --- close ------------------------------------------------------------------


async def test_close_is_idempotent() -> None:
    reporter, _ = make_reporter(heartbeat_interval_s=10.0)
    reporter.start_heartbeat()

    await reporter.close()
    await reporter.close()

    assert reporter._heartbeat_task is None
