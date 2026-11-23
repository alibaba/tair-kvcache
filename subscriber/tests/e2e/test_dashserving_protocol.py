"""End-to-end protocol test against the real ``dashservingd`` binary.

Spawns the actual DashServing daemon (no workers, subscriber integration
enabled) as a subprocess and drives it with the real
``DashservingStateReporter`` over HTTP, verifying:

- ``POST /subscriber_state`` accepts monotonically increasing reports and
  rejects stale/duplicate ones with ``accepted=false`` (200).
- ``GET /readiness`` is 503 during starting (never active), flips to 200
  on active, drops to 503 once a previously-active subscriber reports
  inactive, and recovers to 200 on a higher-seq active.
- ``GET /liveness`` is 200 during starting (startup grace), stays 200
  while active, and drops to 503 when a previously-active subscriber goes
  inactive (spec: 同生同死).
- ``report_failed`` from the starting phase is accepted and terminal; a
  higher-seq active report is rejected.

The suite is explicit opt-in: it skips unless ``DSV_BINARY`` points at a
built ``dashservingd`` binary, so the default pytest run behaves the same
on every machine.
"""

from __future__ import annotations

import asyncio
import os
import socket
import subprocess
import time
from collections.abc import Iterator
from pathlib import Path

import httpx
import pytest

from subscriber.config import SubscriberConfig
from subscriber.health.state_reporter import (
    DashservingStateReporter,
    SubscriberPhase,
)

# e2e is explicit opt-in: set DSV_BINARY to a built dashservingd binary.
# No machine-specific default, so the default pytest run is deterministic
# (these tests always skip) on every machine.
_BINARY = Path(os.environ.get("DSV_BINARY", ""))

pytestmark = [
    pytest.mark.e2e,
    pytest.mark.skipif(
        not _BINARY.is_file(),
        reason="e2e opt-in: set DSV_BINARY to a dashservingd binary "
        "(build with `cargo build` in the dashserving repo)",
    ),
]


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


@pytest.fixture
def dashserving() -> Iterator[dict[str, object]]:
    """Start dashservingd with subscriber integration enabled, no workers.

    ``DSV_PORT`` and ``DSV_CONTROL_PORT`` are deliberately far apart because
    the daemon also binds a secondary listener at ``DSV_PORT+1``.
    """

    port = _free_port()
    # The daemon also binds a secondary listener at DSV_PORT+1, so the control
    # port must avoid both `port` and `port + 1`.
    control_port = _free_port()
    while control_port in (port, port + 1):
        control_port = _free_port()
    env = os.environ.copy()
    env.update(
        {
            "DSV_PORT": str(port),
            "DSV_CONTROL_PORT": str(control_port),
            "DS_LLM_LAUNCH_KV_EVENT_SUBSCRIBER": "1",
            "DSV_SUBSCRIBER_TTL_S": "20",
            "DSV_LOG_FORMAT": "text",
        }
    )
    proc = subprocess.Popen(  # noqa: S603 - fixed-path trusted local binary
        [_BINARY],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    base = f"http://127.0.0.1:{control_port}"
    try:
        deadline = time.monotonic() + 10.0
        while True:
            if proc.poll() is not None:
                output = (
                    proc.stdout.read().decode(errors="replace") if proc.stdout else ""
                )
                pytest.fail(
                    f"dashservingd exited early (code={proc.returncode}): "
                    f"{output[-2000:]}"
                )
            try:
                resp = httpx.get(f"{base}/liveness", timeout=0.5)
            except httpx.HTTPError:
                resp = None
            if resp is not None and resp.status_code == 200:
                break
            if time.monotonic() > deadline:
                proc.terminate()
                pytest.fail(
                    f"dashservingd control port {control_port} did not "
                    "become ready within 10s"
                )
            time.sleep(0.05)
        yield {
            "proc": proc,
            "base": base,
            "control_port": control_port,
        }
    finally:
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=5)


def make_reporter(base: str) -> DashservingStateReporter:
    # Extract port from base URL and set DSV_CONTROL_PORT so the
    # subscriber_state_url property derives the correct endpoint.
    port = base.rsplit(":", 1)[1]
    os.environ["DSV_CONTROL_PORT"] = port
    config = SubscriberConfig(
        subscriber_heartbeat_interval_s=3.0,
        subscriber_state_request_timeout_s=2.0,
        subscriber_shutdown_report_timeout_s=2.0,
    )
    return DashservingStateReporter(config)


async def _wait_status(
    client: httpx.AsyncClient, url: str, expected: int, timeout: float = 5.0
) -> None:
    deadline = time.monotonic() + timeout
    last = 0
    while time.monotonic() < deadline:
        last = (await client.get(url)).status_code
        if last == expected:
            return
        await _sleep(0.05)
    raise AssertionError(
        f"{url}: expected HTTP {expected} within {timeout}s, last={last}"
    )


async def _sleep(seconds: float) -> None:
    await asyncio.sleep(seconds)


async def test_full_protocol_against_real_dashserving(
    dashserving: dict[str, object],
) -> None:
    """Drive the complete subscriber-state protocol against the real binary."""

    base = str(dashserving["base"])
    reporter = make_reporter(base)
    try:
        async with httpx.AsyncClient() as client:
            readiness = f"{base}/readiness"
            liveness = f"{base}/liveness"

            # Subscriber starting (never active) blocks readiness (spec: 503)
            # but liveness stays 200 (startup grace).
            await _wait_status(client, readiness, 503)
            assert (await client.get(liveness)).status_code == 200

            # (a) First active report → accepted, seq starts at 1.
            await reporter.report_active()
            assert reporter.phase is SubscriberPhase.ACTIVE
            assert reporter.next_seq_id == 2
            await _wait_status(client, readiness, 200)

            # (b) Second active with higher seq → accepted.
            await reporter.report_active()
            assert reporter.next_seq_id == 3

            # (c) Terminal inactive → accepted (best-effort, never raises).
            await reporter.report_shutdown_inactive("e2e-test")
            phase: SubscriberPhase = reporter.phase
            assert phase is SubscriberPhase.INACTIVE

            # Readiness must drop to 503 once a previously-active subscriber
            # goes inactive; liveness also drops to 503 (spec: 同生同死).
            await _wait_status(client, readiness, 503)
            await _wait_status(client, liveness, 503)

            # (d) Stale active (lower seq) is rejected with accepted=false,
            # and does NOT flip readiness back to 200.
            stale = await client.post(
                f"{base}/subscriber_state",
                json={"state": "active", "seq_id": 1},
            )
            assert stale.status_code == 200
            assert stale.json() == {"accepted": False, "last_seq_id": 3}
            assert (await client.get(readiness)).status_code == 503

            # A higher-seq active recovers: accepted=true and readiness 200.
            recover = await client.post(
                f"{base}/subscriber_state",
                json={"state": "active", "seq_id": 4},
            )
            assert recover.status_code == 200
            assert recover.json() == {"accepted": True, "last_seq_id": 4}
            await _wait_status(client, readiness, 200)

            # Duplicate (same seq) is rejected as idempotent no-op.
            dup = await client.post(
                f"{base}/subscriber_state",
                json={"state": "active", "seq_id": 4},
            )
            assert dup.status_code == 200
            assert dup.json() == {"accepted": False, "last_seq_id": 4}

            # /status exposes the subscriber snapshot.
            status = (await client.get(f"{base}/status")).json()
            sub = status["subscriber"]
            assert sub["enabled"] is True
            assert sub["phase"] == "active"
            assert sub["last_seq_id"] == 4
    finally:
        await reporter.close()


async def test_failed_from_starting_is_terminal(
    dashserving: dict[str, object],
) -> None:
    """``failed`` is legal while starting and rejects later transitions."""

    base = str(dashserving["base"])
    reporter = make_reporter(base)
    try:
        await reporter.report_failed("e2e startup failure")
        assert reporter.phase is SubscriberPhase.FAILED

        async with httpx.AsyncClient() as client:
            await _wait_status(client, f"{base}/readiness", 503)
            await _wait_status(client, f"{base}/liveness", 503)

            conflict = await client.post(
                f"{base}/subscriber_state",
                json={"state": "active", "seq_id": 2},
            )
            assert conflict.status_code == 409
    finally:
        await reporter.close()
