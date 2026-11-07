"""End-to-end health-linkage tests for the readiness/liveness + subscriber state.

Spawns the real ``dashservingd`` binary (no workers, so worker readiness is
vacuously satisfied and the probes are driven purely by subscriber state) and
verifies the state matrix from
``docs/specs/2026-07-22_subscriber-health-dashserving-integration.md``:

| # | subscriber state              | /readiness | /liveness |
|---|-------------------------------|:----------:|:---------:|
| 1 | starting (never active)       |    503     |    200    |
| 2 | active + TTL valid            |    200     |    200    |
| 3 | inactive after ever active    |    503     |    503    |
| 4 | failed (from starting)        |    503     |    503    |
| 5 | TTL expired (heartbeat stop)  |    503     |    503    |
| 6 | recovery after TTL expiry     |    200     |    200    |
| 7 | stale active after inactive   |    503     |    503    |
| 8 | feature disabled              |  existing  |  existing |

Each scenario that needs a pristine state machine gets its own daemon
subprocess via a factory fixture. The suite is explicit opt-in: it skips
unless ``DSV_BINARY`` points at a built ``dashservingd`` binary, so the
default pytest run behaves the same on every machine.

These assertions encode the spec matrix verbatim so the suite doubles as a
cross-process conformance check.
"""

from __future__ import annotations

import asyncio
import os
import socket
import subprocess
import time
from collections.abc import Callable, Iterator
from pathlib import Path

import httpx
import pytest

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


def _distinct_ports(count: int) -> list[int]:
    """Return ``count`` mutually distinct free ports.

    The daemon binds a secondary listener at ``DSV_PORT+1``, so the control
    port must avoid both the data port and its successor.
    """

    ports: list[int] = []
    while len(ports) < count:
        candidate = _free_port()
        reserved = set(ports)
        reserved.update(p + 1 for p in ports)
        if candidate not in reserved:
            ports.append(candidate)
    return ports


class DashServingInstance:
    """A running ``dashservingd`` subprocess bound to a control port."""

    def __init__(self, proc: subprocess.Popen[bytes], base: str) -> None:
        self.proc = proc
        self.base = base

    @property
    def readiness_url(self) -> str:
        return f"{self.base}/readiness"

    @property
    def liveness_url(self) -> str:
        return f"{self.base}/liveness"

    @property
    def state_url(self) -> str:
        return f"{self.base}/subscriber_state"

    @property
    def status_url(self) -> str:
        return f"{self.base}/status"

    def stop(self) -> None:
        if self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.proc.kill()
                self.proc.wait(timeout=5)


def _spawn(extra_env: dict[str, str]) -> DashServingInstance:
    """Start a fresh daemon and wait until its control port serves /liveness."""

    port, control_port = _distinct_ports(2)
    env = os.environ.copy()
    env.update(
        {
            "DSV_PORT": str(port),
            "DSV_CONTROL_PORT": str(control_port),
            "DSV_LOG_FORMAT": "text",
        }
    )
    env.update(extra_env)
    proc = subprocess.Popen(  # noqa: S603 - fixed-path trusted local binary
        [_BINARY],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    base = f"http://127.0.0.1:{control_port}"
    deadline = time.monotonic() + 10.0
    while True:
        if proc.poll() is not None:
            output = proc.stdout.read().decode(errors="replace") if proc.stdout else ""
            proc.wait()
            raise RuntimeError(
                f"dashservingd exited early (code={proc.returncode}): {output[-2000:]}"
            )
        try:
            resp = httpx.get(f"{base}/liveness", timeout=0.5)
        except httpx.HTTPError:
            resp = None
        if resp is not None and resp.status_code in (200, 503):
            # The control port is up. /liveness may legitimately be 503 in some
            # states, so accept either as "server is listening".
            break
        if time.monotonic() > deadline:
            proc.terminate()
            raise RuntimeError(
                f"dashservingd control port {control_port} did not respond within 10s"
            )
        time.sleep(0.05)
    return DashServingInstance(proc, base)


@pytest.fixture
def spawn_dashserving() -> Iterator[Callable[..., DashServingInstance]]:
    """Factory fixture: spawn a fresh daemon per call, tear all down at end."""

    instances: list[DashServingInstance] = []

    def _factory(
        *,
        enabled: bool = True,
        ttl_s: int = 20,
    ) -> DashServingInstance:
        extra = {
            "DS_LLM_LAUNCH_KV_EVENT_SUBSCRIBER": "1" if enabled else "0",
            "DSV_SUBSCRIBER_TTL_S": str(ttl_s),
        }
        instance = _spawn(extra)
        instances.append(instance)
        return instance

    yield _factory

    for instance in instances:
        instance.stop()


async def _sleep(seconds: float) -> None:
    await asyncio.sleep(seconds)


async def _wait_status(
    client: httpx.AsyncClient,
    url: str,
    expected: int,
    timeout: float = 5.0,
) -> int:
    """Poll ``url`` until it returns ``expected``; return the last status."""

    deadline = time.monotonic() + timeout
    last = 0
    while time.monotonic() < deadline:
        last = (await client.get(url)).status_code
        if last == expected:
            return last
        await _sleep(0.05)
    return last


async def _post_state(
    client: httpx.AsyncClient,
    instance: DashServingInstance,
    state: str,
    seq_id: int,
) -> httpx.Response:
    return await client.post(
        instance.state_url,
        json={"state": state, "seq_id": seq_id},
    )


async def test_scenario_1_starting_never_active(
    spawn_dashserving: Callable[..., DashServingInstance],
) -> None:
    """Spec #1: starting (never active) → readiness 503, liveness 200."""

    instance = spawn_dashserving()
    async with httpx.AsyncClient() as client:
        readiness = (await client.get(instance.readiness_url)).status_code
        liveness = (await client.get(instance.liveness_url)).status_code

    assert liveness == 200, f"starting liveness: expected 200, got {liveness}"
    assert readiness == 503, f"starting readiness: expected 503, got {readiness}"


async def test_scenario_2_active_ttl_valid(
    spawn_dashserving: Callable[..., DashServingInstance],
) -> None:
    """Spec #2: active + TTL valid → readiness 200, liveness 200."""

    instance = spawn_dashserving()
    async with httpx.AsyncClient() as client:
        ack = await _post_state(client, instance, "active", 1)
        assert ack.status_code == 200
        assert ack.json() == {"accepted": True, "last_seq_id": 1}

        readiness = await _wait_status(client, instance.readiness_url, 200)
        liveness = (await client.get(instance.liveness_url)).status_code

    assert readiness == 200, f"active readiness: expected 200, got {readiness}"
    assert liveness == 200, f"active liveness: expected 200, got {liveness}"


async def test_scenario_3_inactive_after_active(
    spawn_dashserving: Callable[..., DashServingInstance],
) -> None:
    """Spec #3: inactive after ever active → readiness 503, liveness 503."""

    instance = spawn_dashserving()
    async with httpx.AsyncClient() as client:
        active = await _post_state(client, instance, "active", 1)
        assert active.json() == {"accepted": True, "last_seq_id": 1}
        await _wait_status(client, instance.readiness_url, 200)

        inactive = await _post_state(client, instance, "inactive", 2)
        assert inactive.json() == {"accepted": True, "last_seq_id": 2}

        readiness = await _wait_status(client, instance.readiness_url, 503)
        liveness = await _wait_status(client, instance.liveness_url, 503)

    assert readiness == 503, f"inactive readiness: expected 503, got {readiness}"
    assert liveness == 503, f"inactive liveness: expected 503, got {liveness}"


async def test_scenario_4_failed_from_starting(
    spawn_dashserving: Callable[..., DashServingInstance],
) -> None:
    """Spec #4: failed from starting → readiness 503, liveness 503."""

    instance = spawn_dashserving()
    async with httpx.AsyncClient() as client:
        failed = await _post_state(client, instance, "failed", 1)
        assert failed.status_code == 200
        assert failed.json() == {"accepted": True, "last_seq_id": 1}

        readiness = (await client.get(instance.readiness_url)).status_code
        liveness = (await client.get(instance.liveness_url)).status_code

    assert readiness == 503, f"failed readiness: expected 503, got {readiness}"
    assert liveness == 503, f"failed liveness: expected 503, got {liveness}"


async def test_scenario_5_ttl_expiry(
    spawn_dashserving: Callable[..., DashServingInstance],
) -> None:
    """Spec #5: TTL expiry (heartbeat stopped) → readiness 503, liveness 503."""

    instance = spawn_dashserving(ttl_s=2)
    async with httpx.AsyncClient() as client:
        active = await _post_state(client, instance, "active", 1)
        assert active.json() == {"accepted": True, "last_seq_id": 1}
        readiness_active = await _wait_status(client, instance.readiness_url, 200)
        assert readiness_active == 200

        # Stop heartbeating and wait past the 2s TTL; expiry is lazy, so probe
        # /status to force evaluation, then read the probes.
        await _sleep(2.5)
        status = (await client.get(instance.status_url)).json()
        assert status["subscriber"]["phase"] == "inactive", status["subscriber"]
        assert status["subscriber"]["expired_total"] >= 1, status["subscriber"]

        readiness = await _wait_status(client, instance.readiness_url, 503)
        liveness = await _wait_status(client, instance.liveness_url, 503)

    assert readiness == 503, f"ttl-expired readiness: expected 503, got {readiness}"
    assert liveness == 503, f"ttl-expired liveness: expected 503, got {liveness}"


async def test_scenario_6_recovery_after_ttl_expiry(
    spawn_dashserving: Callable[..., DashServingInstance],
) -> None:
    """Spec #6: a higher-seq active after TTL expiry recovers readiness 200."""

    instance = spawn_dashserving(ttl_s=2)
    async with httpx.AsyncClient() as client:
        active = await _post_state(client, instance, "active", 1)
        assert active.json() == {"accepted": True, "last_seq_id": 1}
        await _wait_status(client, instance.readiness_url, 200)

        await _sleep(2.5)
        expired = await _wait_status(client, instance.readiness_url, 503)
        assert expired == 503

        recover = await _post_state(client, instance, "active", 2)
        assert recover.status_code == 200
        assert recover.json() == {"accepted": True, "last_seq_id": 2}

        readiness = await _wait_status(client, instance.readiness_url, 200)
        liveness = (await client.get(instance.liveness_url)).status_code

    assert readiness == 200, f"recovery readiness: expected 200, got {readiness}"
    assert liveness == 200, f"recovery liveness: expected 200, got {liveness}"


async def test_scenario_7_stale_active_cannot_revive(
    spawn_dashserving: Callable[..., DashServingInstance],
) -> None:
    """Spec #7: a stale active (lower seq) after inactive cannot revive readiness."""

    instance = spawn_dashserving()
    async with httpx.AsyncClient() as client:
        active = await _post_state(client, instance, "active", 1)
        assert active.json() == {"accepted": True, "last_seq_id": 1}
        await _wait_status(client, instance.readiness_url, 200)

        inactive = await _post_state(client, instance, "inactive", 2)
        assert inactive.json() == {"accepted": True, "last_seq_id": 2}
        await _wait_status(client, instance.readiness_url, 503)

        stale = await _post_state(client, instance, "active", 1)
        assert stale.status_code == 200
        assert stale.json() == {"accepted": False, "last_seq_id": 2}

        readiness = (await client.get(instance.readiness_url)).status_code

    assert readiness == 503, f"stale-active readiness: expected 503, got {readiness}"


async def test_scenario_8_feature_disabled(
    spawn_dashserving: Callable[..., DashServingInstance],
) -> None:
    """Spec #8: feature disabled → probes keep existing semantics (200, no workers).

    With no workers configured and the subscriber integration off, readiness and
    liveness are governed purely by the daemon's own running state and stay 200
    regardless of any subscriber_state reports.
    """

    instance = spawn_dashserving(enabled=False)
    async with httpx.AsyncClient() as client:
        readiness_before = (await client.get(instance.readiness_url)).status_code
        liveness_before = (await client.get(instance.liveness_url)).status_code

        # The endpoint still accepts reports, but they must not affect probes.
        inactive = await _post_state(client, instance, "inactive", 1)
        assert inactive.status_code == 200

        readiness_after = (await client.get(instance.readiness_url)).status_code
        liveness_after = (await client.get(instance.liveness_url)).status_code

        status = (await client.get(instance.status_url)).json()

    assert readiness_before == 200, f"disabled readiness: {readiness_before}"
    assert liveness_before == 200, f"disabled liveness: {liveness_before}"
    assert readiness_after == 200, (
        f"disabled readiness changed by report: {readiness_after}"
    )
    assert liveness_after == 200, (
        f"disabled liveness changed by report: {liveness_after}"
    )
    assert status["subscriber"]["enabled"] is False, status["subscriber"]
