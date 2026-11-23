from __future__ import annotations

import asyncio

import httpx
import pytest
from pytest_mock import MockerFixture

from subscriber.kvcm import manager_client
from subscriber.kvcm.errors import KvcmUnavailableError
from subscriber.kvcm.manager_client import HttpKvCacheManagerClient


async def test_http_manager_client_has_awaitable_readiness() -> None:
    client = HttpKvCacheManagerClient("http://127.0.0.1:8080")

    try:
        assert await client.is_ready() is True
    finally:
        await client.close()


class _FakeResponse:
    def __init__(self, payload: dict[str, object], *, status_code: int = 200) -> None:
        self._payload = payload
        self.status_code = status_code

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict[str, object]:
        return self._payload


class _RaisingHttpClient:
    """Raises a configured exception on reportEvent; OK (no leader) elsewhere."""

    def __init__(self, exc: Exception) -> None:
        self.headers: dict[str, str] = {}
        self._exc = exc
        self.closed = False
        self.report_calls = 0

    async def post(
        self,
        url: str,
        *,
        json: dict[str, object],
        headers: dict[str, str],
        timeout: float,
    ) -> _FakeResponse:
        if url.endswith("/api/reportEvent"):
            self.report_calls += 1
            raise self._exc
        return _FakeResponse({"header": {"status": {"code": "OK"}}})

    async def aclose(self) -> None:
        self.closed = True


async def _started_client(
    exc: Exception, *, auto_discover_leader: bool = True
) -> tuple[HttpKvCacheManagerClient, _RaisingHttpClient]:
    http_client = _RaisingHttpClient(exc)
    client = HttpKvCacheManagerClient(
        "http://manager.test:8080",
        auto_discover_leader=auto_discover_leader,
        discovery_refresh_interval_seconds=60,
        request_timeout_seconds=0.25,
        http_client=http_client,
    )
    await client.start()
    return client, http_client


async def test_request_timeout_reraises_and_notifies_leader_refresh(
    mocker: MockerFixture,
) -> None:
    client, http_client = await _started_client(httpx.ReadTimeout("slow"))
    notify = mocker.patch.object(client, "_notify_leader_refresh")

    with pytest.raises(httpx.ReadTimeout):
        await client.report_event({"trace_id": "t1"})

    assert http_client.report_calls == 1
    notify.assert_called_once_with()
    await client.close()


async def test_request_timeout_logs_distinct_timeout_message(
    mocker: MockerFixture,
) -> None:
    client, _ = await _started_client(httpx.ReadTimeout("slow"))
    warning = mocker.patch.object(manager_client.logger, "warning")

    with pytest.raises(httpx.ReadTimeout):
        await client.report_event({"trace_id": "t1"})

    warning.assert_any_call(
        "kvcm request timed out",
        step="kvcm_request",
        tags={
            "url": "http://manager.test:8080/api/reportEvent",
            "timeout_seconds": 0.25,
            "error": "ReadTimeout",
        },
    )
    await client.close()


async def test_connect_error_reraises_and_notifies_leader_refresh(
    mocker: MockerFixture,
) -> None:
    client, _ = await _started_client(httpx.ConnectError("refused"))
    notify = mocker.patch.object(client, "_notify_leader_refresh")

    with pytest.raises(httpx.ConnectError):
        await client.report_event({"trace_id": "t1"})

    notify.assert_called_once_with()
    await client.close()


async def test_generic_request_error_reraises_without_leader_refresh(
    mocker: MockerFixture,
) -> None:
    exc = httpx.ReadError("connection dropped")
    client, _ = await _started_client(exc)
    notify = mocker.patch.object(client, "_notify_leader_refresh")
    warning = mocker.patch.object(manager_client.logger, "warning")

    with pytest.raises(httpx.ReadError):
        await client.report_event({"trace_id": "t1"})

    notify.assert_not_called()
    warning.assert_any_call(
        "kvcm request failed",
        step="kvcm_request",
        tags={
            "url": "http://manager.test:8080/api/reportEvent",
            "error": "ReadError",
            "message": "connection dropped",
        },
    )
    await client.close()


async def test_notify_leader_refresh_sets_event_only_when_discovery_enabled() -> None:
    enabled, _ = await _started_client(httpx.ReadTimeout("x"))
    enabled._notify_leader_refresh()
    assert enabled._refresh_event.is_set()
    await enabled.close()

    disabled, _ = await _started_client(
        httpx.ReadTimeout("x"), auto_discover_leader=False
    )
    disabled._notify_leader_refresh()
    assert not disabled._refresh_event.is_set()
    await disabled.close()


class _ScriptedHttpClient:
    """Returns a fixed cluster-info payload and a queue of reportEvent payloads."""

    def __init__(
        self,
        cluster_payload: dict[str, object],
        report_payloads: list[dict[str, object]],
    ) -> None:
        self.headers: dict[str, str] = {}
        self.closed = False
        self._cluster_payload = cluster_payload
        self._report_payloads = list(report_payloads)
        self.report_calls = 0
        self.cluster_calls = 0

    async def post(
        self,
        url: str,
        *,
        json: dict[str, object],
        headers: dict[str, str],
        timeout: float,
    ) -> _FakeResponse:
        if url.endswith("/api/getClusterInfo"):
            self.cluster_calls += 1
            return _FakeResponse(self._cluster_payload)
        if url.endswith("/api/reportEvent"):
            self.report_calls += 1
            return _FakeResponse(self._report_payloads.pop(0))
        return _FakeResponse({"header": {"status": {"code": "OK"}}})

    async def aclose(self) -> None:
        self.closed = True


_CLUSTER_WITH_LEADER: dict[str, object] = {
    "header": {"status": {"code": "OK"}},
    "leader_endpoint": {"host": "10.0.0.5", "meta_http_port": 9999},
}


async def test_leader_discovery_switches_base_url_on_start() -> None:
    http_client = _ScriptedHttpClient(_CLUSTER_WITH_LEADER, [])
    client = HttpKvCacheManagerClient(
        "http://manager.test:8080",
        auto_discover_leader=True,
        discovery_refresh_interval_seconds=60,
        http_client=http_client,
    )

    await client.start()

    assert client.base_url == "http://10.0.0.5:9999"
    assert http_client.cluster_calls == 1
    await client.close()


async def test_server_not_leader_triggers_rediscovery_then_succeeds() -> None:
    ok: dict[str, object] = {"header": {"status": {"code": "OK"}}}
    report_payloads: list[dict[str, object]] = [
        {"header": {"status": {"code": "SERVER_NOT_LEADER"}}},
        ok,
    ]
    http_client = _ScriptedHttpClient(_CLUSTER_WITH_LEADER, report_payloads)
    client = HttpKvCacheManagerClient(
        "http://manager.test:8080",
        auto_discover_leader=True,
        leader_retry_count=1,
        leader_retry_base_interval_seconds=0.0,
        discovery_refresh_interval_seconds=60,
        http_client=http_client,
    )
    await client.start()

    payload = await client.report_event({"trace_id": "t1"})

    assert payload == {**ok, "_subscriber_retry_count": 1}
    assert http_client.report_calls == 2
    await client.close()


async def test_server_not_leader_exhausted_raises_unavailable_not_rejected() -> None:
    """When leader discovery fails and retries exhaust, the failure is a
    retryable leader failover, not a permanent report rejection: it must not
    surface as the rejected-report RuntimeError that KvcmClient classifies
    as KvcmReportRejectedError (a permanent drop)."""

    cluster_without_leader: dict[str, object] = {"header": {"status": {"code": "OK"}}}
    not_leader: dict[str, object] = {
        "header": {"status": {"code": "SERVER_NOT_LEADER"}}
    }
    http_client = _ScriptedHttpClient(cluster_without_leader, [not_leader, not_leader])
    client = HttpKvCacheManagerClient(
        "http://manager.test:8080",
        auto_discover_leader=True,
        leader_retry_count=1,
        leader_retry_base_interval_seconds=0.0,
        discovery_refresh_interval_seconds=60,
        http_client=http_client,
    )
    await client.start()

    with pytest.raises(KvcmUnavailableError) as error:
        await client.report_event({"trace_id": "t1"})

    assert error.value.status_code == "SERVER_NOT_LEADER"
    assert error.value.reason == "leader_retry_exhausted"
    assert error.value.retry_count == 1
    await client.close()


async def test_server_not_leader_retries_failed_discovery_until_exhausted() -> None:
    cluster_without_leader: dict[str, object] = {"header": {"status": {"code": "OK"}}}
    not_leader: dict[str, object] = {
        "header": {"status": {"code": "SERVER_NOT_LEADER"}}
    }
    http_client = _ScriptedHttpClient(
        cluster_without_leader, [not_leader, not_leader, not_leader]
    )
    client = HttpKvCacheManagerClient(
        "http://manager.test:8080",
        auto_discover_leader=True,
        leader_retry_count=2,
        leader_retry_base_interval_seconds=0.0,
        discovery_refresh_interval_seconds=60,
        http_client=http_client,
    )
    await client.start()

    with pytest.raises(KvcmUnavailableError):
        await client.report_event({"trace_id": "t1"})

    assert http_client.report_calls == 3
    await client.close()


class _StubEndpoint:
    def __init__(self, host: str) -> None:
        self.host = host


class _StubDiscovery:
    def __init__(self, endpoint: _StubEndpoint | None) -> None:
        self._endpoint = endpoint

    def get_one_endpoint(self) -> _StubEndpoint | None:
        return self._endpoint

    def get_type(self) -> str:
        return "stub"

    async def close(self) -> None:
        return None


async def test_concurrent_is_ready_does_not_skip_leader_discovery() -> None:
    """is_ready() resolving a service-discovery endpoint must not mutate
    base_url outside the leader lock: doing so makes a concurrent
    _discover_leader() see a changed snapshot and skip discovery, leaving
    requests routed to a non-leader endpoint."""

    http_client = _ScriptedHttpClient(_CLUSTER_WITH_LEADER, [])
    client = HttpKvCacheManagerClient(
        "static://seed",
        auto_discover_leader=True,
        discovery_refresh_interval_seconds=60,
        http_client=http_client,
    )
    # Simulate a not-yet-resolved discovery scheme with an endpoint available.
    client._service_discovery = _StubDiscovery(_StubEndpoint("10.0.0.7:8080"))  # type: ignore[assignment]

    async with client._leader_lock:
        discover_task = asyncio.create_task(client._discover_leader())
        await asyncio.sleep(0)  # discover snapshots base_url, blocks on the lock
        ready_task = asyncio.create_task(client.is_ready())
        await asyncio.sleep(0)

    assert await discover_task is True
    assert await ready_task is True
    assert http_client.cluster_calls == 1
    assert client.base_url == "http://10.0.0.5:9999"
    await client.close()
