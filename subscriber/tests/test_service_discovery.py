from __future__ import annotations

import asyncio
import threading

import pytest

from subscriber.kvcm import manager_client, service_discovery
from subscriber.kvcm.manager_client import HttpKvCacheManagerClient
from subscriber.kvcm.service_discovery import (
    ServiceDiscovery,
    ServiceEndpoint,
    SpectrumServiceDiscovery,
    create_service_discovery,
)


class FakeResponse:
    def __init__(self, payload: dict[str, object], *, status_code: int = 200) -> None:
        self._payload = payload
        self.status_code = status_code

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict[str, object]:
        return self._payload


class SequencedFakeHttpClient:
    def __init__(self, **_: object) -> None:
        self.headers: dict[str, str] = {}
        self.calls: list[tuple[str, float]] = []
        self._lock = threading.Lock()
        self._payload = {"instances": [{"ip": "10.0.0.9", "port": 7001}]}
        self.requested = threading.Event()
        self.closed = False

    def set_payload(self, payload: dict[str, object]) -> None:
        with self._lock:
            self._payload = payload

    async def get(self, url: str, timeout: float) -> FakeResponse:
        with self._lock:
            self.calls.append((url, timeout))
            payload = self._payload
        self.requested.set()
        return FakeResponse(payload)

    async def aclose(self) -> None:
        self.closed = True


class RecordingAsyncHttpClient:
    def __init__(self, payload: dict[str, object] | None = None) -> None:
        self.headers: dict[str, str] = {}
        self.post_calls: list[tuple[str, dict[str, object], float]] = []
        self.closed = False
        self.payload = payload or {"header": {"status": {"code": "OK"}}}

    async def post(
        self,
        url: str,
        *,
        json: dict[str, object],
        headers: dict[str, str],
        timeout: float,
    ) -> FakeResponse:
        self.post_calls.append((url, json, timeout))
        return FakeResponse(self.payload)

    async def aclose(self) -> None:
        self.closed = True


class EmptyServiceDiscovery(ServiceDiscovery):
    def __init__(self) -> None:
        self.endpoint: ServiceEndpoint | None = None
        self.closed = False

    def get_type(self) -> str:
        return "Empty"

    def get_all_endpoints(self) -> list[ServiceEndpoint]:
        return [self.endpoint] if self.endpoint is not None else []

    def get_one_endpoint(self) -> ServiceEndpoint | None:
        return self.endpoint

    async def refresh(self) -> bool:
        return True

    async def close(self) -> None:
        self.closed = True


async def test_spectrum_service_discovery_fetches_initial_endpoints() -> None:
    http_client = SequencedFakeHttpClient()

    discovery = SpectrumServiceDiscovery(
        "vs-a",
        refresh_timeout=0.5,
        http_client=http_client,
    )
    await discovery.start()

    assert http_client.calls == [
        (
            "http://127.0.0.1:8880/api/v1/discovery/virtual-services/vs-a/instances",
            0.5,
        )
    ]
    assert [
        (endpoint.ip, endpoint.port) for endpoint in discovery.get_all_endpoints()
    ] == [("10.0.0.9", 7001)]
    await discovery.close()
    assert http_client.closed


async def test_spectrum_service_discovery_polls_changed_endpoints() -> None:
    http_client = SequencedFakeHttpClient()
    discovery = SpectrumServiceDiscovery(
        "vs-a",
        cache_ttl=0.01,
        http_client=http_client,
    )
    await discovery.start()

    http_client.requested.clear()
    http_client.set_payload({"instances": [{"ip": "10.0.0.10", "port": 7002}]})
    assert await asyncio.to_thread(http_client.requested.wait, 1.0)
    assert [
        (endpoint.ip, endpoint.port) for endpoint in discovery.get_all_endpoints()
    ] == [("10.0.0.10", 7002)]
    await discovery.close()


async def test_spectrum_service_discovery_keeps_cached_endpoints_on_poll_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    http_client = SequencedFakeHttpClient()
    discovery = SpectrumServiceDiscovery("vs-a", cache_ttl=60, http_client=http_client)
    await discovery.start()

    async def fail_get(url: str, timeout: float) -> FakeResponse:
        raise RuntimeError("gateway unavailable")

    monkeypatch.setattr(http_client, "get", fail_get)
    assert not await discovery.refresh()
    assert [
        (endpoint.ip, endpoint.port) for endpoint in discovery.get_all_endpoints()
    ] == [("10.0.0.9", 7001)]
    await discovery.close()


async def test_manager_client_recovers_when_initial_spectrum_discovery_is_empty(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    discovery = EmptyServiceDiscovery()
    monkeypatch.setattr(
        manager_client,
        "create_service_discovery",
        lambda _: discovery,
    )

    client = HttpKvCacheManagerClient(
        "spectrum://vs-a",
        auto_discover_leader=False,
    )

    await client.start()

    assert client.is_ready() is False
    assert discovery.closed is False

    discovery.endpoint = ServiceEndpoint(ip="10.0.0.9", port=7001, host="10.0.0.9:7001")

    assert client.is_ready() is True
    assert client.base_url == "http://10.0.0.9:7001"

    await client.close()
    assert discovery.closed is True


async def test_manager_client_reports_with_reused_async_http_client() -> None:
    http_client = RecordingAsyncHttpClient()
    client = HttpKvCacheManagerClient(
        "http://manager.test:8080",
        auto_discover_leader=False,
        request_timeout_seconds=0.25,
        http_client=http_client,
    )
    await client.start()

    payload = await client.report_event({"trace_id": "t1"})
    await client.close()

    assert payload == {"header": {"status": {"code": "OK"}}}
    assert http_client.post_calls == [
        (
            "http://manager.test:8080/api/reportEvent",
            {"trace_id": "t1"},
            0.25,
        )
    ]
    assert http_client.closed


async def test_manager_report_event_raises_for_non_ok_status_by_default() -> None:
    http_client = RecordingAsyncHttpClient(
        {
            "header": {
                "status": {
                    "code": "INTERNAL_ERROR",
                    "message": "ReportEvent partially failed",
                }
            },
            "item_results": ["OK", "INTERNAL_ERROR"],
        }
    )
    client = HttpKvCacheManagerClient(
        "http://manager.test:8080",
        auto_discover_leader=False,
        http_client=http_client,
    )
    await client.start()

    with pytest.raises(
        RuntimeError,
        match=(
            "KVCM /api/reportEvent failed: INTERNAL_ERROR ReportEvent partially failed"
        ),
    ) as exc_info:
        await client.report_event({"trace_id": "t1"})

    await client.close()

    assert "item_results=['OK', 'INTERNAL_ERROR']" in str(exc_info.value)


async def test_manager_report_event_can_return_non_ok_for_explicit_inspection() -> None:
    response = {
        "header": {"status": {"code": "INTERNAL_ERROR", "message": "failed"}},
        "item_results": ["INTERNAL_ERROR"],
    }
    http_client = RecordingAsyncHttpClient(response)
    client = HttpKvCacheManagerClient(
        "http://manager.test:8080",
        auto_discover_leader=False,
        http_client=http_client,
    )
    await client.start()

    payload = await client.report_event({"trace_id": "t1"}, check_response=False)
    await client.close()

    assert payload == response


async def test_manager_initial_leader_discovery_failure_is_nonfatal() -> None:
    class FailingHttpClient(RecordingAsyncHttpClient):
        async def post(
            self,
            url: str,
            *,
            json: dict[str, object],
            headers: dict[str, str],
            timeout: float,
        ) -> FakeResponse:
            raise RuntimeError("manager unavailable")

    http_client = FailingHttpClient()
    client = HttpKvCacheManagerClient(
        "http://manager.test:8080",
        discovery_refresh_interval_seconds=60,
        http_client=http_client,
    )

    await client.start()
    await client.close()

    assert http_client.closed


async def test_create_service_discovery_defaults_to_one_second_spectrum_polling(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        service_discovery.httpx,
        "AsyncClient",
        SequencedFakeHttpClient,
    )

    discovery = create_service_discovery("spectrum://vs-a")
    assert discovery is not None
    await discovery.start()

    assert isinstance(discovery, SpectrumServiceDiscovery)
    assert discovery.cache_ttl == 1
    await discovery.close()
