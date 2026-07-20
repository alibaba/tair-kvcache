from __future__ import annotations

import importlib.util
import inspect

import pytest

from subscriber.utils import network
from subscriber.utils.spectrum import SpectrumEndpoint, fetch_spectrum_endpoints


def test_spectrum_fetch_is_available_without_kvcm_dependency() -> None:
    assert importlib.util.find_spec("subscriber.utils.spectrum") is not None
    assert "subscriber.kvcm" not in inspect.getsource(network)


class FakeResponse:
    def __init__(self, payload: dict[str, object]) -> None:
        self._payload = payload

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict[str, object]:
        return self._payload


class FakeHttpClient:
    def __init__(self, payload: dict[str, object]) -> None:
        self.payload = payload
        self.calls: list[tuple[str, float]] = []

    async def get(self, url: str, timeout: float) -> FakeResponse:
        self.calls.append((url, timeout))
        return FakeResponse(self.payload)


async def test_fetch_spectrum_endpoints_validates_response_once() -> None:
    http_client = FakeHttpClient(
        {
            "instances": [
                {"ip": "10.0.0.8", "port": "8000", "weight": 20},
                {"ip": "10.0.0.7", "port": 7000},
                {"ip": "10.0.0.9", "port": 0},
                {"ip": "10.0.0.10", "port": True},
            ]
        }
    )

    endpoints = await fetch_spectrum_endpoints(
        "engine-vs-a", http_client=http_client, refresh_timeout=0.5
    )

    assert http_client.calls == [
        (
            "http://127.0.0.1:8880/api/v1/discovery/virtual-services/"
            "engine-vs-a/instances",
            0.5,
        )
    ]
    assert endpoints == [
        SpectrumEndpoint(ip="10.0.0.7", port=7000),
        SpectrumEndpoint(
            ip="10.0.0.8",
            port=8000,
            weight=20,
        ),
    ]


class FakeSocket:
    def __init__(
        self,
        connect_error: OSError | None = None,
        local_ip: str = "10.0.0.5",
    ) -> None:
        self._connect_error = connect_error
        self._local_ip = local_ip

    def connect(self, address: tuple[str, int]) -> None:
        if self._connect_error is not None:
            raise self._connect_error

    def getsockname(self) -> tuple[str, int]:
        return (self._local_ip, 12345)

    def __enter__(self) -> FakeSocket:
        return self

    def __exit__(self, *args: object) -> None:
        pass


def test_local_ip_address_returns_probe_address(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        network.socket, "socket", lambda *a, **kw: FakeSocket(local_ip="10.0.0.5")
    )

    assert network.local_ip_address() == "10.0.0.5"


def test_local_ip_address_falls_back_to_getaddrinfo(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        network.socket,
        "socket",
        lambda *a, **kw: FakeSocket(connect_error=OSError("no route")),
    )
    monkeypatch.setattr(
        network.socket,
        "getaddrinfo",
        lambda host, port, type: [
            (None, None, None, None, ("192.168.1.10", 0)),
        ],
    )

    assert network.local_ip_address() == "192.168.1.10"


def test_local_ip_address_returns_loopback_when_all_fail(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        network.socket,
        "socket",
        lambda *a, **kw: FakeSocket(connect_error=OSError("no route")),
    )
    monkeypatch.setattr(
        network.socket,
        "getaddrinfo",
        lambda host, port, type: (_ for _ in ()).throw(OSError("no info")),
    )

    assert network.local_ip_address() == "127.0.0.1"


async def test_resolve_host_ip_port_uses_pai_eas_worker_port(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(network, "local_ip_address", lambda: "10.0.0.7")

    assert await network.resolve_host_ip_port() == "10.0.0.7:8080"


async def test_resolve_host_ip_port_formats_local_ipv6(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(network, "local_ip_address", lambda: "2001:db8::1")

    assert await network.resolve_host_ip_port() == "[2001:db8::1]:8080"
