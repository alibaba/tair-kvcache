from __future__ import annotations

import pytest

from subscriber.utils import network


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


async def test_resolve_host_ip_port_uses_given_port(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(network, "local_ip_address", lambda: "10.0.0.7")

    assert await network.resolve_host_ip_port(8080) == "10.0.0.7:8080"
    assert await network.resolve_host_ip_port(9001) == "10.0.0.7:9001"


async def test_resolve_host_ip_port_formats_local_ipv6(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(network, "local_ip_address", lambda: "2001:db8::1")

    assert await network.resolve_host_ip_port(8080) == "[2001:db8::1]:8080"
