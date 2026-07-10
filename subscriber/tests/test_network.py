from __future__ import annotations

import pytest

from subscriber.utils import network


def test_resolve_host_ip_port_prefers_configured_value() -> None:
    assert (
        network.resolve_host_ip_port("10.0.0.8:9000", "http://127.0.0.1:8123/health")
        == "10.0.0.8:9000"
    )


def test_resolve_host_ip_port_uses_health_url_port(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(network, "local_ip_address", lambda: "10.0.0.7")

    assert (
        network.resolve_host_ip_port("", "http://127.0.0.1:8123/health")
        == "10.0.0.7:8123"
    )


def test_resolve_host_ip_port_formats_ipv6_and_defaults_http_port(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(network, "local_ip_address", lambda: "2001:db8::1")

    assert network.resolve_host_ip_port("", "http://[::1]/health") == "[2001:db8::1]:80"
