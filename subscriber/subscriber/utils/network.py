from __future__ import annotations

import ipaddress
import socket
from urllib.parse import urlsplit

_ROUTE_PROBE_ADDRESS = ("192.0.2.1", 80)


def local_ip_address() -> str:
    """Return the preferred local address, falling back safely when offline."""

    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
        try:
            sock.connect(_ROUTE_PROBE_ADDRESS)
            address = str(sock.getsockname()[0])
            if not ipaddress.ip_address(address).is_loopback:
                return address
        except OSError:
            pass

    try:
        addresses = socket.getaddrinfo(
            socket.gethostname(), None, type=socket.SOCK_DGRAM
        )
    except OSError:
        addresses = []
    for _, _, _, _, sockaddr in addresses:
        address = str(sockaddr[0])
        try:
            if not ipaddress.ip_address(address).is_loopback:
                return address
        except ValueError:
            continue
    return "127.0.0.1"


def resolve_host_ip_port(configured_value: str, engine_health_url: str) -> str:
    """Resolve the KVCM node identity from configuration or local networking."""

    if configured_value:
        return configured_value

    parsed = urlsplit(engine_health_url)
    port = parsed.port
    if port is None and parsed.scheme == "http":
        port = 80
    if port is None:
        port = 0

    address = local_ip_address()
    if ":" in address:
        address = f"[{address}]"
    return f"{address}:{port}"
