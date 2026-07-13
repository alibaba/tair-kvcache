from __future__ import annotations

import ipaddress
import socket

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


async def resolve_host_ip_port() -> str:
    """Resolve the local PAI-EAS worker address."""

    address = local_ip_address()
    if ":" in address:
        address = f"[{address}]"
    # PAI-EAS worker port is always 8080
    return f"{address}:8080"
