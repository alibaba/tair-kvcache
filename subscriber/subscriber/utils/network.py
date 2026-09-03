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


async def resolve_host_ip_port(port: int) -> str:
    """Resolve the local worker identity address as ``ip:port``.

    ``port`` is the worker identity port seen by FlexLB/KVCM; it must match
    the engine endpoint port published in Spectrum (8080 on PAI-EAS).
    """

    address = local_ip_address()
    if ":" in address:
        address = f"[{address}]"
    return f"{address}:{port}"
