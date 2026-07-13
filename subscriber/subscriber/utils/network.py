from __future__ import annotations

import ipaddress
import os
import socket

from subscriber.utils.spectrum import fetch_spectrum_endpoints

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
    """Resolve the local engine address from its Spectrum virtual service."""

    virtual_service_id = os.environ.get("ENGINE_VSERVICE_ID", "")
    if not virtual_service_id:
        raise ValueError("Please specify ENGINE_VSERVICE_ID")
    address = local_ip_address()
    endpoints = await fetch_spectrum_endpoints(virtual_service_id)
    endpoint = next((item for item in endpoints if item.ip == address), None)
    if endpoint is None:
        raise ValueError(
            "Spectrum did not return an endpoint for local IP "
            f"{address!r} in ENGINE_VSERVICE_ID={virtual_service_id!r}"
        )
    if ":" in address:
        address = f"[{address}]"
    return f"{address}:{endpoint.port}"
