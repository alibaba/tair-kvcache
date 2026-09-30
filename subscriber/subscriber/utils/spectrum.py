"""Spectrum virtual-service endpoint requests."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import httpx

from subscriber import logger

_DISCOVERY_STEP = "kvcm_discovery"

SPECTRUM_GATEWAY_BASE_URL = "http://127.0.0.1:8880"
SPECTRUM_INSTANCES_PATH_TEMPLATE = "/api/v1/discovery/virtual-services/{vsid}/instances"


@dataclass(frozen=True)
class SpectrumEndpoint:
    """One valid endpoint returned by Spectrum."""

    ip: str
    port: int
    weight: int = 100


async def fetch_spectrum_endpoints(
    virtual_service_id: str,
    *,
    port_override: int | None = None,
    refresh_timeout: float = 5,
    http_client: Any | None = None,
) -> list[SpectrumEndpoint]:
    """Fetch and validate endpoints for one Spectrum virtual service.

    When ``http_client`` is omitted, this function creates and closes an
    ``httpx.AsyncClient`` for the request. Callers that need connection reuse
    must create and pass their own client; this function does not close it.
    """

    if not virtual_service_id:
        raise ValueError("virtual_service_id must not be empty")
    client: Any
    if http_client is None:
        owns_http_client = True
        client = httpx.AsyncClient(
            headers={
                "Accept": "application/json",
                "Content-Type": "application/json",
            }
        )
    else:
        owns_http_client = False
        client = http_client
    try:
        path = SPECTRUM_INSTANCES_PATH_TEMPLATE.format(vsid=virtual_service_id)
        response = await client.get(
            SPECTRUM_GATEWAY_BASE_URL + path, timeout=refresh_timeout
        )
        response.raise_for_status()
        data = response.json()
        items = data.get("instances", [])
        if not isinstance(items, list):
            logger.warning(
                "Spectrum response 'instances' is not a list",
                step=_DISCOVERY_STEP,
                tags={"virtual_service_id": virtual_service_id},
            )
            return []
        endpoints: list[SpectrumEndpoint] = []
        for item in items:
            if not isinstance(item, dict):
                continue
            ip = item.get("ip")
            port = port_override if port_override is not None else item.get("port")
            if not isinstance(ip, str) or not ip:
                continue
            if not isinstance(port, (int, str)) or isinstance(port, bool):
                continue
            try:
                parsed_port = int(port)
            except (TypeError, ValueError):
                continue
            if parsed_port <= 0 or parsed_port > 65535:
                continue
            endpoints.append(
                SpectrumEndpoint(
                    ip=ip,
                    port=parsed_port,
                    weight=item.get("weight", 100),
                )
            )
        if items and not endpoints:
            logger.warning(
                "Spectrum returned instances but none were valid endpoints",
                step=_DISCOVERY_STEP,
                tags={
                    "virtual_service_id": virtual_service_id,
                    "instance_count": len(items),
                },
            )
        return sorted(endpoints, key=lambda endpoint: (endpoint.ip, endpoint.port))
    finally:
        if owns_http_client:
            await client.aclose()
