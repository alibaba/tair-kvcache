"""Service discovery for Tair KVCM manager endpoints.

Supports URL-based configuration compatible with tair-kvcache C++/Python conventions:
    - ``static://ip:port[,ip:port]...``
    - ``spectrum://<virtual_service_id>[?cache_time=<sec>&timeout=<ms>&retry_time=<n>]``
"""

from __future__ import annotations

import asyncio
import random
from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Literal

import httpx

from subscriber import logger
from subscriber.utils.spectrum import (
    fetch_spectrum_endpoints,
)

_DISCOVERY_STEP = "kvcm_discovery"


@dataclass
class ServiceEndpoint:
    """A resolved service endpoint."""

    ip: str
    port: int
    host: str  # f"{ip}:{port}"
    weight: int = 100
    healthy: bool = True


class ServiceDiscovery(ABC):
    """Service discovery abstract base class."""

    @abstractmethod
    def get_all_endpoints(self) -> list[ServiceEndpoint]:
        """Return all available endpoints; empty list on failure."""

    @abstractmethod
    def get_one_endpoint(self) -> ServiceEndpoint | None:
        """Return a single endpoint via load-balancing; None if unavailable."""

    @abstractmethod
    async def refresh(self) -> bool:
        """Force refresh; return whether successful."""

    @abstractmethod
    def get_type(self) -> str:
        """Return implementation type name (e.g. "Static", "Spectrum")."""

    async def start(self) -> None:
        """Initialize discovery resources; no-op by default."""
        return None

    async def close(self) -> None:
        """Release resources; no-op by default."""
        return None

    def __enter__(self) -> ServiceDiscovery:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: Any,
    ) -> Literal[False]:
        return False


# ---------------------------------------------------------------------------
# Static
# ---------------------------------------------------------------------------


def _parse_host_port_list(host_list: str) -> list[ServiceEndpoint]:
    if not host_list:
        raise ValueError("host_list is empty")
    endpoints: list[ServiceEndpoint] = []
    for token in host_list.split(","):
        token = token.strip()
        if not token:
            continue
        if ":" not in token:
            raise ValueError(f"static endpoint missing host:port format: {token!r}")
        host, _, port_str = token.partition(":")
        if not host or not port_str:
            raise ValueError(f"static endpoint missing host or port: {token!r}")
        if not port_str.isdigit():
            raise ValueError(f"static endpoint port not numeric: {token!r}")
        port = int(port_str)
        if port <= 0 or port > 65535:
            raise ValueError(f"static endpoint port out of range: {token!r}")
        endpoints.append(ServiceEndpoint(ip=host, port=port, host=f"{host}:{port}"))
    if not endpoints:
        raise ValueError(f"static endpoint list is empty after parsing: {host_list!r}")
    return endpoints


class StaticServiceDiscovery(ServiceDiscovery):
    """Fixed ip:port list service discovery with round-robin selection."""

    def __init__(
        self,
        host_list: str | None = None,
        *,
        endpoints: Sequence[ServiceEndpoint] | None = None,
    ):
        if endpoints is not None:
            parsed: list[ServiceEndpoint] = list(endpoints)
        elif host_list is not None:
            parsed = _parse_host_port_list(host_list)
        else:
            raise ValueError("must provide either host_list or endpoints")
        if not parsed:
            raise ValueError("StaticServiceDiscovery init with empty endpoints")
        self._endpoints: list[ServiceEndpoint] = parsed
        self._rr_index = 0
        self._rr_lock = asyncio.Lock()

    def get_type(self) -> str:
        return "Static"

    def get_all_endpoints(self) -> list[ServiceEndpoint]:
        return list(self._endpoints)

    def get_one_endpoint(self) -> ServiceEndpoint | None:
        if not self._endpoints:
            return None
        ep = self._endpoints[self._rr_index % len(self._endpoints)]
        self._rr_index = (self._rr_index + 1) % len(self._endpoints)
        return ep

    async def refresh(self) -> bool:
        return bool(self._endpoints)


# ---------------------------------------------------------------------------
# Spectrum
# ---------------------------------------------------------------------------


class SpectrumServiceDiscovery(ServiceDiscovery):
    """Spectrum gateway service discovery with a background refresh loop."""

    def __init__(
        self,
        virtual_service_id: str,
        *,
        port_override: int | None = None,
        cache_ttl: float = 1,
        refresh_timeout: float = 5,
        auto_refresh: bool = True,
        retry_count: int = 0,
        http_client: Any | None = None,
    ) -> None:
        if not virtual_service_id:
            raise ValueError("virtual_service_id must not be empty")
        self.virtual_service_id = virtual_service_id
        self.port_override = port_override
        self.cache_ttl = cache_ttl
        self.refresh_timeout = refresh_timeout
        self.auto_refresh = auto_refresh
        self.retry_count = max(0, retry_count)
        if self.cache_ttl <= 0:
            raise ValueError("spectrum poll interval must be positive")

        self._cache: list[ServiceEndpoint] = []
        self._cache_lock = asyncio.Lock()
        self._refresh_lock = asyncio.Lock()
        self._closed = False

        self._http_client = http_client or httpx.AsyncClient(
            headers={
                "Accept": "application/json",
                "Content-Type": "application/json",
            }
        )
        self._refresh_task: asyncio.Task[None] | None = None

    async def start(self) -> None:
        await self.refresh()
        if self.auto_refresh and self._refresh_task is None:
            self._refresh_task = asyncio.create_task(
                self._refresh_loop(),
                name="spectrum-service-discovery-refresh",
            )

    def get_type(self) -> str:
        return "Spectrum"

    def get_all_endpoints(self) -> list[ServiceEndpoint]:
        return self._cache.copy()

    def get_one_endpoint(self) -> ServiceEndpoint | None:
        endpoints = self.get_all_endpoints()
        if not endpoints:
            return None
        return random.choice(endpoints)

    async def refresh(self) -> bool:
        async with self._refresh_lock:
            total_attempts = self.retry_count + 1
            last_err: Exception | None = None
            for _ in range(total_attempts):
                try:
                    endpoints = await self._fetch_from_spectrum()
                    changed = endpoints != self._cache
                    if changed:
                        self._cache = endpoints
                    if changed:
                        logger.info(
                            "Spectrum service discovery updated endpoints",
                            step=_DISCOVERY_STEP,
                            tags={
                                "virtual_service_id": self.virtual_service_id,
                                "endpoint_count": len(endpoints),
                                "endpoints": [endpoint.host for endpoint in endpoints],
                            },
                        )
                    return True
                except Exception as e:
                    last_err = e
            logger.error(
                "failed to refresh Spectrum service discovery; "
                "keeping cached endpoints",
                step=_DISCOVERY_STEP,
                tags={
                    "virtual_service_id": self.virtual_service_id,
                    "attempt_count": total_attempts,
                    "error": last_err.__class__.__name__ if last_err else "Unknown",
                    "message": str(last_err) if last_err else "",
                    "cached_endpoint_count": len(self._cache),
                },
            )
            return False

    async def close(self) -> None:
        self._closed = True
        if self._refresh_task is not None:
            self._refresh_task.cancel()
            try:
                await self._refresh_task
            except asyncio.CancelledError:
                pass
            self._refresh_task = None
        await self._http_client.aclose()

    async def _refresh_loop(self) -> None:
        while not self._closed:
            await asyncio.sleep(self.cache_ttl)
            if self._closed:
                break
            await self.refresh()

    async def _fetch_from_spectrum(self) -> list[ServiceEndpoint]:
        spectrum_endpoints = await fetch_spectrum_endpoints(
            self.virtual_service_id,
            port_override=self.port_override,
            refresh_timeout=self.refresh_timeout,
            http_client=self._http_client,
        )
        return [
            ServiceEndpoint(
                ip=endpoint.ip,
                port=endpoint.port,
                host=f"{endpoint.ip}:{endpoint.port}",
                weight=endpoint.weight,
            )
            for endpoint in spectrum_endpoints
        ]


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

_SCHEME_STATIC = "static"
_SCHEME_SPECTRUM = "spectrum"


def _parse_discovery_url(
    url: str,
) -> tuple[str, str, dict[str, str]] | None:
    """Parse ``<scheme>://<body>[?k=v(&k=v)*]``."""
    if not url:
        return None
    sep = url.find("://")
    if sep <= 0:
        logger.error(
            "invalid service discovery url",
            step=_DISCOVERY_STEP,
            tags={"reason": "missing_scheme", "url": url},
        )
        return None
    scheme = url[:sep]
    rest = url[sep + 3 :]
    if not rest:
        logger.error(
            "invalid service discovery url",
            step=_DISCOVERY_STEP,
            tags={"reason": "empty_body", "url": url},
        )
        return None
    body, sep_char, query = rest.partition("?")
    params: dict[str, str] = {}
    if sep_char:
        for kv in query.split("&"):
            if not kv:
                continue
            k, eq, v = kv.partition("=")
            params[k] = v if eq else ""
    if not body:
        logger.error(
            "invalid service discovery url",
            step=_DISCOVERY_STEP,
            tags={"reason": "empty_body", "url": url},
        )
        return None
    return scheme, body, params


def _get_int_param(params: dict[str, str], key: str, default: int) -> int:
    raw = params.get(key)
    if raw is None or raw == "":
        return default
    try:
        return int(raw)
    except (TypeError, ValueError):
        return default


def create_service_discovery(url: str) -> ServiceDiscovery | None:
    """Create a ServiceDiscovery instance from a URL string.

    Supported schemes: ``static://``, ``spectrum://``.
    Returns None for empty/invalid URLs.
    """
    if not url:
        return None
    parsed = _parse_discovery_url(url)
    if parsed is None:
        return None
    scheme, body, params = parsed

    if scheme == _SCHEME_STATIC:
        try:
            return StaticServiceDiscovery(body)
        except Exception as e:
            logger.error(
                "failed to create static service discovery",
                step=_DISCOVERY_STEP,
                tags={
                    "url": url,
                    "error": e.__class__.__name__,
                    "message": str(e),
                },
            )
            return None

    if scheme == _SCHEME_SPECTRUM:
        # spectrum://vsid or spectrum://vsid:port
        port_override: int | None = None
        vsid = body
        colon_pos = body.rfind(":")
        if colon_pos > 0:
            maybe_port = body[colon_pos + 1 :]
            if maybe_port.isdigit():
                port_override = int(maybe_port)
                vsid = body[:colon_pos]
        cache_ttl = _get_int_param(params, "cache_time", 1)
        timeout_ms = _get_int_param(params, "timeout", 0)
        refresh_timeout = max(1, timeout_ms // 1000) if timeout_ms > 0 else 5
        retry_count = _get_int_param(params, "retry_time", 0)
        try:
            return SpectrumServiceDiscovery(
                vsid,
                port_override=port_override,
                cache_ttl=cache_ttl,
                refresh_timeout=refresh_timeout,
                retry_count=retry_count,
            )
        except Exception as e:
            logger.error(
                "failed to create Spectrum service discovery",
                step=_DISCOVERY_STEP,
                tags={
                    "url": url,
                    "error": e.__class__.__name__,
                    "message": str(e),
                },
            )
            return None

    logger.error(
        "unsupported service discovery scheme",
        step=_DISCOVERY_STEP,
        tags={"scheme": scheme, "url": url},
    )
    return None
