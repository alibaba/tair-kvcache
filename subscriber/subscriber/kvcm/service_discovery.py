"""Service discovery for Tair KVCM manager endpoints.

Supports URL-based configuration compatible with tair-kvcache C++/Python conventions:
    - ``static://ip:port[,ip:port]...``
    - ``spectrum://<virtual_service_id>[?cache_time=<sec>&timeout=<ms>&retry_time=<n>]``
"""

from __future__ import annotations

import logging
import random
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import requests

logger = logging.getLogger(__name__)


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
    def get_all_endpoints(self) -> List[ServiceEndpoint]:
        """Return all available endpoints; empty list on failure."""

    @abstractmethod
    def get_one_endpoint(self) -> Optional[ServiceEndpoint]:
        """Return a single endpoint via load-balancing; None if unavailable."""

    @abstractmethod
    def refresh(self) -> bool:
        """Force refresh; return whether successful."""

    @abstractmethod
    def get_type(self) -> str:
        """Return implementation type name (e.g. "Static", "Spectrum")."""

    def close(self) -> None:
        """Release resources; no-op by default."""

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False


# ---------------------------------------------------------------------------
# Static
# ---------------------------------------------------------------------------


def _parse_host_port_list(host_list: str) -> List[ServiceEndpoint]:
    if not host_list:
        raise ValueError("host_list is empty")
    endpoints: List[ServiceEndpoint] = []
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
        host_list: Optional[str] = None,
        *,
        endpoints: Optional[Sequence[ServiceEndpoint]] = None,
    ):
        if endpoints is not None:
            parsed: List[ServiceEndpoint] = list(endpoints)
        elif host_list is not None:
            parsed = _parse_host_port_list(host_list)
        else:
            raise ValueError("must provide either host_list or endpoints")
        if not parsed:
            raise ValueError("StaticServiceDiscovery init with empty endpoints")
        self._endpoints: List[ServiceEndpoint] = parsed
        self._rr_index = 0
        self._rr_lock = threading.Lock()

    def get_type(self) -> str:
        return "Static"

    def get_all_endpoints(self) -> List[ServiceEndpoint]:
        return list(self._endpoints)

    def get_one_endpoint(self) -> Optional[ServiceEndpoint]:
        if not self._endpoints:
            return None
        with self._rr_lock:
            ep = self._endpoints[self._rr_index % len(self._endpoints)]
            self._rr_index = (self._rr_index + 1) % len(self._endpoints)
        return ep

    def refresh(self) -> bool:
        return bool(self._endpoints)


# ---------------------------------------------------------------------------
# Spectrum
# ---------------------------------------------------------------------------

SPECTRUM_GATEWAY_BASE_URL = "http://127.0.0.1:8880"
SPECTRUM_INSTANCES_PATH_TEMPLATE = (
    "/api/v1/discovery/virtual-services/{vsid}/instances"
)


class SpectrumServiceDiscovery(ServiceDiscovery):
    """Spectrum gateway based service discovery with TTL cache."""

    def __init__(
        self,
        virtual_service_id: str,
        *,
        port_override: Optional[int] = None,
        cache_ttl: int = 30,
        refresh_timeout: int = 5,
        auto_refresh: bool = True,
        retry_count: int = 0,
    ):
        if not virtual_service_id:
            raise ValueError("virtual_service_id must not be empty")
        self.virtual_service_id = virtual_service_id
        self.port_override = port_override
        self.cache_ttl = cache_ttl
        self.refresh_timeout = refresh_timeout
        self.auto_refresh = auto_refresh
        self.retry_count = max(0, retry_count)

        self._cache: List[ServiceEndpoint] = []
        self._cache_time: float = 0.0
        self._cache_lock = threading.Lock()

        self._session = requests.Session()
        self._session.headers.update(
            {"Accept": "application/json", "Content-Type": "application/json"}
        )
        self.refresh()

    def get_type(self) -> str:
        return "Spectrum"

    @property
    def service_url(self) -> str:
        path = SPECTRUM_INSTANCES_PATH_TEMPLATE.format(vsid=self.virtual_service_id)
        return SPECTRUM_GATEWAY_BASE_URL + path

    def get_all_endpoints(self) -> List[ServiceEndpoint]:
        with self._cache_lock:
            if not (self._is_cache_expired() and self.auto_refresh):
                return self._cache.copy()
        self.refresh()
        with self._cache_lock:
            return self._cache.copy()

    def get_one_endpoint(self) -> Optional[ServiceEndpoint]:
        endpoints = self.get_all_endpoints()
        if not endpoints:
            return None
        return random.choice(endpoints)

    def refresh(self) -> bool:
        total_attempts = self.retry_count + 1
        last_err: Optional[Exception] = None
        for _ in range(total_attempts):
            try:
                endpoints = self._fetch_from_spectrum()
                with self._cache_lock:
                    self._cache = endpoints
                    self._cache_time = time.time()
                if endpoints:
                    logger.debug(
                        "Spectrum service discovery refreshed: %d endpoints for vsid=%s",
                        len(endpoints),
                        self.virtual_service_id,
                    )
                else:
                    logger.warning(
                        "Spectrum service discovery returned no endpoints for vsid=%s",
                        self.virtual_service_id,
                    )
                return True
            except Exception as e:
                last_err = e
                continue
        logger.error(
            "Failed to refresh Spectrum service discovery for vsid=%s "
            "after %d attempts: %s",
            self.virtual_service_id,
            total_attempts,
            last_err,
        )
        return False

    def close(self) -> None:
        self._session.close()

    def _is_cache_expired(self) -> bool:
        return (time.time() - self._cache_time) > self.cache_ttl

    def _fetch_from_spectrum(self) -> List[ServiceEndpoint]:
        response = self._session.get(self.service_url, timeout=self.refresh_timeout)
        response.raise_for_status()
        data = response.json()
        items = data.get("instances", [])
        if not isinstance(items, list):
            logger.warning(
                "Spectrum response 'instances' is not a list for vsid=%s",
                self.virtual_service_id,
            )
            return []
        endpoints: List[ServiceEndpoint] = []
        for item in items:
            if not isinstance(item, dict):
                continue
            if "ip" not in item or "port" not in item:
                continue
            port = self.port_override if self.port_override is not None else item["port"]
            endpoints.append(
                ServiceEndpoint(
                    ip=item["ip"],
                    port=port,
                    host=f"{item['ip']}:{port}",
                    weight=item.get("weight", 100),
                    healthy=True,
                )
            )
        return endpoints


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

_SCHEME_STATIC = "static"
_SCHEME_SPECTRUM = "spectrum"


def _parse_discovery_url(
    url: str,
) -> Optional[tuple[str, str, Dict[str, str]]]:
    """Parse ``<scheme>://<body>[?k=v(&k=v)*]``."""
    if not url:
        return None
    sep = url.find("://")
    if sep <= 0:
        logger.error("invalid service discovery url, missing scheme: %r", url)
        return None
    scheme = url[:sep]
    rest = url[sep + 3:]
    if not rest:
        logger.error("invalid service discovery url, empty body: %r", url)
        return None
    body, sep_char, query = rest.partition("?")
    params: Dict[str, str] = {}
    if sep_char:
        for kv in query.split("&"):
            if not kv:
                continue
            k, eq, v = kv.partition("=")
            params[k] = v if eq else ""
    if not body:
        logger.error("invalid service discovery url, empty body: %r", url)
        return None
    return scheme, body, params


def _get_int_param(params: Dict[str, str], key: str, default: int) -> int:
    raw = params.get(key)
    if raw is None or raw == "":
        return default
    try:
        return int(raw)
    except (TypeError, ValueError):
        return default


def create_service_discovery(url: str) -> Optional[ServiceDiscovery]:
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
                "failed to create StaticServiceDiscovery for url=%r: %s", url, e
            )
            return None

    if scheme == _SCHEME_SPECTRUM:
        # spectrum://vsid or spectrum://vsid:port
        port_override: Optional[int] = None
        vsid = body
        colon_pos = body.rfind(":")
        if colon_pos > 0:
            maybe_port = body[colon_pos + 1:]
            if maybe_port.isdigit():
                port_override = int(maybe_port)
                vsid = body[:colon_pos]
        cache_ttl = _get_int_param(params, "cache_time", 30)
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
                "failed to create SpectrumServiceDiscovery for url=%r: %s", url, e
            )
            return None

    logger.error("unsupported service discovery scheme=%r, url=%r", scheme, url)
    return None
