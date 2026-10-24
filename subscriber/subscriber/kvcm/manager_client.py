"""KVCM manager HTTP client with service discovery and leader discovery.

Extracted from ``store.py`` to keep manager-client concerns separate from
the store/transfer layer.
"""

from __future__ import annotations

import asyncio
import random
import time
from typing import Any

import httpx

from subscriber import logger
from subscriber.kvcm.base import AbstractKvCacheManagerClient
from subscriber.kvcm.errors import KvcmResponseRejectedError, KvcmUnavailableError
from subscriber.kvcm.service_discovery import (
    ServiceDiscovery,
    create_service_discovery,
)

_DISCOVERY_STEP = "kvcm_discovery"
_REQUEST_STEP = "kvcm_request"


class HttpKvCacheManagerClient(AbstractKvCacheManagerClient):
    """HTTP manager client with optional service-discovery and leader-discovery.

    Compatible with the ``KvCacheManagerClient`` API surface used by
    ``TairKVCMClient`` (register_instance, get_cache_location, …).
    """

    def __init__(
        self,
        base_url: str,
        *,
        instance_id: str = "",
        auto_discover_leader: bool = True,
        leader_retry_count: int = 1,
        leader_retry_base_interval_seconds: float = 0.005,
        discovery_refresh_interval_seconds: int = 30,
        min_discover_interval_seconds: float = 1.0,
        request_timeout_seconds: float = 5.0,
        http_client: Any | None = None,
    ):
        """Initialize the HTTP manager client.

        Args:
            base_url: Manager service address. Supported formats:
                - ``http://host:port`` or ``https://host:port`` — used directly
                - ``static://ip:port[,ip:port]...`` — resolved via static
                  service discovery (round-robin)
                - ``spectrum://<vsid>[:port]`` with optional
                  ``cache_time`` / ``timeout`` / ``retry_time`` query params
                  — resolved via Spectrum gateway
        """
        super().__init__()

        self.headers = {
            "Accept": "application/json",
            "Content-Type": "application/json",
        }
        self._http_client = http_client or httpx.AsyncClient(headers=self.headers)
        self._request_timeout_seconds = float(request_timeout_seconds)
        if self._request_timeout_seconds <= 0:
            raise ValueError("request_timeout_seconds must be positive")

        # --- service discovery ---
        # http(s):// → use directly; other schemes → try service discovery
        self._configured_base_url = base_url
        self._service_discovery: ServiceDiscovery | None = None

        self.base_url = base_url.rstrip("/")
        # Immutable seed address for leader discovery requests.
        self._discovery_url = self.base_url

        # --- leader discovery ---
        self._instance_id = instance_id
        self._auto_discover_leader = auto_discover_leader
        self._leader_retry_count = leader_retry_count
        self._leader_retry_base_interval = leader_retry_base_interval_seconds
        self._discovery_refresh_interval = discovery_refresh_interval_seconds
        self._min_discover_interval = min_discover_interval_seconds

        self._leader_lock = asyncio.Lock()
        self._refresh_event = asyncio.Event()
        self._closed = False
        self._last_discover_time: float = 0.0
        self._refresh_task: asyncio.Task[None] | None = None
        self._started = False

    async def start(self) -> None:
        if self._started:
            return
        base_url = self._configured_base_url
        if base_url and not base_url.startswith(("http://", "https://")):
            self._service_discovery = create_service_discovery(base_url)
            if self._service_discovery is None:
                await self._http_client.aclose()
                raise ValueError(f"Invalid service discovery address: {base_url}")
            await self._service_discovery.start()
            ep = self._service_discovery.get_one_endpoint()
            if ep is not None:
                base_url = f"http://{ep.host}"
                logger.info(
                    "service discovery resolved manager endpoint",
                    step=_DISCOVERY_STEP,
                    tags={
                        "discovery_type": self._service_discovery.get_type(),
                        "base_url": base_url,
                    },
                )
            else:
                logger.warning(
                    "service discovery returned no endpoints; "
                    "waiting for background refresh",
                    step=_DISCOVERY_STEP,
                    tags={"discovery_url": base_url},
                )

        self.base_url = base_url.rstrip("/")
        self._discovery_url = self.base_url

        if self._auto_discover_leader:
            try:
                if await self.is_ready():
                    await self._discover_leader()
            except Exception as e:
                logger.warning(
                    "initial leader discovery failed; keeping base URL",
                    step=_DISCOVERY_STEP,
                    tags={
                        "base_url": self.base_url,
                        "error": e.__class__.__name__,
                        "message": str(e),
                    },
                )
            self._refresh_task = asyncio.create_task(
                self._leader_refresh_loop(),
                name="kvcm-leader-refresh",
            )
        self._started = True

    async def is_ready(self) -> bool:
        """Return whether requests have a usable HTTP endpoint."""

        if self.base_url.startswith(("http://", "https://")):
            return True
        if self._service_discovery is None:
            return False
        endpoint = self._service_discovery.get_one_endpoint()
        if endpoint is None:
            return False
        async with self._leader_lock:
            # Re-check under the lock: a concurrent leader discovery may have
            # resolved base_url already, and it must not be clobbered.
            if not self.base_url.startswith(("http://", "https://")):
                self.base_url = f"http://{endpoint.host}"
                logger.info(
                    "service discovery recovered manager endpoint",
                    step=_DISCOVERY_STEP,
                    tags={
                        "discovery_type": self._service_discovery.get_type(),
                        "base_url": self.base_url,
                    },
                )
        return True

    # ----- leader discovery internals -----

    @staticmethod
    def _get_status_code(response_data: dict[str, Any]) -> str | None:
        code: str | None = response_data.get("header", {}).get("status", {}).get("code")
        return code

    async def _discover_leader(self) -> bool:
        snapshot = self.base_url
        async with self._leader_lock:
            if self.base_url != snapshot:
                return True
            return await self._do_discover_leader()

    def _resolve_discovery_url(self) -> str:
        """Return a fresh URL for leader discovery queries.

        When service discovery is available, pick a (possibly different)
        endpoint each time so we are not stuck on a single dead node.
        Falls back to the static ``_discovery_url`` seed address.
        """
        if self._service_discovery is not None:
            ep = self._service_discovery.get_one_endpoint()
            if ep is not None:
                return f"http://{ep.host}"
        return self._discovery_url

    async def _do_discover_leader(self) -> bool:
        """Actual discovery logic. Must be called under ``_leader_lock``."""
        url = self._resolve_discovery_url()
        try:
            try:
                resp = await self._http_client.post(
                    url + "/api/getClusterInfo",
                    json={
                        "trace_id": f"leader_discovery_{time.monotonic()}",
                        "instance_id": self._instance_id,
                    },
                    headers=self.headers,
                    timeout=self._request_timeout_seconds,
                )
            except Exception as e:
                logger.warning(
                    "leader discovery request failed",
                    step=_DISCOVERY_STEP,
                    tags={
                        "url": url,
                        "error": e.__class__.__name__,
                        "message": str(e),
                    },
                )
                return False

            if resp.status_code != 200:
                logger.warning(
                    "leader discovery returned unexpected status",
                    step=_DISCOVERY_STEP,
                    tags={"url": url, "status_code": resp.status_code},
                )
                return False

            try:
                data = resp.json()
            except Exception as e:
                logger.warning(
                    "leader discovery response is not valid JSON",
                    step=_DISCOVERY_STEP,
                    tags={
                        "url": url,
                        "error": e.__class__.__name__,
                        "message": str(e),
                    },
                )
                return False

            if self._get_status_code(data) != "OK":
                msg = data.get("header", {}).get("status", {}).get("message", "unknown")
                logger.warning(
                    "leader discovery returned error",
                    step=_DISCOVERY_STEP,
                    tags={
                        "url": url,
                        "kvcm_status_code": self._get_status_code(data) or "Unknown",
                        "message": msg,
                    },
                )
                return False

            leader_ep = data.get("leader_endpoint")
            if (
                not leader_ep
                or not leader_ep.get("host")
                or not leader_ep.get("meta_http_port")
            ):
                logger.warning(
                    "leader discovery response missing leader endpoint",
                    step=_DISCOVERY_STEP,
                    tags={"url": url},
                )
                return False

            new_url = f"http://{leader_ep['host']}:{leader_ep['meta_http_port']}"
            if new_url != self.base_url:
                logger.info(
                    "leader discovery switched manager endpoint",
                    step=_DISCOVERY_STEP,
                    tags={
                        "previous_base_url": self.base_url,
                        "base_url": new_url,
                    },
                )
                self.base_url = new_url
            return True
        finally:
            self._last_discover_time = time.monotonic()

    async def _leader_refresh_loop(self) -> None:
        """Background daemon: periodically refresh leader address."""
        while not self._closed:
            try:
                await asyncio.wait_for(
                    self._refresh_event.wait(),
                    timeout=self._discovery_refresh_interval,
                )
            except TimeoutError:
                pass
            self._refresh_event.clear()
            if self._closed:
                break
            remaining = self._min_discover_interval - (
                time.monotonic() - self._last_discover_time
            )
            if remaining > 0:
                await asyncio.sleep(remaining)
                if self._closed:
                    break
            try:
                await self._discover_leader()
            except Exception as e:
                logger.warning(
                    "background leader refresh failed",
                    step=_DISCOVERY_STEP,
                    tags={
                        "error": e.__class__.__name__,
                        "message": str(e),
                    },
                )

    # ----- HTTP request helpers -----

    def _notify_leader_refresh(self) -> None:
        """Wake the background leader-refresh loop when discovery is enabled."""
        if self._auto_discover_leader:
            self._refresh_event.set()

    async def _request(
        self,
        endpoint: str,
        data: dict[str, Any],
        check_response: bool = True,
    ) -> dict[str, Any]:
        retries_left = self._leader_retry_count if self._auto_discover_leader else 0
        retry_count = 0

        while True:
            url = self.base_url + endpoint
            try:
                response = await self._http_client.post(
                    url,
                    json=data,
                    headers=self.headers,
                    timeout=self._request_timeout_seconds,
                )
            except httpx.TimeoutException as e:
                logger.warning(
                    "kvcm request timed out",
                    step=_REQUEST_STEP,
                    tags={
                        "url": url,
                        "timeout_seconds": self._request_timeout_seconds,
                        "error": e.__class__.__name__,
                    },
                )
                self._notify_leader_refresh()
                raise
            except httpx.ConnectError as e:
                logger.warning(
                    "kvcm request connection failed; notifying leader refresh",
                    step=_REQUEST_STEP,
                    tags={
                        "url": url,
                        "error": e.__class__.__name__,
                        "message": str(e),
                    },
                )
                self._notify_leader_refresh()
                raise
            except httpx.RequestError as e:
                logger.warning(
                    "kvcm request failed",
                    step=_REQUEST_STEP,
                    tags={
                        "url": url,
                        "error": e.__class__.__name__,
                        "message": str(e),
                    },
                )
                raise

            try:
                response.raise_for_status()
            except httpx.HTTPStatusError:
                logger.warning(
                    "kvcm request returned http error status",
                    step=_REQUEST_STEP,
                    tags={"url": url, "status_code": response.status_code},
                )
                raise
            try:
                payload: dict[str, Any] = response.json()
            except ValueError as e:
                logger.warning(
                    "kvcm response is not valid JSON",
                    step=_REQUEST_STEP,
                    tags={
                        "url": url,
                        "status_code": response.status_code,
                        "error": e.__class__.__name__,
                        "message": str(e),
                    },
                )
                raise

            # SERVER_NOT_LEADER handling
            if (
                self._auto_discover_leader
                and self._get_status_code(payload) == "SERVER_NOT_LEADER"
            ):
                if retries_left > 0:
                    retries_left -= 1
                    retry_count += 1
                    sleep_time = (
                        self._leader_retry_base_interval * retry_count
                        + random.uniform(0, self._leader_retry_base_interval)
                    )
                    logger.warning(
                        "kvcm request returned SERVER_NOT_LEADER; retrying",
                        step=_REQUEST_STEP,
                        tags={
                            "endpoint": endpoint,
                            "retry_delay_seconds": round(sleep_time, 3),
                            "retries_left": retries_left,
                        },
                    )
                    await asyncio.sleep(sleep_time)
                    await self._discover_leader()
                    continue
                if retries_left <= 0:
                    logger.error(
                        "kvcm leader discovery retries exhausted",
                        step=_REQUEST_STEP,
                        tags={"endpoint": endpoint},
                    )
                    # A leader failover in progress is a transient outage, not
                    # a report rejection: check_response would raise the
                    # rejected-report error and the batch would be dropped
                    # permanently instead of retried.
                    raise KvcmUnavailableError(
                        f"KVCM leader unavailable for {endpoint}: "
                        "SERVER_NOT_LEADER and leader discovery exhausted",
                        status_code="SERVER_NOT_LEADER",
                        reason="leader_retry_exhausted",
                        retry_count=retry_count,
                    )

            if check_response:
                status = payload.get("header", {}).get("status", {})
                if status.get("code") != "OK":
                    item_results = payload.get("item_results")
                    item_results_detail = (
                        f"; item_results={item_results!r}" if item_results else ""
                    )
                    raise KvcmResponseRejectedError(
                        f"KVCM {endpoint} failed: "
                        f"{status.get('code')} {status.get('message')}"
                        f"{item_results_detail}",
                        status_code=self._get_status_code(payload) or "UNKNOWN",
                        retry_count=retry_count,
                    )
            if retry_count:
                payload["_subscriber_retry_count"] = retry_count
            return payload

    # ----- public API (matches KvCacheManagerClient interface) -----

    async def register_instance(
        self, data: dict[str, Any], check_response: bool = True
    ) -> dict[str, Any]:
        return await self._request("/api/registerInstance", data, check_response)

    async def report_event(
        self, data: dict[str, Any], check_response: bool = True
    ) -> dict[str, Any]:
        return await self._request("/api/reportEvent", data, check_response)

    async def close(self) -> None:
        self._closed = True
        self._refresh_event.set()
        if self._refresh_task is not None:
            self._refresh_task.cancel()
            try:
                await self._refresh_task
            except asyncio.CancelledError:
                pass
            self._refresh_task = None
        await self._http_client.aclose()
        if self._service_discovery is not None:
            await self._service_discovery.close()
