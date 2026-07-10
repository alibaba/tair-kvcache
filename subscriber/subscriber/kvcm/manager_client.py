"""KVCM manager HTTP client with service discovery and leader discovery.

Extracted from ``store.py`` to keep manager-client concerns separate from
the store/transfer layer.
"""

from __future__ import annotations

import logging
import random
import threading
import time
from typing import Any, Dict, Optional

import requests

from subscriber.kvcm.service_discovery import (
    ServiceDiscovery,
    create_service_discovery,
)

logger = logging.getLogger(__name__)


class HttpKvCacheManagerClient:
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
        request_timeout_seconds: float = 1.0,
    ):
        """Initialize the HTTP manager client.

        Args:
            base_url: Manager service address. Supported formats:
                - ``http://host:port`` or ``https://host:port`` — used directly
                - ``static://ip:port[,ip:port]...`` — resolved via static
                  service discovery (round-robin)
                - ``spectrum://<vsid>[:port][?cache_time=<sec>&timeout=<ms>&retry_time=<n>]``
                  — resolved via Spectrum gateway
        """
        self.session = requests.Session()
        self.headers = {
            "Accept": "application/json",
            "Content-Type": "application/json",
        }
        self._request_timeout_seconds = float(request_timeout_seconds)
        if self._request_timeout_seconds <= 0:
            raise ValueError("request_timeout_seconds must be positive")

        # --- service discovery ---
        # http):// → use directly; other schemes → try service discovery
        self._service_discovery: Optional[ServiceDiscovery] = None
        if base_url and not base_url.startswith("http://"):
            self._service_discovery = create_service_discovery(base_url)
            if self._service_discovery is not None:
                ep = self._service_discovery.get_one_endpoint()
                if ep is not None:
                    base_url = f"http://{ep.host}"
                    logger.info(
                        "Service discovery (%s) resolved manager endpoint: %s",
                        self._service_discovery.get_type(),
                        base_url,
                    )
                else:
                    logger.warning(
                        "Service discovery returned no endpoints for %s",
                        base_url,
                    )

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

        self._leader_lock = threading.Lock()
        self._refresh_event = threading.Event()
        self._closed = threading.Event()
        self._last_discover_time: float = 0.0
        self._refresh_thread: Optional[threading.Thread] = None

        if self._auto_discover_leader:
            try:
                self._discover_leader()
            except Exception as e:
                logger.warning(
                    "Initial leader discovery failed, keeping base_url %s: %s",
                    self.base_url,
                    e,
                )
            self._refresh_thread = threading.Thread(
                target=self._leader_refresh_loop,
                daemon=True,
                name="kvcm-leader-refresh",
            )
            self._refresh_thread.start()

    # ----- leader discovery internals -----

    @staticmethod
    def _get_status_code(response_data: Dict[str, Any]) -> Optional[str]:
        return response_data.get("header", {}).get("status", {}).get("code")

    def _discover_leader(self) -> bool:
        snapshot = self.base_url
        with self._leader_lock:
            if self.base_url != snapshot:
                return True
            return self._do_discover_leader()

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

    def _do_discover_leader(self) -> bool:
        """Actual discovery logic. Must be called under ``_leader_lock``."""
        url = self._resolve_discovery_url()
        try:
            try:
                resp = requests.post(
                    url + "/api/getClusterInfo",
                    json={
                        "trace_id": f"leader_discovery_{time.monotonic()}",
                        "instance_id": self._instance_id,
                    },
                    headers=self.headers,
                    timeout=self._request_timeout_seconds,
                )
            except Exception as e:
                logger.warning("Leader discovery request to %s failed: %s", url, e)
                return False

            if resp.status_code != 200:
                logger.warning(
                    "Leader discovery to %s returned status %d",
                    url,
                    resp.status_code,
                )
                return False

            try:
                data = resp.json()
            except Exception as e:
                logger.warning(
                    "Leader discovery response from %s is not valid JSON: %s", url, e
                )
                return False

            if self._get_status_code(data) != "OK":
                msg = (
                    data.get("header", {}).get("status", {}).get("message", "unknown")
                )
                logger.warning(
                    "Leader discovery from %s returned error: %s", url, msg
                )
                return False

            leader_ep = data.get("leader_endpoint")
            if (
                not leader_ep
                or not leader_ep.get("host")
                or not leader_ep.get("meta_http_port")
            ):
                logger.warning(
                    "Leader discovery from %s: leader_endpoint missing or incomplete",
                    url,
                )
                return False

            new_url = f"http://{leader_ep['host']}:{leader_ep['meta_http_port']}"
            if new_url != self.base_url:
                logger.info(
                    "Leader discovered: switching base_url from %s to %s",
                    self.base_url,
                    new_url,
                )
                self.base_url = new_url
            return True
        finally:
            self._last_discover_time = time.monotonic()

    def _leader_refresh_loop(self) -> None:
        """Background daemon: periodically refresh leader address."""
        while not self._closed.is_set():
            self._refresh_event.wait(timeout=self._discovery_refresh_interval)
            self._refresh_event.clear()
            if self._closed.is_set():
                break
            remaining = self._min_discover_interval - (
                time.monotonic() - self._last_discover_time
            )
            if remaining > 0:
                if self._closed.wait(timeout=remaining):
                    break
            try:
                self._discover_leader()
            except Exception as e:
                logger.warning("Background leader refresh failed: %s", e)

    # ----- HTTP request helpers -----

    def _request(
        self,
        endpoint: str,
        data: Dict[str, Any],
        check_response: bool = True,
    ) -> Dict[str, Any]:
        retries_left = self._leader_retry_count if self._auto_discover_leader else 0

        while True:
            url = self.base_url + endpoint
            try:
                response = self.session.post(
                    url,
                    json=data,
                    headers=self.headers,
                    timeout=self._request_timeout_seconds,
                )
            except requests.ConnectionError:
                if self._auto_discover_leader:
                    logger.warning(
                        "Connection to %s failed, notifying background refresh",
                        self.base_url,
                    )
                    self._refresh_event.set()
                raise

            response.raise_for_status()
            payload = response.json()

            # SERVER_NOT_LEADER handling
            if (
                self._auto_discover_leader
                and self._get_status_code(payload) == "SERVER_NOT_LEADER"
            ):
                if retries_left > 0:
                    retries_left -= 1
                    attempt = self._leader_retry_count - retries_left
                    sleep_time = (
                        self._leader_retry_base_interval * attempt
                        + random.uniform(0, self._leader_retry_base_interval)
                    )
                    logger.warning(
                        "Request to %s returned SERVER_NOT_LEADER, "
                        "retrying after %.3fs (retries left: %d)",
                        endpoint,
                        sleep_time,
                        retries_left,
                    )
                    time.sleep(sleep_time)
                    if self._discover_leader():
                        continue
                if retries_left <= 0:
                    logger.error(
                        "All leader discovery retries exhausted for %s", endpoint
                    )

            if check_response:
                status = payload.get("header", {}).get("status", {})
                if status.get("code") != "OK":
                    raise RuntimeError(
                        f"KVCM {endpoint} failed: "
                        f"{status.get('code')} {status.get('message')}"
                    )
            return payload

    # ----- public API (matches KvCacheManagerClient interface) -----

    def register_instance(
        self, data: Dict[str, Any], check_response: bool = True
    ) -> Dict[str, Any]:
        return self._request("/api/registerInstance", data, check_response)

    def report_event(
        self, data: Dict[str, Any], check_response: bool = True
    ) -> Dict[str, Any]:
        return self._request("/api/reportEvent", data, check_response)

    def close(self) -> None:
        self._closed.set()
        self._refresh_event.set()
        if self._refresh_thread and self._refresh_thread.is_alive():
            self._refresh_thread.join(timeout=5)
        self.session.close()
        if self._service_discovery is not None:
            self._service_discovery.close()
