import random
import threading
import time
from typing import Any, Mapping, Optional

import requests

from kv_cache_manager.py_connector.common.logger import logger
from kv_cache_manager.py_connector.common.service_discovery import (
    ServiceDiscovery,
    ServiceEndpoint,
)
from kv_cache_manager.py_connector.common.service_discovery_factory import (
    create_service_discovery,
)

_LEADER_DISCOVERY_TIMEOUT_SECONDS = 5.0


class KvCacheManagerClient:
    @classmethod
    def from_connector_config(
        cls, config: Mapping[str, Any]
    ) -> "KvCacheManagerClient":
        """Create a client from the shared connector configuration surface."""
        return cls(
            config["manager_uri"],
            instance_id=config.get("instance_id", ""),
            auto_discover_leader=config.get("auto_discover_leader", False),
            leader_retry_count=config.get("leader_retry_count", 1),
            leader_retry_base_interval_seconds=config.get(
                "leader_retry_base_interval_seconds", 0.005
            ),
            discovery_refresh_interval_seconds=config.get(
                "discovery_refresh_interval_seconds", 30
            ),
            min_discover_interval_seconds=config.get(
                "min_discover_interval_seconds", 1
            ),
            request_timeout_seconds=config.get("request_timeout_seconds", 1.0),
        )

    def __init__(self, base_url, *, instance_id="", auto_discover_leader=False, leader_retry_count=1,
                 leader_retry_base_interval_seconds=0.005,
                 discovery_refresh_interval_seconds=30,
                 min_discover_interval_seconds=1,
                 request_timeout_seconds=1.0):
        """
        Args:
            base_url: Manager HTTP(S) address or a service-discovery URL. When
                auto_discover_leader is enabled, leader discovery always starts from
                this entry point instead of the current leader.
            request_timeout_seconds: Timeout in seconds for regular Manager HTTP
                requests. Defaults to 1 second. Leader discovery keeps its dedicated
                5-second timeout.
        """
        self._request_timeout_seconds = float(request_timeout_seconds)
        if self._request_timeout_seconds <= 0:
            raise ValueError("request_timeout_seconds must be positive")

        self.session = requests.Session()
        self.headers = {'Accept': 'application/json', 'Content-Type': 'application/json'}

        # Resolve service-discovery URLs before issuing any HTTP requests. Keep the
        # discovery object alive so each leader refresh can start from a fresh endpoint.
        self._service_discovery: Optional[ServiceDiscovery] = None
        resolved_url = base_url
        if not self._is_http_url(base_url):
            self._service_discovery = create_service_discovery(base_url)
            if self._service_discovery is None:
                self.session.close()
                raise ValueError(
                    f"failed to create service discovery from manager address {base_url!r}"
                )

            try:
                endpoint = self._service_discovery.get_one_endpoint()
            except Exception as e:
                self._close_service_discovery()
                self.session.close()
                raise RuntimeError(
                    f"failed to resolve manager endpoints from {base_url!r}"
                ) from e
            if endpoint is None:
                self._close_service_discovery()
                self.session.close()
                raise RuntimeError(
                    f"service discovery returned no manager endpoints for {base_url!r}"
                )

            resolved_url = self._endpoint_url(endpoint)
            logger.info(
                "Service discovery (%s) resolved manager endpoint: %s",
                self._service_discovery.get_type(),
                resolved_url,
            )

        self.base_url = resolved_url.rstrip('/')

        # Manager route settings
        self._instance_id = instance_id
        # Immutable fallback address used when service discovery cannot return a fresh
        # endpoint. For a direct HTTP URL, this remains the stable discovery seed.
        self._discovery_url = self.base_url
        self._auto_discover_leader = auto_discover_leader
        self._leader_retry_count = leader_retry_count
        self._leader_retry_base_interval = leader_retry_base_interval_seconds
        self._discovery_refresh_interval = discovery_refresh_interval_seconds
        self._min_discover_interval = min_discover_interval_seconds

        self._route_lock = threading.Lock()
        self._refresh_event = threading.Event()
        self._closed = threading.Event()
        self._last_route_refresh_time = 0.0  # time.monotonic()
        self._refresh_thread = None

        if self._auto_discover_leader:
            try:
                self._refresh_manager_route()
            except Exception as e:
                logger.warning("Initial leader discovery failed, keeping original base_url %s: %s",
                               self.base_url, e)

        # One route-refresh thread serves both modes. Leader discovery wakes
        # periodically; service-discovery-only mode waits for transport failures.
        if self._auto_discover_leader or self._service_discovery is not None:
            self._refresh_thread = threading.Thread(
                target=self._route_refresh_loop, daemon=True,
                name="kvcm-route-refresh")
            self._refresh_thread.start()

    @staticmethod
    def _is_http_url(url):
        return url.startswith(('http://', 'https://'))

    @staticmethod
    def _endpoint_url(endpoint: ServiceEndpoint):
        """Build a Manager URL from the host:port-only HTTP endpoint contract."""
        return f"http://{endpoint.host}"

    def _close_service_discovery(self):
        discovery = self._service_discovery
        self._service_discovery = None
        if discovery is not None:
            discovery.close()

    @staticmethod
    def _get_status_code(response_data):
        """Extract status code from a standard API response."""
        return response_data.get('header', {}).get('status', {}).get('code')

    def _refresh_manager_route(self, force_service_refresh=False):
        """Refresh the Manager route through service and optional leader discovery."""
        snapshot = self.base_url
        with self._route_lock:
            # Another thread already updated base_url
            if self.base_url != snapshot:
                return True

            try:
                route_url = self._resolve_discovery_url(force_service_refresh)
                if not self._auto_discover_leader:
                    if route_url != self.base_url:
                        logger.info(
                            "Service discovery refreshed manager route: %s -> %s",
                            self.base_url,
                            route_url,
                        )
                        self.base_url = route_url
                    return True
                return self._do_discover_leader(route_url)
            finally:
                self._last_route_refresh_time = time.monotonic()

    def _do_discover_leader(self, url):
        """Resolve and switch to the leader. Must be called under _route_lock."""
        try:
            resp = requests.post(
                url + '/api/getClusterInfo',
                json={
                    "trace_id": f"leader_discovery_{time.monotonic()}",
                    "instance_id": self._instance_id,
                },
                headers=self.headers,
                timeout=_LEADER_DISCOVERY_TIMEOUT_SECONDS,
            )
        except Exception as e:
            logger.warning("Leader discovery request to %s failed: %s", url, e)
            return False

        if resp.status_code != 200:
            logger.warning("Leader discovery to %s returned status %d", url, resp.status_code)
            return False

        try:
            data = resp.json()
        except Exception as e:
            logger.warning("Leader discovery response from %s is not valid JSON: %s", url, e)
            return False

        if self._get_status_code(data) != 'OK':
            msg = data.get('header', {}).get('status', {}).get('message', 'unknown')
            logger.warning("Leader discovery from %s returned error: %s", url, msg)
            return False

        leader_ep = data.get('leader_endpoint')
        if not leader_ep or not leader_ep.get('host') or not leader_ep.get('meta_http_port'):
            logger.warning("Leader discovery from %s: leader_endpoint missing or incomplete", url)
            return False

        new_url = f"http://{leader_ep['host']}:{leader_ep['meta_http_port']}"
        if new_url != self.base_url:
            logger.info("Leader discovered: switching base_url from %s to %s", self.base_url, new_url)
            self.base_url = new_url
        return True

    def _resolve_discovery_url(self, force_refresh=False):
        """Resolve a fresh leader-discovery seed, falling back to the initial one."""
        if self._service_discovery is not None:
            try:
                if force_refresh and not self._service_discovery.refresh():
                    logger.warning(
                        "Failed to force-refresh manager service discovery; using cached endpoints"
                    )
                endpoint = self._service_discovery.get_one_endpoint()
            except Exception as e:
                logger.warning(
                    "Failed to refresh manager endpoint through service discovery: %s",
                    e,
                )
            else:
                if endpoint is not None:
                    return self._endpoint_url(endpoint)
                logger.warning(
                    "Service discovery returned no manager endpoints; using %s",
                    self._discovery_url,
                )
        return self._discovery_url

    def _route_refresh_loop(self):
        """Background daemon for periodic leader and event-driven route refresh."""
        while not self._closed.is_set():
            timeout = (
                self._discovery_refresh_interval
                if self._auto_discover_leader
                else None
            )
            refresh_requested = self._refresh_event.wait(timeout=timeout)
            self._refresh_event.clear()
            if self._closed.is_set():
                break
            # Min interval protection: wait remaining time instead of skipping
            remaining = self._min_discover_interval - (
                time.monotonic() - self._last_route_refresh_time
            )
            if remaining > 0:
                if self._closed.wait(timeout=remaining):
                    break
            try:
                self._refresh_manager_route(
                    force_service_refresh=refresh_requested,
                )
            except Exception as e:
                logger.warning("Background manager route refresh failed: %s", e)

    def _make_request(self, method, endpoint, data=None):
        """Helper method to make HTTP requests to the service"""
        url = self.base_url + endpoint

        if method == 'POST':
            response = self.session.post(
                url,
                json=data,
                headers=self.headers,
                timeout=self._request_timeout_seconds,
            )
        elif method == 'GET':
            response = self.session.get(
                url,
                params=data,
                headers=self.headers,
                timeout=self._request_timeout_seconds,
            )
        else:
            raise ValueError(f"Unsupported HTTP method: {method}")

        return response

    def _check_response(self, endpoint, response, response_data):
        """Validate API response, raise AssertionError on failure."""
        if response.status_code != 200:
            raise AssertionError(f"Request to {endpoint} failed with status code {response.status_code}")

        if 'header' not in response_data:
            raise AssertionError(f"Response from {endpoint} missing 'header' field")

        if response_data['header']['status']['code'] != "OK":
            raise AssertionError(
                f"Request to {endpoint} failed with error: {response_data['header']['status']['message']}")

    def _make_api_request(self, endpoint, data=None, check_response=True):
        """Helper method to make POST requests to API endpoints and optionally validate response"""
        retries_left = self._leader_retry_count if self._auto_discover_leader else 0

        while True:
            try:
                response = self._make_request('POST', endpoint, data)
            except (requests.ConnectionError, requests.Timeout):
                # Never retry the current request after a transport failure: without
                # a response, the client cannot know whether Manager applied it, and
                # the API set includes non-idempotent writes. Refresh only benefits
                # subsequent calls.
                if self._refresh_thread is not None:
                    logger.warning(
                        "Transport request to %s failed, notifying route refresh",
                        self.base_url,
                    )
                    self._refresh_event.set()
                raise

            response_data = response.json()

            # SERVER_NOT_LEADER handling: rediscover leader and retry with backoff
            if self._auto_discover_leader and self._get_status_code(response_data) == 'SERVER_NOT_LEADER':
                if retries_left > 0:
                    retries_left -= 1
                    attempt = self._leader_retry_count - retries_left  # 1-based
                    sleep_time = self._leader_retry_base_interval * attempt + random.uniform(
                        0, self._leader_retry_base_interval)
                    logger.warning("Request to %s returned SERVER_NOT_LEADER, "
                                   "retrying after %.3fs (retries left: %d)",
                                   endpoint, sleep_time, retries_left)
                    time.sleep(sleep_time)
                    if self._refresh_manager_route():
                        continue
                if retries_left <= 0:
                    logger.error("All leader discovery retries exhausted for %s", endpoint)

            if check_response:
                self._check_response(endpoint, response, response_data)

            return response_data

    def register_instance(self, data, check_response=True):
        """Register an instance with the service"""
        return self._make_api_request('/api/registerInstance', data, check_response)

    def get_instance_info(self, data, check_response=True):
        """Get information about a registered instance"""
        return self._make_api_request('/api/getInstanceInfo', data, check_response)

    def get_cache_meta(self, data, check_response=True):
        """Get cache metadata for specified block keys"""
        return self._make_api_request('/api/getCacheMeta', data, check_response)

    def get_cache_location(self, data, check_response=True):
        """Get cache location for specified block keys"""
        return self._make_api_request('/api/getCacheLocation', data, check_response)

    def get_cache_location_len(self, data, check_response=True):
        """Get the number of cache locations matching the specified block keys"""
        return self._make_api_request('/api/getCacheLocationLen', data, check_response)

    def get_cache_locations_by_backend(self, data, check_response=True):
        """Get cache locations selected independently for each storage backend."""
        return self._make_api_request('/api/getCacheLocationsByBackend', data, check_response)

    def start_write_cache(self, data, check_response=True):
        """Start writing cache data"""
        return self._make_api_request('/api/startWriteCache', data, check_response)

    def finish_write_cache(self, data, check_response=True):
        """Finish writing cache data"""
        return self._make_api_request('/api/finishWriteCache', data, check_response)

    def remove_cache(self, data, check_response=True):
        """Remove cache data for specified block keys"""
        return self._make_api_request('/api/removeCache', data, check_response)

    def report_event(self, data, check_response=True):
        """Report node, cache block, host-down, or heartbeat events."""
        return self._make_api_request('/api/reportEvent', data, check_response)

    def trim_cache(self, data, check_response=True):
        """Trim cache data based on specified strategy"""
        return self._make_api_request('/api/trimCache', data, check_response)

    def get_cluster_info(self, data, check_response=True):
        """Get cluster info including leader endpoint (leader discovery API)"""
        return self._make_api_request('/api/getClusterInfo', data, check_response)

    def close(self):
        """Close the HTTP session, discovery client, and background refresh thread."""
        self._closed.set()
        self._refresh_event.set()
        if self._refresh_thread and self._refresh_thread.is_alive():
            self._refresh_thread.join(timeout=5)
        self.session.close()
        self._close_service_discovery()
