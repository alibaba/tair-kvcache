import random
import threading
import time
from typing import List

import requests

from kv_cache_manager.py_connector.common.logger import logger


class KvCacheManagerClient:
    def __init__(self, base_url, *, instance_id="", auto_discover_leader=False, leader_retry_count=1,
                 leader_retry_base_interval_seconds=0.005,
                 discovery_refresh_interval_seconds=30,
                 min_discover_interval_seconds=1):
        """
        Args:
            base_url: Manager service address. When auto_discover_leader is enabled, leader
                discovery will always query this address (not the current leader). It is
                recommended to use a load-balancer address that fronts all manager nodes.
        """
        self.base_url = base_url
        self.session = requests.Session()
        self.headers = {'Accept': 'application/json', 'Content-Type': 'application/json'}

        # Leader discovery settings
        self._instance_id = instance_id
        # Immutable seed address used for all discovery requests.
        # When auto_discover_leader is enabled, getClusterInfo is always sent to this
        # address rather than the current base_url, so it should be a stable entry point
        # (e.g. a load-balancer) that can reach any manager node.
        self._discovery_url = base_url
        self._auto_discover_leader = auto_discover_leader
        self._leader_retry_count = leader_retry_count
        self._leader_retry_base_interval = leader_retry_base_interval_seconds
        self._discovery_refresh_interval = discovery_refresh_interval_seconds
        self._min_discover_interval = min_discover_interval_seconds

        self._leader_lock = threading.Lock()
        self._refresh_event = threading.Event()
        self._closed = threading.Event()
        self._last_discover_time = 0.0  # time.monotonic()
        self._refresh_thread = None

        if self._auto_discover_leader:
            try:
                self._discover_leader()
            except Exception as e:
                logger.warning("Initial leader discovery failed, keeping original base_url %s: %s",
                               self.base_url, e)
            self._refresh_thread = threading.Thread(
                target=self._leader_refresh_loop, daemon=True,
                name="kvcm-leader-refresh")
            self._refresh_thread.start()

    @staticmethod
    def _get_status_code(response_data):
        """Extract status code from a standard API response."""
        return response_data.get('header', {}).get('status', {}).get('code')

    def _discover_leader(self):
        """Query getClusterInfo to discover and switch to the leader node.
        Returns True if base_url is up-to-date, False on failure.
        """
        snapshot = self.base_url
        with self._leader_lock:
            # Another thread already updated base_url
            if self.base_url != snapshot:
                return True
            return self._do_discover_leader()

    def _do_discover_leader(self):
        """Actual discovery logic. Must be called under _leader_lock."""
        url = self._discovery_url
        try:
            try:
                resp = requests.post(
                    url + '/api/getClusterInfo',
                    json={
                        "trace_id": f"leader_discovery_{time.monotonic()}",
                        "instance_id": self._instance_id,
                    },
                    headers=self.headers,
                    timeout=5,
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
        finally:
            self._last_discover_time = time.monotonic()

    def _leader_refresh_loop(self):
        """Background daemon: periodically refresh leader address, supports urgent wakeup."""
        while not self._closed.is_set():
            self._refresh_event.wait(timeout=self._discovery_refresh_interval)
            self._refresh_event.clear()
            if self._closed.is_set():
                break
            # Min interval protection: wait remaining time instead of skipping
            remaining = self._min_discover_interval - (time.monotonic() - self._last_discover_time)
            if remaining > 0:
                if self._closed.wait(timeout=remaining):
                    break
            try:
                self._discover_leader()
            except Exception as e:
                logger.warning("Background leader refresh failed: %s", e)

    def _make_request(self, method, endpoint, data=None):
        """Helper method to make HTTP requests to the service"""
        url = self.base_url + endpoint

        if method == 'POST':
            response = self.session.post(url, json=data, headers=self.headers)
        elif method == 'GET':
            response = self.session.get(url, params=data, headers=self.headers)
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
            except requests.ConnectionError:
                if self._auto_discover_leader:
                    logger.warning("Connection to %s failed, notifying background refresh", self.base_url)
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
                    if self._discover_leader():
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

    def get_cache_location(self, data, check_response=True):
        """Get cache location for specified block keys"""
        return self._make_api_request('/api/getCacheLocation', data, check_response)

    def start_write_cache(self, data, check_response=True):
        """Start writing cache data"""
        return self._make_api_request('/api/startWriteCache', data, check_response)

    def finish_write_cache(self, data, check_response=True):
        """Finish writing cache data"""
        return self._make_api_request('/api/finishWriteCache', data, check_response)

    def remove_cache(self, data, check_response=True):
        """Remove cache data for specified block keys"""
        return self._make_api_request('/api/removeCache', data, check_response)

    def trim_cache(self, data, check_response=True):
        """Trim cache data based on specified strategy"""
        return self._make_api_request('/api/trimCache', data, check_response)

    def get_cluster_info(self, data, check_response=True):
        """Get cluster info including leader endpoint (leader discovery API)"""
        return self._make_api_request('/api/getClusterInfo', data, check_response)

    def close(self):
        """Close the HTTP session and stop background refresh thread"""
        self._closed.set()
        self._refresh_event.set()
        if self._refresh_thread and self._refresh_thread.is_alive():
            self._refresh_thread.join(timeout=5)
        self.session.close()
