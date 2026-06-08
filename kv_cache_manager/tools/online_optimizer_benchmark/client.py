"""HTTP client for OptimizerService endpoints."""

import uuid
import logging
from typing import Any, List

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from .config import BenchmarkConfig

logger = logging.getLogger(__name__)


class OptimizerClient:
    """Thread-safe HTTP client wrapping OptimizerService REST API."""

    API_PREFIX = "/api/optimizer"

    def __init__(self, config: BenchmarkConfig):
        self._base_url = config.base_url
        self._timeout = (config.connection_timeout, config.request_timeout)
        self._session = self._build_session()

    def _build_session(self) -> requests.Session:
        session = requests.Session()
        retry_strategy = Retry(total=2, backoff_factor=0.1,
                               status_forcelist=[502, 503, 504])
        adapter = HTTPAdapter(
            pool_connections=32,
            pool_maxsize=64,
            max_retries=retry_strategy,
        )
        session.mount("http://", adapter)
        session.mount("https://", adapter)
        session.headers.update({"Content-Type": "application/json"})
        return session

    def _post(self, path: str, payload: dict) -> dict:
        url = f"{self._base_url}{self.API_PREFIX}/{path}"
        response = self._session.post(url, json=payload, timeout=self._timeout)
        response.raise_for_status()
        return response.json()

    @staticmethod
    def _new_trace_id() -> str:
        return str(uuid.uuid4())

    # ── InstanceGroup management ──

    def create_instance_group(
        self,
        name: str,
        capacity_gb: float,
        indexer_type: str = "bst_lru",
        max_key_count: int = 0,
    ) -> dict:
        payload = {
            "trace_id": self._new_trace_id(),
            "instance_group": {
                "name": name,
                "enabled": True,
                "capacity_gb": [capacity_gb],
                "primary_capacity_index": 0,
                "indexer_type": indexer_type,
                "max_key_count": max_key_count,
            },
        }
        return self._post("createInstanceGroup", payload)

    def remove_instance_group(self, name: str) -> dict:
        payload = {"trace_id": self._new_trace_id(), "name": name}
        return self._post("removeInstanceGroup", payload)

    # ── Instance management ──

    def register_instance(
        self,
        instance_group: str,
        instance_id: str,
        block_size: int,
    ) -> dict:
        payload = {
            "trace_id": self._new_trace_id(),
            "instance_group": instance_group,
            "instance_id": instance_id,
            "block_size": block_size,
            "location_spec_infos": [{"name": "default", "size": 1}],
        }
        return self._post("registerInstance", payload)

    def remove_instance(self, instance_id: str) -> dict:
        payload = {"trace_id": self._new_trace_id(), "instance_id": instance_id}
        return self._post("removeInstance", payload)

    def list_instances(self, instance_group: str = "") -> dict:
        payload = {"trace_id": self._new_trace_id(), "instance_group": instance_group}
        return self._post("listInstances", payload)

    # ── TraceQuery ──

    def trace_query(self, instance_id: str, block_keys: List[int]) -> dict:
        """Send a TraceQuery and return the parsed JSON response."""
        payload = {
            "trace_id": self._new_trace_id(),
            "instance_id": instance_id,
            "block_keys": block_keys,
        }
        return self._post("traceQuery", payload)

    def trace_query_raw(self, instance_id: str, block_keys: List[int]) -> requests.Response:
        """Send a TraceQuery and return the raw Response (for latency measurement)."""
        url = f"{self._base_url}{self.API_PREFIX}/traceQuery"
        payload = {
            "trace_id": self._new_trace_id(),
            "instance_id": instance_id,
            "block_keys": block_keys,
        }
        return self._session.post(url, json=payload, timeout=self._timeout)

    # ── Stats ──

    def reset_stats(self, instance_id: str) -> dict:
        payload = {"trace_id": self._new_trace_id(), "instance_id": instance_id}
        return self._post("resetStats", payload)

    def close(self):
        self._session.close()
