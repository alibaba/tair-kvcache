"""Manager-side location queries for the external-match hook.

``get_num_new_matched_tokens`` must never block the scheduler loop, so
queries run on the http executor and the answer is cached per request
until consumed. The cache is request-lifecycle scoped -- produced by the
match hook, consumed by ``update_state_after_alloc`` (which turns the
answer into a LoadRequest) and invalidated when the request retires. No
TTL: the producer and the consumer are two hooks of the same request's
life, so nothing can outlive its usefulness for long.
"""

import threading
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

from kv_cache_manager.py_connector.common.manager_client import KvCacheManagerClient
from kv_cache_manager.py_connector.common.logger import logger


@dataclass
class _QueryEntry:
    """One in-flight (or answered) location query for a request."""

    computed_blocks: int            # offset the query was issued at
    locations: list = None          # None until the manager answered
    in_flight: bool = True


class LocationQueryManager:
    """Per-request location query cache: produce on match, consume on alloc."""

    def __init__(self, manager_client: KvCacheManagerClient, http_executor,
                 instance_id: str, async_get_cache_location: bool):
        self._manager_client = manager_client
        self._http_executor = http_executor
        self._instance_id = instance_id
        self._async_get_cache_location = async_get_cache_location
        self._lock = threading.Lock()
        self._entries: Dict[str, _QueryEntry] = {}

    def shutdown(self):
        pass

    def _fetch_from_manager(self, request, computed_blocks: int):
        """Run the actual GetCacheLocation call (http thread or inline)."""
        get_request = {
            "trace_id": request.request_id,
            "token_ids": request.prompt_token_ids,
            "instance_id": self._instance_id,
            "query_type": "QT_PREFIX_MATCH",
            "block_mask": {"offset": computed_blocks},
        }
        logger.debug("get_kvcache_location request: %s", get_request)
        result = self._manager_client.get_cache_location(get_request)
        logger.debug("get_kvcache_location result: %s", result)
        return result["locations"]

    def _query_async(self, request, computed_blocks: int) -> None:
        def run():
            try:
                locations = self._fetch_from_manager(request, computed_blocks)
            except Exception as e:
                logger.warning("get_cache_location error, request_id: %s, error: %s",
                               request.request_id, e)
                with self._lock:
                    # Drop the entry: the next match hook re-issues the query.
                    if (entry := self._entries.get(request.request_id)) is not None \
                            and entry.in_flight:
                        self._entries.pop(request.request_id, None)
                return
            with self._lock:
                entry = self._entries.get(request.request_id)
                if entry is None or not entry.in_flight:
                    # Re-issued with a different offset meanwhile.
                    return
                entry.locations = locations
                entry.in_flight = False

        self._http_executor.submit(run)

    def get_locations_for_query(self, request, computed_blocks: int) -> Optional[list]:
        """Ask (or re-ask) for the request's external match.

        Returns the locations when the answer is already cached for this
        offset, None while a query is in flight (the scheduler re-asks next
        step), and [] when the query answered "no hit". A failed query drops
        the entry so the next hook call re-issues it.
        """
        req_id = request.request_id
        with self._lock:
            entry = self._entries.get(req_id)
            if entry is not None and entry.in_flight:
                # A query for this request is already running (this offset or
                # an older one); re-issuing on every scheduler re-ask would
                # fan out one full getCacheLocation per engine step while the
                # answer is in flight. Wait for the running one instead.
                return None
            if entry is not None:
                if entry.computed_blocks == computed_blocks:
                    return entry.locations
                # Stale answer for an older offset: re-issue below.
                self._entries.pop(req_id, None)
                entry = None
            if entry is None:
                self._entries[req_id] = _QueryEntry(computed_blocks=computed_blocks)

        if self._async_get_cache_location:
            self._query_async(request, computed_blocks)
            return None
        try:
            locations = self._fetch_from_manager(request, computed_blocks)
        except Exception as e:
            logger.warning("get_cache_location error, request_id: %s, error: %s", req_id, e)
            with self._lock:
                self._entries.pop(req_id, None)
            return []
        with self._lock:
            cur = self._entries.get(req_id)
            if cur is not None:
                cur.locations = locations
                cur.in_flight = False
        return locations

    def store_result(self, req_id: str, locations: list) -> None:
        """Overwrite the cached answer (the match hook clamps it to a
        vLLM-safe prefix before the allocation consumes it)."""
        with self._lock:
            entry = self._entries.get(req_id)
            if entry is not None:
                entry.locations = locations
                entry.in_flight = False

    def consume_locations(self, req_id: str) -> Optional[Tuple[list, int]]:
        """Pop the answered query: (locations, computed_blocks), or None when
        nothing is cached (no query was issued / still in flight)."""
        with self._lock:
            entry = self._entries.pop(req_id, None)
            if entry is None or entry.in_flight:
                return None
            return entry.locations, entry.computed_blocks

    def invalidate(self, req_id: str) -> None:
        """Drop any cached query for a request that is going away."""
        with self._lock:
            self._entries.pop(req_id, None)
