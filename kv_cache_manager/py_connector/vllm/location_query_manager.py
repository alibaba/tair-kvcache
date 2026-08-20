"""Manager-side location queries for the external-match hook.

``get_num_new_matched_tokens`` must never block the scheduler loop, so
queries run on the http executor and the answer is cached per request
until consumed. The cache is request-lifecycle scoped -- produced by the
match hook, consumed by ``update_state_after_alloc`` (which turns the
answer into a LoadRequest) and invalidated when the request retires. No
TTL: the producer and the consumer are two hooks of the same request's
life, so nothing can outlive its usefulness for long.

The cache key is the full query identity ``(req_id, query_type,
token_length, computed_blocks)`` -- the same key origin/main used. Two
re-asks are the same query only when the offset matches: a re-ask at a
grown offset (chunked prefill, re-ask after partial compute) is a
*different* query. It must neither be deduplicated against an older
offset's answer nor blocked by an older offset's in-flight query.
"""

import threading
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

from kv_cache_manager.py_connector.common.manager_client import KvCacheManagerClient
from kv_cache_manager.py_connector.common.logger import logger

QUERY_TYPE = "QT_PREFIX_MATCH"


@dataclass(frozen=True)
class QueryCacheKey:
    """Identity of one location query; different offsets are different queries."""

    req_id: str
    query_type: str
    token_length: int
    computed_blocks: int


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
        # req_id -> {QueryCacheKey: _QueryEntry}: one request may hold entries
        # for several offsets at once (an older answered/in-flight query plus
        # a re-ask at a grown offset).
        self._entries: Dict[str, Dict[QueryCacheKey, _QueryEntry]] = {}
        # req_id -> key of the entry the match hook last returned an answer
        # for; that is the one the allocation consumes.
        self._last_answered: Dict[str, QueryCacheKey] = {}

    def shutdown(self):
        pass

    @staticmethod
    def _key(request, computed_blocks: int) -> QueryCacheKey:
        return QueryCacheKey(
            req_id=request.request_id,
            query_type=QUERY_TYPE,
            token_length=len(request.prompt_token_ids),
            computed_blocks=computed_blocks)

    def _fetch_from_manager(self, request, computed_blocks: int):
        """Run the actual GetCacheLocation call (http thread or inline)."""
        get_request = {
            "trace_id": request.request_id,
            "token_ids": request.prompt_token_ids,
            "instance_id": self._instance_id,
            "query_type": QUERY_TYPE,
            "block_mask": {"offset": computed_blocks},
        }
        logger.debug("get_kvcache_location request: %s", get_request)
        result = self._manager_client.get_cache_location(get_request)
        logger.debug("get_kvcache_location result: %s", result)
        return result["locations"]

    def _query_async(self, request, key: QueryCacheKey) -> None:
        def run():
            try:
                locations = self._fetch_from_manager(request, key.computed_blocks)
            except Exception as e:
                logger.warning("get_cache_location error, request_id: %s, error: %s",
                               request.request_id, e)
                with self._lock:
                    # Drop the entry: the next match hook re-issues the query.
                    per_req = self._entries.get(key.req_id)
                    if per_req is not None:
                        entry = per_req.get(key)
                        if entry is not None and entry.in_flight:
                            per_req.pop(key, None)
                return
            with self._lock:
                per_req = self._entries.get(key.req_id)
                entry = per_req.get(key) if per_req is not None else None
                if entry is None or not entry.in_flight:
                    # Replaced or dropped meanwhile (re-issued / invalidated).
                    return
                entry.locations = locations
                entry.in_flight = False

        self._http_executor.submit(run)

    def get_locations_for_query(self, request, computed_blocks: int) -> Optional[list]:
        """Ask (or re-ask) for the request's external match at this offset.

        Returns the locations when the answer is already cached for this
        exact query key, None while a query is in flight for it (the
        scheduler re-asks next step), and [] when the query answered "no
        hit". An in-flight query for a *different* offset never blocks this
        one. A failed query drops its entry so the next hook call re-issues.
        """
        key = self._key(request, computed_blocks)
        with self._lock:
            per_req = self._entries.setdefault(request.request_id, {})
            entry = per_req.get(key)
            if entry is not None:
                if entry.in_flight:
                    # A query for this exact key is already on the wire;
                    # re-issuing on every scheduler re-ask would fan out one
                    # full getCacheLocation per engine step.
                    return None
                self._last_answered[request.request_id] = key
                return entry.locations
            per_req[key] = _QueryEntry(computed_blocks=computed_blocks)

        if self._async_get_cache_location:
            self._query_async(request, key)
            return None
        try:
            locations = self._fetch_from_manager(request, computed_blocks)
        except Exception as e:
            logger.warning("get_cache_location error, request_id: %s, error: %s",
                           request.request_id, e)
            with self._lock:
                per_req = self._entries.get(request.request_id)
                if per_req is not None:
                    per_req.pop(key, None)
            return []
        with self._lock:
            per_req = self._entries.get(request.request_id)
            cur = per_req.get(key) if per_req is not None else None
            if cur is not None:
                cur.locations = locations
                cur.in_flight = False
            self._last_answered[request.request_id] = key
        return locations

    def store_result(self, req_id: str, locations: list) -> None:
        """Overwrite the cached answer (the match hook clamps it to a
        vLLM-safe prefix before the allocation consumes it). Writes back to
        the entry the hook last returned an answer for."""
        with self._lock:
            key = self._last_answered.get(req_id)
            if key is None:
                return
            per_req = self._entries.get(req_id)
            entry = per_req.get(key) if per_req is not None else None
            if entry is not None:
                entry.locations = locations
                entry.in_flight = False

    def consume_locations(self, req_id: str) -> Optional[Tuple[list, int]]:
        """Pop the answered query the match hook last returned: (locations,
        computed_blocks), or None when nothing is cached (no query was issued
        / still in flight)."""
        with self._lock:
            key = self._last_answered.get(req_id)
            if key is None:
                return None
            self._last_answered.pop(req_id, None)
            per_req = self._entries.get(req_id)
            entry = per_req.pop(key, None) if per_req is not None else None
            if entry is None or entry.in_flight:
                return None
            return entry.locations, entry.computed_blocks

    def invalidate(self, req_id: str) -> None:
        """Drop any cached query for a request that is going away."""
        with self._lock:
            self._entries.pop(req_id, None)
            self._last_answered.pop(req_id, None)
