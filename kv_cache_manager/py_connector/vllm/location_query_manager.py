"""Manager-side location queries for the external-match hook.

``get_num_new_matched_tokens`` must never block the scheduler loop, so
queries run on the http executor and the answer is cached per request
until consumed. The cache is request-lifecycle scoped -- produced by the
match hook, consumed by ``update_state_after_alloc`` (which turns the
answer into a LoadRequest) and invalidated when the request retires --
plus the expiry below, because a queued request can sit between the two
hooks for arbitrarily long.

**One slot per request, superseded by the newest ask.** The scheduler
asks twice per request's life: ``get_num_new_matched_tokens`` (with the
offset) and ``update_state_after_alloc`` (without it). vLLM's contract
fills the gap: the alloc hook always consumes the answer of the *last*
match hook that returned a hit -- it passes the matched token count back
verbatim. So the cache needs no per-offset addressing; a new ask at a
different offset is a *different* query that supersedes the old one, and
only the newest ask's answer may ever be consumed.

**Answers expire.** The manager may delete the blocks a cached answer
points to at any time after it answered, so a hit is served only while
younger than ``max_answer_age_s``, measured from query issue. An
expired hit is a miss: the slot is superseded and the query re-issued.
In-flight queries are never expired -- the client's request timeout
bounds them.

**Object identity is the version.** A superseded (or invalidated /
consumed) query must not write its late answer into the new slot. Each
ask captures the slot object it created; the async callback compares
identity before writing. A monotonic version number is kept for logs
only -- correctness never depends on it.
"""

import threading
import time
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
class _Query:
    """The request's one active query. ``locations is None`` = in flight."""

    key: QueryCacheKey
    version: int = 0
    locations: list = None          # None until the manager answered
    ask_time: float = 0.0           # monotonic clock, at query issue


class LocationQueryManager:
    """Per-request location query cache: produce on match, consume on alloc.

    One ``_Query`` slot per request. Re-asking the slot's own offset
    deduplicates (in flight -> wait, answered -> serve while fresh);
    asking a different offset *supersedes* the slot. An answered hit
    older than ``max_answer_age_s`` is a miss: the slot is superseded
    and the query re-issued.
    """

    def __init__(self, manager_client: KvCacheManagerClient, http_executor,
                 instance_id: str, async_get_cache_location: bool,
                 max_answer_age_s: float = 1.0):
        self._manager_client = manager_client
        self._http_executor = http_executor
        self._instance_id = instance_id
        self._async_get_cache_location = async_get_cache_location
        self._max_answer_age_s = max_answer_age_s
        self._lock = threading.Lock()
        # req_id -> the request's one active query slot.
        self._queries: Dict[str, _Query] = {}
        # Hits refused for age and re-fetched.
        self.stale_supersede_count = 0

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

    def _query_async(self, request, key: QueryCacheKey, q: _Query) -> None:
        def run():
            try:
                locations = self._fetch_from_manager(request, key.computed_blocks)
            except Exception as e:
                logger.warning("get_cache_location error, request_id: %s, error: %s",
                               request.request_id, e)
                with self._lock:
                    # Drop the slot: the next match hook re-issues the query.
                    if self._queries.get(request.request_id) is q:
                        self._queries.pop(request.request_id, None)
                return
            with self._lock:
                if self._queries.get(request.request_id) is q:
                    q.locations = locations
                else:
                    # Superseded / consumed / invalidated meanwhile: the
                    # answer of a dead ask must never write into the new
                    # slot (late answers only win against older asks, never
                    # against newer ones).
                    logger.debug("get_cache_location answer dropped: request %s "
                                 "query v%d was superseded",
                                 request.request_id, q.version)

        self._http_executor.submit(run)

    def get_locations_for_query(self, request, computed_blocks: int) -> Optional[list]:
        """Ask (or re-ask) for the request's external match at this offset.

        Returns the locations when a fresh answer is already cached for
        this exact query key, None while a query is in flight for it (the
        scheduler re-asks next step), and [] when the query answered "no
        hit". An ask at a different offset -- or one whose cached answer
        has expired -- supersedes the slot: a new query starts
        immediately, and the old answer can no longer be consumed. A
        failed query drops its slot so the next hook call re-issues.
        """
        req_id = request.request_id
        key = self._key(request, computed_blocks)
        with self._lock:
            q = self._queries.get(req_id)
            if q is not None and q.key == key:
                if q.locations is None:
                    # Same query still in flight: dedupe (the scheduler
                    # re-asks every step; the client timeout bounds it).
                    return None
                age = time.monotonic() - q.ask_time
                if age <= self._max_answer_age_s:
                    return q.locations
                # Expired: a miss. Old answers are never served.
                self.stale_supersede_count += 1
                logger.info("req:%s location answer expired after %.3fs "
                            "(max %.3fs); re-querying", req_id, age,
                            self._max_answer_age_s)
            # First ask, a different offset, or an expired answer: the
            # newest ask supersedes.
            q = _Query(key=key,
                       version=q.version + 1 if q is not None else 0,
                       ask_time=time.monotonic())
            self._queries[req_id] = q

        if self._async_get_cache_location:
            self._query_async(request, key, q)
            return None
        try:
            locations = self._fetch_from_manager(request, computed_blocks)
        except Exception as e:
            logger.warning("get_cache_location error, request_id: %s, error: %s",
                           req_id, e)
            with self._lock:
                if self._queries.get(req_id) is q:
                    self._queries.pop(req_id, None)
            return []
        with self._lock:
            if self._queries.get(req_id) is q:
                q.locations = locations
        return locations

    def store_result(self, req_id: str, locations: list) -> None:
        """Overwrite the cached answer (the match hook clamps it to a
        vLLM-safe prefix before the allocation consumes it). The slot is
        necessarily the query the hook just got its answer from: the hook
        runs on the scheduler thread and only it supersedes slots."""
        with self._lock:
            q = self._queries.get(req_id)
            if q is not None:
                q.locations = locations

    def consume_locations(self, req_id: str) -> Optional[Tuple[list, int]]:
        """Pop the request's answered query: (locations, computed_blocks),
        or None when nothing is cached (no query was issued / still in
        flight / already consumed)."""
        with self._lock:
            q = self._queries.pop(req_id, None)
            if q is None or q.locations is None:
                return None
            return q.locations, q.key.computed_blocks

    def invalidate(self, req_id: str) -> None:
        """Drop the request's query: it is going away, and any answer still
        in flight must not write into a future slot."""
        with self._lock:
            self._queries.pop(req_id, None)
