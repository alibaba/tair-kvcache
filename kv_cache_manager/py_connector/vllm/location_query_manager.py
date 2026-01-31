"""缓存管理模块，包含TairKvCacheConnector的本地缓存管理逻辑"""

import enum
import time
import threading
from dataclasses import dataclass
from typing import Any, Tuple, Dict
from kv_cache_manager.py_connector.common.manager_client import KvCacheManagerClient
from kv_cache_manager.py_connector.common.logger import logger


@dataclass(frozen=True)
class QueryCacheKey:
    req_id: str
    query_type: str
    token_length: int
    computed_manager_block_size: int


class QueryCacheStatus(enum.Enum):
    RUNNING = 0
    FINISHED = 1
    NOT_FOUND = 2


@dataclass
class QueryCacheValue:
    locations: list[Any]
    query_time: float
    is_done: bool = False


class LocationQueryManager:
    """Location请求管理器，负责Location请求和Location信息缓存的管理"""

    def __init__(self, manager_client: KvCacheManagerClient, http_executor, instance_id: str,
                 async_get_cache_location: bool):
        """
        初始化缓存管理器
        
        Args:
            manager_client: Manager客户端
            http_executor: HTTP执行器
            instance_id: 实例ID
            async_get_cache_location: 是否启用异步GetCacheLocation
        """
        self._manager_client = manager_client
        self._http_executor = http_executor
        self._instance_id = instance_id
        self._async_get_cache_location = async_get_cache_location

        self._local_query_cache_lock = threading.Lock()
        self._local_query_cache: Dict[QueryCacheKey, QueryCacheValue] = {}

        self._cleanup_stop_event = threading.Event()
        self._cleanup_thread = threading.Thread(target=self._cleanup_expired_local_cache, daemon=True)
        self._cleanup_thread.start()

    def shutdown(self):
        """关闭缓存管理器"""
        self._cleanup_stop_event.set()
        if self._cleanup_thread.is_alive():
            self._cleanup_thread.join(timeout=2.0)

    def _cleanup_expired_local_cache(self):
        """清理过期的本地缓存条目"""
        while not self._cleanup_stop_event.wait(1.0):
            try:
                current_time = time.time()
                expired_keys = []

                with self._local_query_cache_lock:
                    for key, value in self._local_query_cache.items():
                        # TODO: editable ttl
                        if current_time - value.query_time > 1:
                            expired_keys.append(key)
                    for key in expired_keys:
                        self._local_query_cache.pop(key)

                if expired_keys:
                    logger.debug("Cleaned up %d expired query cache entries", len(expired_keys))
            except Exception as e:
                logger.warning("Error during query cache cleanup: %s", e)

    def _get_cache_from_manager(self, request, computed_manager_block_size: int, query_key: QueryCacheKey):
        """
        从管理器获取缓存位置
        
        Args:
            request: 请求对象
            computed_manager_block_size: 本地已命中的block数量
            query_key: 查询键
        """
        try:
            get_request = {
                "trace_id": request.request_id,
                "token_ids": request.prompt_token_ids,
                "instance_id": self._instance_id,
                "query_type": "QT_PREFIX_MATCH",
                "block_mask": {
                    "offset": computed_manager_block_size
                }
            }
            logger.debug("get_kvcache_location request: %s", get_request)
            result = self._manager_client.get_cache_location(get_request)
            logger.debug("get_kvcache_location result: %s", result)
            need_load_locations = result["locations"]
            with self._local_query_cache_lock:
                if query_key not in self._local_query_cache:
                    logger.warning("_local_query_cache not found %s when request finished", request.request_id)
                    return
                self._local_query_cache[query_key].locations = need_load_locations
                self._local_query_cache[query_key].is_done = True
        except Exception as e:
            logger.warning("get_cache_location error, request_id: %s, error: %s", request.request_id, e)
            with self._local_query_cache_lock:
                if query_key in self._local_query_cache:
                    self._local_query_cache.pop(query_key)

    def _try_get_locations_from_local_cache(self, query_key: QueryCacheKey) -> Tuple[QueryCacheStatus, list]:
        """
        尝试从本地缓存获取位置
        
        Args:
            query_key: 查询键
            
        Returns:
            (状态, 位置列表)
        """
        with self._local_query_cache_lock:
            if query_key not in self._local_query_cache:
                return QueryCacheStatus.NOT_FOUND, []
            query_value = self._local_query_cache[query_key]
            # TODO: editable ttl
            if time.time() - query_value.query_time > 1:
                # cache timeout
                self._local_query_cache.pop(query_key)
                return QueryCacheStatus.NOT_FOUND, []
            if query_value.is_done:
                return QueryCacheStatus.FINISHED, query_value.locations
            else:
                return QueryCacheStatus.RUNNING, []

    def get_locations_for_query(self, request, computed_manager_block_size: int) -> Tuple[bool, list]:
        """
        获取查询的位置信息
        
        Args:
            request: 请求对象
            computed_manager_block_size: 本地已命中的block数量
            
        Returns:
            (查询是否完成, 位置列表)
        """
        # TODO: async get_kvcache_location
        query_key = QueryCacheKey(
            req_id=request.request_id,
            query_type="QT_PREFIX_MATCH",
            token_length=len(request.prompt_token_ids),
            computed_manager_block_size=computed_manager_block_size)
        try:
            status, locations = self._try_get_locations_from_local_cache(query_key)
            if status == QueryCacheStatus.RUNNING:
                return False, []
            elif status == QueryCacheStatus.FINISHED:
                return True, locations

            # status == QueryCacheStatus.NOT_FOUND
            # insert new cache
            with self._local_query_cache_lock:
                self._local_query_cache[query_key] = QueryCacheValue(
                    locations=[], query_time=time.time(), is_done=False)
            if self._async_get_cache_location:
                future = self._http_executor.submit(self._get_cache_from_manager, request, computed_manager_block_size,
                                                    query_key)
                return False, []
            else:
                self._get_cache_from_manager(request, computed_manager_block_size, query_key)
                # only sync call need check again
                status, locations = self._try_get_locations_from_local_cache(query_key)
                if status == QueryCacheStatus.NOT_FOUND:
                    # do_get_cache_locations error, bypass load
                    return True, []
                elif status == QueryCacheStatus.RUNNING:
                    return False, []
                elif status == QueryCacheStatus.FINISHED:
                    return True, locations
                else:
                    logger.warning("unknown local cache query status: %s", status)
                    return True, []
        except Exception as e:
            logger.warning("get_locations_for_query error, request_id: %s, error: %s", request.request_id, e)
            return True, []
