# -*- coding: utf-8 -*-
"""Spectrum Service Discovery Client

通过本地 Spectrum 网关接口按 virtual_service_id 获取服务实例列表，
并提供 TTL 缓存和负载均衡能力。

接口（固定）：
    GET http://127.0.0.1:8880/api/v1/discovery/virtual-services/{id}/instances

响应格式（示例）：
{
  "virtual_service_id": "v-ad2d143d",
  "instances": [
    {
      "ip": "172.1.2.10",
      "port": 8080,
      "name": "ds-abdedesd-ad2d-sded",
      "physical_service_id": "abdedesd"
    }
  ]
}
"""

import threading
import time
import random
import requests
from typing import List, Optional

from kv_cache_manager.py_connector.common.logger import logger
from kv_cache_manager.py_connector.common.service_discovery import (
    ServiceDiscovery,
    ServiceEndpoint,
)


SPECTRUM_GATEWAY_BASE_URL = "http://127.0.0.1:8880"
SPECTRUM_INSTANCES_PATH_TEMPLATE = "/api/v1/discovery/virtual-services/{vsid}/instances"


class SpectrumServiceDiscovery(ServiceDiscovery):
    """Spectrum 服务发现客户端

    调用方只需传入 ``virtual_service_id``（如 ``"v-ad2d143d"``），客户端会
    向固定的本地 Spectrum 网关地址发送 HTTP GET 请求，获取实例列表并缓存。

    Args:
        virtual_service_id: Spectrum 虚拟服务 ID
        cache_ttl: 缓存有效期（秒），默认 30 秒
        refresh_timeout: 请求超时时间（秒），默认 5 秒
        auto_refresh: 是否在缓存过期时自动刷新，默认 True
        retry_count: 单次刷新内的额外重试次数（不含首次），默认 0
    """

    def __init__(
        self,
        virtual_service_id: str,
        *,
        cache_ttl: int = 30,
        refresh_timeout: int = 5,
        auto_refresh: bool = True,
        retry_count: int = 0,
    ):
        if not virtual_service_id:
            raise ValueError("virtual_service_id must not be empty")

        self.virtual_service_id = virtual_service_id
        self.cache_ttl = cache_ttl
        self.refresh_timeout = refresh_timeout
        self.auto_refresh = auto_refresh
        self.retry_count = max(0, retry_count)

        self._cache: List[ServiceEndpoint] = []
        self._cache_time: float = 0.0
        self._cache_lock = threading.Lock()

        self._session = requests.Session()
        self._session.headers.update({
            'Accept': 'application/json',
            'Content-Type': 'application/json',
        })

        self.refresh()

    def get_type(self) -> str:
        return "Spectrum"

    @property
    def service_url(self) -> str:
        """构造完整的 Spectrum 实例查询 URL（用于日志和调试）。"""
        path = SPECTRUM_INSTANCES_PATH_TEMPLATE.format(vsid=self.virtual_service_id)
        return SPECTRUM_GATEWAY_BASE_URL + path

    def get_all_endpoints(self) -> List[ServiceEndpoint]:
        """获取所有可用服务节点

        Returns:
            服务端点列表，如果获取失败返回空列表
        """
        with self._cache_lock:
            if self._is_cache_expired() and self.auto_refresh:
                pass
            else:
                return self._cache.copy()

        self.refresh()

        with self._cache_lock:
            return self._cache.copy()

    def get_one_endpoint(self) -> Optional[ServiceEndpoint]:
        """获取单个服务节点（随机负载均衡）

        Returns:
            单个服务端点，如果没有可用节点返回 None
        """
        endpoints = self.get_all_endpoints()
        if not endpoints:
            return None
        return random.choice(endpoints)

    def refresh(self) -> bool:
        """强制刷新缓存

        retry_count > 0 时会做重试；任一尝试成功立刻返回 True。

        Returns:
            刷新是否成功
        """
        total_attempts = self.retry_count + 1
        last_err: Optional[Exception] = None
        for _ in range(total_attempts):
            try:
                endpoints = self._fetch_from_spectrum()
                with self._cache_lock:
                    self._cache = endpoints
                    self._cache_time = time.time()
                logger.info(
                    f"Spectrum service discovery refreshed: {len(endpoints)} endpoints "
                    f"for vsid={self.virtual_service_id}"
                )
                return True
            except Exception as e:
                last_err = e
                continue

        logger.error(
            f"Failed to refresh Spectrum service discovery for "
            f"vsid={self.virtual_service_id} after {total_attempts} attempts: {last_err}"
        )
        return False

    def close(self):
        """关闭客户端，释放资源"""
        self._session.close()

    def _is_cache_expired(self) -> bool:
        return (time.time() - self._cache_time) > self.cache_ttl

    def _fetch_from_spectrum(self) -> List[ServiceEndpoint]:
        """从 Spectrum 网关获取实例列表。"""
        url = self.service_url
        response = self._session.get(url, timeout=self.refresh_timeout)
        response.raise_for_status()

        data = response.json()

        endpoints: List[ServiceEndpoint] = []
        items = data.get('instances', [])
        if not isinstance(items, list):
            logger.warning(
                f"Spectrum response 'instances' is not a list for "
                f"vsid={self.virtual_service_id}"
            )
            return endpoints

        for item in items:
            if not isinstance(item, dict):
                continue
            if 'ip' not in item or 'port' not in item:
                logger.warning(
                    f"Spectrum instance missing ip or port, skipping: {item}"
                )
                continue

            endpoint = ServiceEndpoint(
                ip=item['ip'],
                port=item['port'],
                host=f"{item['ip']}:{item['port']}",
                weight=item.get('weight', 100),
                healthy=True,
            )
            endpoints.append(endpoint)

        if not endpoints:
            logger.warning(
                f"No valid endpoints found in Spectrum response for "
                f"vsid={self.virtual_service_id}"
            )

        return endpoints
