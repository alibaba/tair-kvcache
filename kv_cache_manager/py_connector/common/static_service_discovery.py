# -*- coding: utf-8 -*-
"""Static Service Discovery：固定 ip:port 列表的本地实现。

适用场景：
    - 测试 / 单机部署直接写死 endpoint 列表
    - 不需要动态订阅但又想统一走 ServiceDiscovery 抽象的下游

URL 形式（由工厂解析后注入）::

    static://11.22.33.44:8080,33.55.66.77:8080
"""

import threading
from typing import List, Optional, Sequence

from kv_cache_manager.py_connector.common.logger import logger
from kv_cache_manager.py_connector.common.service_discovery import (
    ServiceDiscovery,
    ServiceEndpoint,
)


def parse_host_port_list(host_list: str) -> List[ServiceEndpoint]:
    """把 "ip1:port1,ip2:port2" 解析成 ServiceEndpoint 列表。

    任一段非法（缺 port / port 非数 / 空 host）即抛 ValueError。
    """
    if not host_list:
        raise ValueError("host_list is empty")
    endpoints: List[ServiceEndpoint] = []
    for token in host_list.split(','):
        token = token.strip()
        if not token:
            continue
        if ':' not in token:
            raise ValueError(f"static endpoint missing host:port format, token={token!r}")
        host, _, port_str = token.partition(':')
        if not host or not port_str:
            raise ValueError(f"static endpoint missing host or port, token={token!r}")
        if not port_str.isdigit():
            raise ValueError(f"static endpoint port not numeric, token={token!r}")
        port = int(port_str)
        if port <= 0 or port > 65535:
            raise ValueError(f"static endpoint port out of range, token={token!r}")
        endpoints.append(ServiceEndpoint(ip=host, port=port, host=f"{host}:{port}"))
    if not endpoints:
        raise ValueError(f"static endpoint list is empty after parsing, raw={host_list!r}")
    return endpoints


class StaticServiceDiscovery(ServiceDiscovery):
    """静态 ip:port 列表的 ServiceDiscovery 实现。

    Args:
        host_list: "ip1:port1,ip2:port2,..." 形式的字符串
        endpoints: 已解析好的端点列表；与 host_list 二选一
    """

    def __init__(
        self,
        host_list: Optional[str] = None,
        *,
        endpoints: Optional[Sequence[ServiceEndpoint]] = None,
    ):
        if endpoints is not None:
            parsed: List[ServiceEndpoint] = list(endpoints)
        elif host_list is not None:
            parsed = parse_host_port_list(host_list)
        else:
            raise ValueError("must provide either host_list or endpoints")
        if not parsed:
            raise ValueError("StaticServiceDiscovery init with empty endpoints")
        self._endpoints: List[ServiceEndpoint] = parsed
        self._rr_index = 0
        self._rr_lock = threading.Lock()

    def get_type(self) -> str:
        return "Static"

    def get_all_endpoints(self) -> List[ServiceEndpoint]:
        return list(self._endpoints)

    def get_one_endpoint(self) -> Optional[ServiceEndpoint]:
        if not self._endpoints:
            return None
        with self._rr_lock:
            ep = self._endpoints[self._rr_index % len(self._endpoints)]
            self._rr_index = (self._rr_index + 1) % len(self._endpoints)
        return ep

    def refresh(self) -> bool:
        return bool(self._endpoints)

    def close(self) -> None:
        # 无外部资源
        return None
