# -*- coding: utf-8 -*-
"""Service Discovery 抽象基类与统一数据模型。

各种服务发现实现（VIPServer / Spectrum / Static / 未来扩展类型）都遵循这个接口，
业务调用方拿到 ``ServiceDiscovery`` 后只关心端点列表，不关心底层实现。

新增类型时只需：
    1. 实现 ``ServiceDiscovery`` 子类
    2. 在 ``service_discovery_factory`` 中扩展 URL scheme 分支
    3. 配置层把对应 URL（如 ``mytype://...``）传过来即可
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional


@dataclass
class ServiceEndpoint:
    """统一的服务端点信息。"""
    ip: str
    port: int
    host: str  # f"{ip}:{port}"
    weight: int = 100
    healthy: bool = True


class ServiceDiscovery(ABC):
    """服务发现抽象基类。

    所有具体实现都必须暴露这套接口；调用方应通过本基类引用而不是具体子类，
    便于在不同发现机制间切换。
    """

    @abstractmethod
    def get_all_endpoints(self) -> List[ServiceEndpoint]:
        """获取所有可用端点；失败时返回空列表。"""

    @abstractmethod
    def get_one_endpoint(self) -> Optional[ServiceEndpoint]:
        """按实现自定的负载均衡策略返回单个端点；无可用端点时返回 None。"""

    @abstractmethod
    def refresh(self) -> bool:
        """强制刷新底层数据；返回是否成功。"""

    @abstractmethod
    def get_type(self) -> str:
        """返回实现类型名（如 "VIPServer"、"Spectrum"），主要用于日志。"""

    def close(self) -> None:
        """释放底层资源；默认无操作，子类按需实现。"""

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False
