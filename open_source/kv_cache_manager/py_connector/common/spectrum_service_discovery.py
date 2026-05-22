# -*- coding: utf-8 -*-
"""Spectrum Service Discovery - Open Source Placeholder

开源构建下的占位实现。接口形态与 internal 端完全一致，
但运行期所有方法都抛出 NotImplementedError，
以便在没有 Spectrum 网关的开源环境下保持接口一致 + 运行时清晰报错。
"""

from typing import List, Optional

from kv_cache_manager.py_connector.common.service_discovery import (
    ServiceDiscovery,
    ServiceEndpoint,
)


class SpectrumServiceDiscovery(ServiceDiscovery):
    """开源构建下的 Spectrum 服务发现占位实现。"""

    def __init__(
        self,
        virtual_service_id: str,
        *,
        cache_ttl: int = 30,
        refresh_timeout: int = 5,
        auto_refresh: bool = True,
        retry_count: int = 0,
        custom_port: int = 0,
    ):
        raise NotImplementedError(
            "SpectrumServiceDiscovery is not available in open-source build. "
            "This feature requires Spectrum gateway."
        )

    def get_type(self) -> str:
        return "Spectrum"

    @property
    def service_url(self) -> str:
        raise NotImplementedError(
            "SpectrumServiceDiscovery is not available in open-source build."
        )

    def get_all_endpoints(self) -> List[ServiceEndpoint]:
        raise NotImplementedError(
            "SpectrumServiceDiscovery is not available in open-source build."
        )

    def get_one_endpoint(self) -> Optional[ServiceEndpoint]:
        raise NotImplementedError(
            "SpectrumServiceDiscovery is not available in open-source build."
        )

    def refresh(self) -> bool:
        raise NotImplementedError(
            "SpectrumServiceDiscovery is not available in open-source build."
        )

    def close(self):
        raise NotImplementedError(
            "SpectrumServiceDiscovery is not available in open-source build."
        )
