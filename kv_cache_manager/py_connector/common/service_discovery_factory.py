# -*- coding: utf-8 -*-
"""按 URL 形式的服务发现配置创建并初始化对应的 :class:`ServiceDiscovery` 实例。

支持的 URL（与 C++ 端 ``CreateServiceDiscovery`` 完全一致）：

    - ``vipserver://<domain>[?timeout=<sec>]``
        Python 端暂未提供原生 VIPServer 实现，命中此分支会打 ERROR 返回 None。
    - ``spectrum://<virtual_service_id>[?cache_time=<sec>&retry_time=<n>&timeout=<ms>&port=<custom_port>]``
    - ``static://<ip:port>[,<ip:port>...]``
    - 空字符串：返回 None，调用方按"不使用服务发现"语义降级。

设计目标：
    1. 调用方不需要 import 任何具体实现类。
    2. 新增类型时只需扩展 URL scheme 分支，业务代码零改动。
    3. 与 C++ 端语义完全一致，便于配置在前后端复用同一份字符串。
"""

from typing import Dict, Optional, Tuple

from kv_cache_manager.py_connector.common.logger import logger
from kv_cache_manager.py_connector.common.service_discovery import (
    ServiceDiscovery,
)


_SCHEME_VIPSERVER = "vipserver"
_SCHEME_SPECTRUM = "spectrum"
_SCHEME_STATIC = "static"


def _parse_url(url: str) -> Optional[Tuple[str, str, Dict[str, str]]]:
    """解析 ``<scheme>://<body>[?k=v(&k=v)*]``，返回 (scheme, body, params) 或 None。

    之所以不用 urllib.parse，是因为 ``static://ip:port,ip:port`` 含逗号会被
    urllib 错误地塞进 hostname 校验流程，自己实现更稳。
    """
    if not url:
        return None
    sep = url.find("://")
    if sep <= 0:
        logger.error(f"invalid service discovery url, missing scheme: {url!r}")
        return None
    scheme = url[:sep]
    rest = url[sep + 3:]
    if not rest:
        logger.error(f"invalid service discovery url, empty body: {url!r}")
        return None
    body, sep_pos, query = rest.partition("?")
    params: Dict[str, str] = {}
    if sep_pos:
        for kv in query.split("&"):
            if not kv:
                continue
            k, eq, v = kv.partition("=")
            if not eq:
                logger.error(f"invalid service discovery url, missing '=' in {kv!r}")
                continue
            params[k] = v if eq else ""
    if not body:
        logger.error(f"invalid service discovery url, empty body: {url!r}")
        return None
    return scheme, body, params


def _get_int_param(params: Dict[str, str], key: str, default: int) -> int:
    raw = params.get(key)
    if raw is None or raw == "":
        return default
    try:
        return int(raw)
    except (TypeError, ValueError):
        return default


def create_service_discovery(url: str) -> Optional[ServiceDiscovery]:
    """按 URL 创建并初始化对应的 ServiceDiscovery 实例。

    返回值约定：
        - 空字符串 / 解析失败：返回 None
        - 子类构造失败（如非法参数）：打 ERROR 并返回 None
        - 未知 scheme：打 ERROR 并返回 None
        - 仅 ``static://`` 与 ``spectrum://`` 在 py_connector 中有原生实现；
          ``vipserver://`` 始终返回 None 并打 ERROR。
    """
    if not url:
        return None
    parsed = _parse_url(url)
    if parsed is None:
        return None
    scheme, body, params = parsed

    if scheme == _SCHEME_STATIC:
        from kv_cache_manager.py_connector.common.static_service_discovery import (
            StaticServiceDiscovery,
        )
        try:
            return StaticServiceDiscovery(body)
        except Exception as e:
            logger.error(f"failed to create StaticServiceDiscovery for url={url!r}: {e}")
            return None

    if scheme == _SCHEME_SPECTRUM:
        # Spectrum 是内部实现，从 stub_source 导入（指向 internal_source）
        # 如果导入失败，说明是开源环境，不支持 Spectrum
        try:
            from stub_source.kv_cache_manager.py_connector.common.spectrum_service_discovery import (
                SpectrumServiceDiscovery,
            )
        except ImportError:
            logger.error(
                f"SpectrumServiceDiscovery not available (requires internal build), url={url!r}"
            )
            return None
        
        cache_ttl = _get_int_param(params, "cache_time", 30)
        # URL 里 timeout 单位是毫秒（与 C++ 端对齐），Python SDK 用秒，这里换算。
        timeout_ms = _get_int_param(params, "timeout", 0)
        refresh_timeout = max(1, timeout_ms // 1000) if timeout_ms > 0 else 5
        retry_count = _get_int_param(params, "retry_time", 0)
        # port 表示自定义端口；> 0 时强制覆盖 Spectrum 网关返回的端口。
        custom_port = _get_int_param(params, "port", 0)
        try:
            return SpectrumServiceDiscovery(
                body,
                cache_ttl=cache_ttl,
                refresh_timeout=refresh_timeout,
                retry_count=retry_count,
                custom_port=custom_port,
            )
        except NotImplementedError as e:
            # 开源占位实现会抛出 NotImplementedError
            logger.error(f"Spectrum service discovery not available: {e}")
            return None
        except Exception as e:
            logger.error(f"failed to create SpectrumServiceDiscovery for url={url!r}: {e}")
            return None

    if scheme == _SCHEME_VIPSERVER:
        logger.error(
            f"VIPServer service discovery is not implemented in py_connector, url={url!r}"
        )
        return None

    logger.error(f"unsupported service discovery scheme={scheme!r}, url={url!r}")
    return None
