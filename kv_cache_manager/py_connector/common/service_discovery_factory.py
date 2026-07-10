# -*- coding: utf-8 -*-
"""Create service-discovery instances from provider-neutral URLs.

``static://host:port[,host:port...]`` is built in. Other schemes can be
registered by the active build through :func:`register_service_discovery_provider`.
"""

import importlib
import threading
from typing import Callable, Dict, Optional, Tuple

from kv_cache_manager.py_connector.common.logger import logger
from kv_cache_manager.py_connector.common.service_discovery import ServiceDiscovery


ServiceDiscoveryProvider = Callable[[str, Dict[str, str]], Optional[ServiceDiscovery]]

_SCHEME_STATIC = "static"
_PROVIDER_EXTENSION_MODULE = (
    "stub_source.kv_cache_manager.py_connector.common."
    "service_discovery_factory_extension"
)
_provider_factories: Dict[str, ServiceDiscoveryProvider] = {}
_registry_lock = threading.RLock()
_extensions_loaded = False


def _parse_url(url: str) -> Optional[Tuple[str, str, Dict[str, str]]]:
    """Parse ``<scheme>://<body>[?k=v(&k=v)*]``."""
    if not url:
        return None
    sep = url.find("://")
    if sep <= 0:
        logger.error(f"invalid service discovery url, missing scheme: {url!r}")
        return None
    scheme = url[:sep].lower()
    rest = url[sep + 3:]
    if not rest:
        logger.error(f"invalid service discovery url, empty body: {url!r}")
        return None
    body, has_query, query = rest.partition("?")
    if not body:
        logger.error(f"invalid service discovery url, empty body: {url!r}")
        return None

    params: Dict[str, str] = {}
    if has_query:
        for item in query.split("&"):
            if not item:
                continue
            key, separator, value = item.partition("=")
            if not separator or not key:
                logger.error(
                    f"invalid service discovery url query parameter: {item!r}"
                )
                return None
            params[key] = value
    return scheme, body, params


def register_service_discovery_provider(
    scheme: str,
    provider: ServiceDiscoveryProvider,
    *,
    replace: bool = False,
) -> None:
    """Register a provider factory for one URL scheme.

    Registration is thread-safe. Duplicate registrations are rejected unless
    ``replace=True`` is explicitly requested.
    """
    normalized_scheme = scheme.strip().lower()
    if not normalized_scheme or "://" in normalized_scheme:
        raise ValueError(f"invalid service discovery scheme: {scheme!r}")
    if not callable(provider):
        raise TypeError("provider must be callable")
    with _registry_lock:
        if normalized_scheme in _provider_factories and not replace:
            raise ValueError(
                f"service discovery provider already registered: {normalized_scheme}"
            )
        _provider_factories[normalized_scheme] = provider


def _load_provider_extensions() -> None:
    """Load the build-selected provider registration module exactly once."""
    global _extensions_loaded
    with _registry_lock:
        if _extensions_loaded:
            return
        try:
            extension = importlib.import_module(_PROVIDER_EXTENSION_MODULE)
            register = getattr(extension, "register_service_discovery_providers", None)
            if register is not None:
                register(register_service_discovery_provider)
        except ImportError as error:
            logger.debug("service discovery extension is unavailable: %s", error)
        except Exception as error:
            logger.error("failed to load service discovery extension: %s", error)
        finally:
            _extensions_loaded = True


def create_service_discovery(url: str) -> Optional[ServiceDiscovery]:
    """Create an initialized discovery instance for ``url``.

    Empty, malformed, unsupported, or provider-rejected URLs return ``None``.
    """
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
        except Exception as error:
            logger.error(
                f"failed to create static service discovery for url={url!r}: {error}"
            )
            return None

    _load_provider_extensions()
    with _registry_lock:
        provider = _provider_factories.get(scheme)
    if provider is None:
        logger.error(
            f"unsupported service discovery scheme={scheme!r}, url={url!r}"
        )
        return None

    try:
        return provider(body, dict(params))
    except Exception as error:
        logger.error(
            f"failed to create service discovery for scheme={scheme!r}, "
            f"url={url!r}: {error}"
        )
        return None
