"""Public API for the standalone KVCM Manager Python client."""

from kv_cache_manager.py_connector.common.manager_client import KvCacheManagerClient

try:
    from kv_cache_manager.py_connector.common._version_info import (
        FULL_VERSION as __version__,
    )
except ImportError:  # Source-tree imports outside Bazel do not generate this module.
    from importlib.metadata import PackageNotFoundError, version

    try:
        __version__ = version("tair-kvcache-manager-client")
    except PackageNotFoundError:
        __version__ = "0.0.0+unknown"

__all__ = ["KvCacheManagerClient", "__version__"]
