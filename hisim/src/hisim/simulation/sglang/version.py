from __future__ import annotations

from typing import Callable, Optional, Dict
from packaging.version import Version, InvalidVersion

from sglang.version import __version__ as sglang_version
from hisim.utils.logger import get_logger

logger = get_logger()

# Verified compatible versions.
COMPATIBLE_VERSIONS = [
    "0.5.6",
    "0.5.6.post1",
    "0.5.6.post2",
    "0.5.7",
    "0.5.8",
    "0.5.8.post1",
    "0.5.9",
]


class VersionDispatcher:
    def __init__(self) -> None:
        # name -> {Version: Callable}
        self._methods: Dict[str, Dict[Version, Callable]] = {}

    @staticmethod
    def _parse_version(v: str) -> Version:
        try:
            return Version(v)
        except InvalidVersion as e:
            raise ValueError(f"Invalid version string: {v!r}") from e

    def register_method(self, name: str, versions: list[str], method: Callable) -> None:
        bucket = self._methods.setdefault(name, {})
        for v in versions:
            pv = self._parse_version(v)
            if pv in bucket:
                raise ValueError(f"The method {name} of {v} had been registered yet.")
            bucket[pv] = method

    def get_compat_method(self, name: str, version: Optional[str] = None) -> Callable:
        if name not in self._methods or not self._methods[name]:
            raise ValueError(f"The method {name} had not been registered yet.")

        bucket = self._methods[name]
        req_v = self._parse_version(version or sglang_version)

        # 1) Exact match
        if req_v in bucket:
            return bucket[req_v]

        # 2) Fallback to newest implementation
        newest_v = max(bucket.keys())
        logger.debug(
            "Method %s for version %s is not registered, fallback to newest %s",
            name,
            req_v,
            newest_v,
        )
        return bucket[newest_v]
