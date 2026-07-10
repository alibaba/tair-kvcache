from __future__ import annotations

from subscriber.engine import vllm as _vllm
from subscriber.engine.base import AbstractEngineAdapter

__all__ = ["AbstractEngineAdapter"]

# Reference the module so linters keep the side-effect import.
_ = _vllm
