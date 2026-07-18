from __future__ import annotations

from subscriber.engine import rtp_llm as _rtp_llm
from subscriber.engine import vllm as _vllm
from subscriber.engine.base import AbstractEngineAdapter

__all__ = ["AbstractEngineAdapter"]

# Keep adapter modules referenced after their registration side effects.
_REGISTERED_ADAPTER_MODULES = (_vllm, _rtp_llm)
