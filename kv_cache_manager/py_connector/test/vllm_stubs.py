"""Shared test stubs: make ``v1_connector`` importable without vLLM/CUDA/pybind.

``v1_connector`` imports vLLM, the compiled ``kvcm_py_client`` and several
third-party runtime deps (torch, triton, orjson, zmq, requests) at module
level. For pure-logic unit tests we register lightweight stand-ins in
``sys.modules`` *before* the first import, then build connector instances via
``__new__`` with only the attributes the code under test reads. No production
module is modified; real modules are preferred whenever they are importable
(e.g. on a dev machine with a full vLLM venv).
"""

import importlib.util
import json
import sys
import types
from typing import Optional
from unittest.mock import MagicMock


def _module(name: str) -> types.ModuleType:
    mod = sys.modules.get(name)
    if mod is None:
        mod = types.ModuleType(name)
        sys.modules[name] = mod
    return mod


def _importable(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):
        return False


def _stub_third_party():
    """Register stand-ins for third-party deps missing from the environment
    (the open-source CI runs these tests without torch/triton/orjson/zmq/
    requests installed). Real modules always win."""
    # Pure-attribute deps: a MagicMock module is enough because the pure-logic
    # tests never execute tensor/socket/http work at module import time.
    for name in ("torch", "triton", "triton.language", "zmq", "requests"):
        if name not in sys.modules and not _importable(name):
            sys.modules[name] = MagicMock(__name__=name)

    # orjson is used functionally (CoordinateMsgSerializer round trips), so
    # the stand-in must actually (de)serialize; stdlib json handles the
    # dataclass payloads via __dict__.
    if "orjson" not in sys.modules and not _importable("orjson"):
        orjson = _module("orjson")
        orjson.dumps = lambda obj: json.dumps(
            obj, default=lambda o: o.__dict__).encode()
        orjson.loads = json.loads


def _install_stubs():
    _stub_third_party()
    existing = sys.modules.get("vllm")
    if existing is not None:
        # Either our stub is already in place or the real vLLM is importable;
        # in both cases the connector import will succeed as-is.
        return

    # ---- kv_cache_manager.client.pybind (compiled extension) ----
    pybind = _module("kv_cache_manager.client.pybind")
    kvcm_py_client = MagicMock()
    kvcm_py_client.ClientErrorCode.ER_OK = 0
    pybind.kvcm_py_client = kvcm_py_client

    # ---- kv_cache_manager.py_connector.common._version_info (generated) ----
    version = _module("kv_cache_manager.py_connector.common._version_info")
    version.FULL_VERSION = "0.0.0-test"
    version.GIT_COMMIT = "test"
    version.BUILD_TIME = "test"

    # ---- vllm ----
    vllm = _module("vllm")
    vllm._kvcm_test_stub = True

    config = _module("vllm.config")
    config.VllmConfig = MagicMock
    vllm.config = config

    distributed = _module("vllm.distributed")
    distributed.get_tensor_model_parallel_rank = lambda: 0
    vllm.distributed = distributed
    _module("vllm.distributed.kv_transfer")
    _module("vllm.distributed.kv_transfer.kv_connector")
    _module("vllm.distributed.kv_transfer.kv_connector.v1")
    base = _module("vllm.distributed.kv_transfer.kv_connector.v1.base")

    class KVConnectorRole:
        SCHEDULER = 0
        WORKER = 1

    class KVConnectorMetadata:
        pass

    class KVConnectorBase_V1:
        def __init__(self, vllm_config, role, kv_cache_config=None):
            self._connector_metadata = None

        def _get_connector_metadata(self):
            return self._connector_metadata

    class SupportsHMA:
        pass

    base.KVConnectorBase_V1 = KVConnectorBase_V1
    base.KVConnectorMetadata = KVConnectorMetadata
    base.KVConnectorRole = KVConnectorRole
    base.SupportsHMA = SupportsHMA

    utils = _module("vllm.utils")
    torch_utils = _module("vllm.utils.torch_utils")
    torch_utils.get_kv_cache_torch_dtype = MagicMock()
    network_utils = _module("vllm.utils.network_utils")
    network_utils.get_ip = lambda: "127.0.0.1"
    utils.torch_utils = torch_utils
    utils.network_utils = network_utils

    v1 = _module("vllm.v1")
    kv_cache_interface = _module("vllm.v1.kv_cache_interface")

    class FullAttentionSpec:
        def __init__(self, block_size, page_size_bytes, page_size_padded=None):
            self.block_size = block_size
            self.page_size_padded = page_size_padded
            self.real_page_size_bytes = page_size_bytes
            # Mirror vLLM's AttentionSpec: page_size_bytes returns the padded
            # size when padding is set.
            self.page_size_bytes = (page_size_padded if page_size_padded
                                    is not None else page_size_bytes)

    class MambaSpec:
        def __init__(self, block_size, page_size_bytes):
            self.block_size = block_size
            self.page_size_bytes = page_size_bytes

    kv_cache_interface.FullAttentionSpec = FullAttentionSpec
    kv_cache_interface.MambaSpec = MambaSpec

    _module("vllm.v1.core")
    sched = _module("vllm.v1.core.sched")
    output = _module("vllm.v1.core.sched.output")
    output.SchedulerOutput = MagicMock
    sched.output = output

    outputs = _module("vllm.v1.outputs")
    outputs.KVConnectorOutput = MagicMock
    v1.kv_cache_interface = kv_cache_interface
    v1.outputs = outputs


_install_stubs()

# Import after stubs are in place.
from kv_cache_manager.py_connector.vllm.v1_connector import (  # noqa: E402
    TairKvCacheConnector, GroupMeta, ReqState)


def make_connector(manager_block_size: int = 16,
                   vllm_block_size: Optional[int] = None,
                   num_groups: int = 1) -> TairKvCacheConnector:
    """Build a bare TairKvCacheConnector (no __init__) with the minimal state
    used by the pure translation / scheduler-side logic under test."""
    conn = TairKvCacheConnector.__new__(TairKvCacheConnector)
    conn._manager_block_size = manager_block_size
    conn._vllm_block_size = vllm_block_size or manager_block_size
    conn._num_groups = num_groups
    conn._group_metas = [
        GroupMeta(group_idx=i, is_attention=True, layer_names=[f"l{i}"],
                  block_size=conn._vllm_block_size, per_block_bytes=0)
        for i in range(num_groups)
    ]
    return conn
