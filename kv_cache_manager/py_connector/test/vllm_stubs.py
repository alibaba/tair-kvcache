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
from kv_cache_manager.py_connector.vllm.vllm_common import (  # noqa: E402
    GroupMeta, ReqState)
from kv_cache_manager.py_connector.vllm.scheduler_core import SchedulerCore  # noqa: E402
from kv_cache_manager.py_connector.vllm.worker_core import WorkerCore  # noqa: E402


def _make_group_metas(num_groups: int, num_state_groups: int,
                      block_size: int) -> list:
    """Attention groups first, then mamba-style state groups; group_idx is the
    vLLM group index (what block tables are indexed by)."""
    return [
        GroupMeta(group_idx=i, is_attention=True, layer_names=[f"l{i}"],
                  block_size=block_size, per_block_bytes=0)
        for i in range(num_groups)
    ] + [
        GroupMeta(group_idx=num_groups + i, is_attention=False,
                  layer_names=[f"m{i}"], block_size=block_size,
                  per_block_bytes=0)
        for i in range(num_state_groups)
    ]


def make_connector(manager_block_size: int = 16,
                   vllm_block_size: Optional[int] = None,
                   num_groups: int = 1,
                   num_state_groups: int = 0,
                   tp_size: int = 1) -> WorkerCore:
    """Build a bare WorkerCore (no __init__) with the minimal state used by
    the pure translation logic under test (block index translation, transfer
    group building).

    ``num_state_groups`` appends that many mamba-style (non-attention) groups
    after the ``num_groups`` attention groups, which is what turns on the
    hybrid-only logic (spec groups, per-block state completeness, hit
    truncation)."""
    conn = WorkerCore.__new__(WorkerCore)
    conn._manager_block_size = manager_block_size
    conn._vllm_block_size = vllm_block_size or manager_block_size
    conn._tp_size = tp_size
    conn._tp_rank = 0
    conn._self_spec_names = {}
    conn._device = "cpu"
    conn._group_metas = _make_group_metas(
        num_groups, num_state_groups, conn._vllm_block_size)
    conn._num_groups = len(conn._group_metas)
    conn._state_group_idxs = [m.group_idx for m in conn._group_metas
                              if not m.is_attention]
    return conn


def make_scheduler_core(manager_block_size: int = 16,
                        vllm_block_size: Optional[int] = None,
                        num_groups: int = 1,
                        num_state_groups: int = 0,
                        tp_size: int = 1,
                        locations=None) -> SchedulerCore:
    """Build a bare SchedulerCore (no __init__) with the scheduler-loop state
    build_connector_meta and friends need, plus a mocked LocationQueryManager
    answering ``locations`` (None means "still in flight")."""
    from unittest.mock import MagicMock
    core = SchedulerCore.__new__(SchedulerCore)
    core._manager_block_size = manager_block_size
    core._vllm_block_size = vllm_block_size or manager_block_size
    core._tp_size = tp_size
    core._group_metas = _make_group_metas(
        num_groups, num_state_groups, core._vllm_block_size)
    core._num_groups = len(core._group_metas)
    core._state_group_idxs = [m.group_idx for m in core._group_metas
                              if not m.is_attention]
    core._epoch = 0
    core._alive_requests = {}
    core._waiting_to_load_requests = []
    import threading
    core._waiting_to_save_requests_lock = threading.Lock()
    core._waiting_to_save_requests = []
    core._waiting_to_finish_requests = []
    core._canceled_save_request_ids_lock = threading.Lock()
    core._canceled_save_request_ids = []
    core._http_executor = MagicMock()
    core._manager_client = MagicMock()
    core._coordinator_client = MagicMock()
    core._location_query_manager = MagicMock()
    core._location_query_manager.get_locations_for_query.return_value = (
        locations if locations is not None else [])
    return core
