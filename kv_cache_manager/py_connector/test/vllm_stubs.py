"""Shared test stubs: make ``v1_connector`` importable without vLLM/CUDA/pybind.

``v1_connector`` imports vLLM, the compiled ``kvcm_py_client`` and several
third-party runtime deps (torch, triton, orjson, zmq, requests) at module
level. For pure-logic unit tests we register lightweight stand-ins in
``sys.modules`` *before* the first import, then build connector instances via
``__new__`` with only the attributes the code under test reads. No production
module is modified; real modules are preferred whenever they are importable
(e.g. on a dev machine with a full vLLM venv).

The stand-in spec zoo mirrors the *union* of the vLLM eras the connector
supports: the vllm 0.31 class tree and capability flags (prefix_cacheable /
uses_slot_mapping / has_layer_views, which is what the parse gate dispatches
on) plus the 0.26 ``compress_ratio`` spelling next to ``tokens_per_state``.
Both spellings live on one class because a real engine spec carries exactly one
of them -- the connector reads whichever is not the default (see
``vllm_common._tokens_per_state``).
"""

import importlib.util
import threading
import json
import sys
import threading as _threading
import types
from enum import IntEnum
from typing import Any, Optional
from unittest.mock import MagicMock

#: Modules this run replaced with stand-ins (empty when the real deps are
#: importable). Tests that need real behaviour (e.g. an actual torch tensor)
#: skip on membership instead of failing against MagicMocks.
STUBBED: set = set()


def _module(name: str) -> Any:
    """Return (or register) the stub module; Any because the whole point of
    this helper is dynamic attribute injection that no static type
    describes."""
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
    for name in ("torch", "triton", "triton.language", "zmq"):
        if name not in sys.modules and not _importable(name):
            sys.modules[name] = MagicMock(__name__=name)
            STUBBED.add(name.split(".")[0])

    # requests needs real exception classes, not a MagicMock: production code
    # subclasses them (manager_client defines KvCacheManagerHTTPError(
    # requests.HTTPError, AssertionError)), and a MagicMock attribute cannot
    # serve as a base class (metaclass conflict at class-creation time). The
    # stand-in mirrors requests' own hierarchy (RequestException(IOError));
    # the functional entry points stay MagicMocks because the tests patch
    # them (requests.post / requests.Session) before any call.
    if "requests" not in sys.modules and not _importable("requests"):
        req = _module("requests")
        STUBBED.add("requests")

        class _RequestException(IOError):
            """Stand-in for requests.exceptions.RequestException."""

        class _HTTPError(_RequestException):
            pass

        class _ConnectionError(_RequestException):
            pass

        class _Timeout(_RequestException):
            pass

        req.RequestException = _RequestException
        req.HTTPError = _HTTPError
        req.ConnectionError = _ConnectionError
        req.Timeout = _Timeout
        req.Session = MagicMock(name="requests.Session")
        req.post = MagicMock(name="requests.post")
        req.get = MagicMock(name="requests.get")
        # manager_client's annotations reference requests.Response, which is
        # evaluated at import time even when never called.
        req.Response = MagicMock(name="requests.Response")

    # orjson is used functionally (CoordinateMsgSerializer round trips), so
    # the stand-in must actually (de)serialize; stdlib json handles the
    # dataclass payloads via __dict__.
    if "orjson" not in sys.modules and not _importable("orjson"):
        orjson = _module("orjson")
        orjson.dumps = lambda obj: json.dumps(
            obj, default=lambda o: o.__dict__
        ).encode()
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

    class KVQuantMode(IntEnum):
        """Mirrors vLLM's KVQuantMode: the values are the spec contract."""

        NONE = 0
        FP8_PER_TENSOR = 1
        INT8_PER_TOKEN_HEAD = 2
        FP8_PER_TOKEN_HEAD = 3
        INT4_PER_TOKEN_HEAD = 4
        NVFP4 = 5

    #: Bytes per element, keyed by dtype name. The stub specs accept a real
    #: torch dtype or its name: the open-source CI has no torch, and the
    #: connector only ever reads str(spec.dtype).
    _DTYPE_ITEMSIZE = {
        "uint8": 1,
        "float8_e4m3fn": 1,
        "float16": 2,
        "bfloat16": 2,
        "float32": 4,
    }

    def _dtype_name(dtype: Any) -> str:
        # A MagicMock stand-in for torch.uint8 remembers the attribute name.
        mock_name = getattr(dtype, "_mock_name", None)
        if isinstance(mock_name, str):
            return mock_name
        return str(dtype)

    def _dtype_itemsize(dtype: Any) -> int:
        itemsize = getattr(dtype, "itemsize", None)
        if isinstance(itemsize, int):
            return itemsize
        return _DTYPE_ITEMSIZE[_dtype_name(dtype).rsplit(".", 1)[-1]]

    def _quant_mode_value(mode: Any) -> int:
        return int(mode or 0)

    class KVCacheSpec:
        """The vLLM 0.31 KVCacheSpec root: the three capability flags the parse
        gate reads. Every real spec answers them, and only the ring-buffer /
        host-tier ones answer False (per-class overrides below). The flags
        arrived across vLLM 0.29 - 0.30, which is why the connector reads them
        with ``getattr(..., True)``."""

        @property
        def prefix_cacheable(self) -> bool:
            return True

        @property
        def uses_slot_mapping(self) -> bool:
            return True

        @property
        def has_layer_views(self) -> bool:
            return True

    class AttentionSpec(KVCacheSpec):
        """vLLM 0.26 AttentionSpec: the derived page sizes the connector reads
        are computed from the raw fields (see the real vLLM for the algebra).

        ``page_size_bytes`` is a stub-only shortcut that *pins* the compact
        page size for hand-built fixtures -- the repo's older tests pass it
        positionally. Leave it None to exercise the real algebra.

        ``compress_ratio`` (vllm <= 0.28) and ``tokens_per_state`` (vllm >=
        0.29) are both carried: this stand-in mirrors the union of the two
        eras, and a real engine spec carries exactly one of them. The
        *effective* value -- the one the algebra uses -- is whichever is not
        the default 1, exactly as ``vllm_common._tokens_per_state`` reads it."""

        def __init__(
            self,
            block_size,
            page_size_bytes=None,
            page_size_padded=None,
            *,
            num_kv_heads=1,
            head_size=None,
            dtype=None,
            kv_quant_mode=KVQuantMode.NONE,
            indexes_kv_by_block_stride=False,
            head_size_v=None,
            tokens_per_state=1,
        ):
            self.block_size = block_size
            self.num_kv_heads = num_kv_heads
            self.head_size = head_size
            self.head_size_v = head_size if head_size_v is None else head_size_v
            self.dtype = dtype
            self.kv_quant_mode = kv_quant_mode
            self.indexes_kv_by_block_stride = indexes_kv_by_block_stride
            self.tokens_per_state = tokens_per_state
            self.page_size_padded = page_size_padded
            self._pinned_page_bytes = page_size_bytes
            try:
                real = (
                    page_size_bytes
                    if page_size_bytes is not None
                    else self._compute_real_page_size_bytes()
                )
            except (ArithmeticError, TypeError, AssertionError, KeyError):
                # A malformed spec (compress_ratio < 1, missing head_size, ...)
                # is the gate's business: vLLM computes these lazily, this stub
                # eagerly, so keep the derived sizes None instead of failing the
                # construction and let the gate refuse before any read.
                real = None
            self.real_page_size_bytes = real
            # Per-token-head modes budget their scale tensors inside the page
            # (vLLM's AttentionSpec.unpadded_page_size_bytes).
            self.unpadded_page_size_bytes = (
                None
                if real is None
                else real
                + (
                    2 * block_size * num_kv_heads * 4
                    if _quant_mode_value(kv_quant_mode) in (2, 3, 4)
                    else 0
                )
            )

        def _compute_real_page_size_bytes(self):
            assert self.head_size is not None, (
                "the stub spec needs head_size to derive its page size"
            )
            head_dim = (
                self.head_size // 2
                if _quant_mode_value(self.kv_quant_mode)
                == KVQuantMode.INT4_PER_TOKEN_HEAD
                else self.head_size
            )
            head_dim_v = (
                self.head_size_v // 2
                if _quant_mode_value(self.kv_quant_mode)
                == KVQuantMode.INT4_PER_TOKEN_HEAD
                else self.head_size_v
            )
            # One K row and one V row per token (head_size_v defaults to
            # head_size, i.e. the K/V-symmetric case).
            return (
                self.block_size
                * self.num_kv_heads
                * (head_dim + head_dim_v)
                * _dtype_itemsize(self.dtype)
            )

        def _effective_tokens_per_state(self):
            """The union of the 0.26 compress_ratio and the 0.31
            tokens_per_state: the non-default one is the effective value (see
            the class docstring)."""
            for value in (self.tokens_per_state, getattr(self, "compress_ratio", 1)):
                if value != 1:
                    return value
            return 1

        @property
        def storage_block_size(self):
            return self.block_size

        @property
        def page_size_bytes(self):
            if self.page_size_padded is not None:
                assert self.page_size_padded >= self.unpadded_page_size_bytes, (
                    "page_size_padded below the compact page size"
                )
                return self.page_size_padded
            return self.unpadded_page_size_bytes

    def _apply_alignment_padding(spec: Any) -> None:
        """Mirrors vLLM's helper: the alignment rounds the compact page up."""
        if spec.alignment is None:
            return
        actual = spec.real_page_size_bytes
        padded = -(-actual // spec.alignment) * spec.alignment
        if padded != actual:
            spec.page_size_padded = padded

    class FullAttentionSpec(AttentionSpec):
        def __init__(
            self,
            block_size,
            page_size_bytes=None,
            page_size_padded=None,
            *,
            num_kv_heads=1,
            head_size=None,
            dtype=None,
            kv_quant_mode=KVQuantMode.NONE,
            indexes_kv_by_block_stride=False,
            head_size_v=None,
            sliding_window=None,
            attention_chunk_size=None,
            non_causal=False,
            tokens_per_state=1,
        ):
            super().__init__(
                block_size,
                page_size_bytes,
                page_size_padded,
                num_kv_heads=num_kv_heads,
                head_size=head_size,
                dtype=dtype,
                kv_quant_mode=kv_quant_mode,
                indexes_kv_by_block_stride=indexes_kv_by_block_stride,
                head_size_v=head_size_v,
                tokens_per_state=tokens_per_state,
            )
            self.sliding_window = sliding_window
            self.attention_chunk_size = attention_chunk_size
            self.non_causal = non_causal

    class TQFullAttentionSpec(FullAttentionSpec):
        """vllm 0.22 - 0.27: full attention whose page size counts TQ slots."""

        def __init__(self, *args, tq_slot_size=0, **kwargs):
            super().__init__(*args, **kwargs)
            self.tq_slot_size = tq_slot_size

    class SinkFullAttentionSpec(FullAttentionSpec):
        """OpenPangu static sink: the sink constant lives in every request's own
        blocks, so the blocks still hold the whole prefix."""

        def __init__(self, *args, sink_len=None, **kwargs):
            super().__init__(*args, **kwargs)
            self.sink_len = sink_len

    class RSWASpec(FullAttentionSpec):
        """R-SWA: prefill stays global, only the last ``rswa_window`` generated
        tokens are kept -- the gap in between is freed while the request lives."""

        def __init__(self, *args, rswa_window=None, **kwargs):
            super().__init__(*args, **kwargs)
            self.rswa_window = rswa_window

    class MLAAttentionSpec(FullAttentionSpec):
        """Mirrors the real spec's consumed contract: the MLA-only fields the
        connector gates on plus the byte algebra FROZEN pins down."""

        def __init__(
            self,
            block_size,
            page_size_bytes=None,
            page_size_padded=None,
            *,
            num_kv_heads=1,
            head_size=None,
            dtype=None,
            kv_quant_mode=KVQuantMode.NONE,
            indexes_kv_by_block_stride=False,
            cache_dtype_str=None,
            compress_ratio=1,
            alignment=None,
            model_version=None,
            head_size_v=0,
            state_content_bytes=None,
            tokens_per_state=1,
        ):
            self.cache_dtype_str = cache_dtype_str
            self.compress_ratio = compress_ratio
            self.alignment = alignment
            self.model_version = model_version
            self.state_content_bytes = state_content_bytes
            super().__init__(
                block_size,
                page_size_bytes,
                page_size_padded,
                num_kv_heads=num_kv_heads,
                head_size=head_size,
                dtype=dtype,
                kv_quant_mode=kv_quant_mode,
                indexes_kv_by_block_stride=indexes_kv_by_block_stride,
                head_size_v=head_size_v,
                tokens_per_state=tokens_per_state,
            )
            _apply_alignment_padding(self)

        @property
        def storage_block_size(self):
            return self.block_size // self._effective_tokens_per_state()

        def _compute_real_page_size_bytes(self):
            assert self.head_size is not None, (
                "the stub spec needs head_size to derive its page size"
            )
            if self.state_content_bytes is not None:
                # vllm 0.31 packs NoPE + RoPE + scales into one opaque cell.
                return self.storage_block_size * self.state_content_bytes
            if self.cache_dtype_str == "fp8_ds_mla":
                if self.model_version == "deepseek_v4":
                    # DeepSeek V4: 448 B NoPE + 128 B RoPE + 8 B scale/token.
                    return self.storage_block_size * 584
                # V3.2 main MLA: a 656-byte custom row per token.
                return self.block_size * 656
            head_dim = (
                self.head_size // 2
                if _quant_mode_value(self.kv_quant_mode)
                == KVQuantMode.INT4_PER_TOKEN_HEAD
                else self.head_size
            )
            # One latent row per token, no K/V factor of two.
            return (
                self.storage_block_size
                * self.num_kv_heads
                * head_dim
                * _dtype_itemsize(self.dtype)
            )

    class HiddenStateCacheSpec(MLAAttentionSpec):
        """extract_hidden_states marker: an MLA-shaped cache of hidden states."""

        pass

    class SlidingWindowSpec(AttentionSpec):
        """Windowed KV: blocks below the window are freed while the request
        lives (so they are not a token prefix)."""

        def __init__(
            self,
            block_size,
            sliding_window=None,
            page_size_bytes=None,
            page_size_padded=None,
            *,
            num_kv_heads=1,
            head_size=None,
            dtype=None,
            kv_quant_mode=KVQuantMode.NONE,
            head_size_v=None,
            extra_retained_tokens=0,
            tokens_per_state=1,
        ):
            self.sliding_window = sliding_window
            self.extra_retained_tokens = extra_retained_tokens
            super().__init__(
                block_size,
                page_size_bytes,
                page_size_padded,
                num_kv_heads=num_kv_heads,
                head_size=head_size,
                dtype=dtype,
                kv_quant_mode=kv_quant_mode,
                head_size_v=head_size_v,
                tokens_per_state=tokens_per_state,
            )

    class KpoolTailSpec(SlidingWindowSpec):
        """GLM-5-Next kpool indexer tail: one ring block per request."""

        @property
        def prefix_cacheable(self):
            return False

        @property
        def uses_slot_mapping(self):
            return False

    class CircularBufferSpec(AttentionSpec):
        """One block per request holding the open compression group's raw keys."""

        @property
        def prefix_cacheable(self):
            return False

        @property
        def uses_slot_mapping(self):
            return False

    class ChunkedLocalAttentionSpec(AttentionSpec):
        """Chunked-local KV: blocks outside the current chunk are freed."""

        def __init__(self, *args, attention_chunk_size=None, **kwargs):
            super().__init__(*args, **kwargs)
            self.attention_chunk_size = attention_chunk_size

    class CrossAttentionSpec(AttentionSpec):
        """Encoder-decoder cross attention: encoder states, not a token prefix."""

        pass

    class EncoderOnlyAttentionSpec(AttentionSpec):
        """Encoder-only layers: vLLM returns no spec for them at all."""

        pass

    class HiSparseHotSpec(KVCacheSpec):
        """Ephemeral per-request HiSparse hot pages."""

        def __init__(self, block_size=None, *, page_size, blocks_per_request=1):
            self.block_size = block_size
            self.page_size = page_size
            self.blocks_per_request = blocks_per_request

        @property
        def page_size_bytes(self):
            return self.page_size

        @property
        def prefix_cacheable(self):
            return False

        @property
        def uses_slot_mapping(self):
            return False

        @property
        def has_layer_views(self):
            return False

    class HiSparseResidentSpec(KVCacheSpec):
        """Reclaimable GPU-resident pages for host-backed HiSparse KV.

        ``uses_slot_mapping`` stays True (the real spec inherits the base
        default) -- the measured reason the scratch predicate cannot rely on
        slot mapping alone."""

        def __init__(self, block_size=None, *, page_size):
            self.block_size = block_size
            self.page_size = page_size

        @property
        def page_size_bytes(self):
            return self.page_size

        @property
        def prefix_cacheable(self):
            return False

        @property
        def has_layer_views(self):
            return False

    class SlidingWindowMLASpec(SlidingWindowSpec):
        """Windowed / partial-state MLA cache (V4 SWA and compressor state)."""

        def __init__(
            self,
            block_size,
            sliding_window=None,
            page_size_bytes=None,
            page_size_padded=None,
            *,
            num_kv_heads=1,
            head_size=None,
            dtype=None,
            kv_quant_mode=KVQuantMode.NONE,
            cache_dtype_str=None,
            compress_ratio=1,
            alignment=None,
            model_version=None,
            bounded_replay=False,
            head_size_v=0,
            tokens_per_state=1,
        ):
            self.cache_dtype_str = cache_dtype_str
            self.compress_ratio = compress_ratio
            self.alignment = alignment
            self.model_version = model_version
            self.bounded_replay = bounded_replay
            super().__init__(
                block_size,
                sliding_window,
                page_size_bytes,
                page_size_padded,
                num_kv_heads=num_kv_heads,
                head_size=head_size,
                dtype=dtype,
                kv_quant_mode=kv_quant_mode,
                head_size_v=head_size_v,
                tokens_per_state=tokens_per_state,
            )
            _apply_alignment_padding(self)

        @property
        def prefix_cacheable(self):
            # vllm 0.31: bounded replay hands those layers back to the engine.
            return not self.bounded_replay

        @property
        def storage_block_size(self):
            return self.block_size // self._effective_tokens_per_state()

        def _compute_real_page_size_bytes(self):
            assert self.head_size is not None, (
                "the stub spec needs head_size to derive its page size"
            )
            if (
                self.model_version == "deepseek_v4"
                and self.cache_dtype_str == "fp8_ds_mla"
            ):
                return self.storage_block_size * 584
            assert self.model_version in (None, "deepseek_v4"), (
                f"Unsupported model version: {self.model_version}"
            )
            return (
                self.storage_block_size
                * self.num_kv_heads
                * self.head_size
                * _dtype_itemsize(self.dtype)
            )

    class UniformTypeKVCacheSpecs(KVCacheSpec):
        """vLLM's wrapper: one group of layers whose specs share a block size
        but may differ in page size (V3.2 main + indexer)."""

        def __init__(self, block_size, kv_cache_specs):
            self.block_size = block_size
            self.kv_cache_specs = kv_cache_specs

        @property
        def first_spec(self):
            return next(iter(self.kv_cache_specs.values()))

        @property
        def page_size_bytes(self):
            return sum(spec.page_size_bytes for spec in self.kv_cache_specs.values())

        @property
        def prefix_cacheable(self):
            return all(spec.prefix_cacheable for spec in self.kv_cache_specs.values())

        @property
        def uses_slot_mapping(self):
            return self.first_spec.uses_slot_mapping

        def get_page_sizes(self):
            return list(
                set(spec.page_size_bytes for spec in self.kv_cache_specs.values())
            )

        @classmethod
        def from_specs(cls, kv_cache_specs):
            block_sizes = set(spec.block_size for spec in kv_cache_specs.values())
            if len(block_sizes) > 1:
                return None
            return cls(
                block_size=next(iter(kv_cache_specs.values())).block_size,
                kv_cache_specs=kv_cache_specs,
            )

    class MambaSpec(KVCacheSpec):
        def __init__(self, block_size, page_size_bytes):
            self.block_size = block_size
            self.page_size_bytes = page_size_bytes

    kv_cache_interface.KVQuantMode = KVQuantMode
    kv_cache_interface.KVCacheSpec = KVCacheSpec
    kv_cache_interface.AttentionSpec = AttentionSpec
    kv_cache_interface.FullAttentionSpec = FullAttentionSpec
    kv_cache_interface.TQFullAttentionSpec = TQFullAttentionSpec
    kv_cache_interface.SinkFullAttentionSpec = SinkFullAttentionSpec
    kv_cache_interface.RSWASpec = RSWASpec
    kv_cache_interface.MLAAttentionSpec = MLAAttentionSpec
    kv_cache_interface.HiddenStateCacheSpec = HiddenStateCacheSpec
    kv_cache_interface.SlidingWindowSpec = SlidingWindowSpec
    kv_cache_interface.SlidingWindowMLASpec = SlidingWindowMLASpec
    kv_cache_interface.KpoolTailSpec = KpoolTailSpec
    kv_cache_interface.CircularBufferSpec = CircularBufferSpec
    kv_cache_interface.ChunkedLocalAttentionSpec = ChunkedLocalAttentionSpec
    kv_cache_interface.CrossAttentionSpec = CrossAttentionSpec
    kv_cache_interface.EncoderOnlyAttentionSpec = EncoderOnlyAttentionSpec
    kv_cache_interface.HiSparseHotSpec = HiSparseHotSpec
    kv_cache_interface.HiSparseResidentSpec = HiSparseResidentSpec
    kv_cache_interface.UniformTypeKVCacheSpecs = UniformTypeKVCacheSpecs
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
    RETENTION_SLIDING,
    AttentionGroupMeta,
    StateGroupMeta,
    WindowRetention,
    spec_group_plan,
)
from kv_cache_manager.py_connector.vllm.connector_scheduler import ConnectorScheduler  # noqa: E402
from kv_cache_manager.py_connector.vllm.connector_worker import ConnectorWorker  # noqa: E402


def _make_group_metas(
    num_groups: int,
    num_state_groups: int,
    block_size: int,
    windowed: int = 0,
    window_tokens: int = 1024,
) -> list:
    """Attention groups first, then mamba-style state groups; group_idx is the
    vLLM group index (what block tables are indexed by).

    ``windowed`` makes that many of the attention groups sliding-window ones
    (retention metadata included), which turns on the eager-capture and hit
    clamp paths; the rest stay full-prefix."""
    return [
        AttentionGroupMeta(
            group_idx=i,
            layer_names=[f"l{i}"],
            block_size=block_size,
            per_block_bytes=0,
            # A positive page size: the tensor-side cross-check rejects 0, and
            # the translation paths under test never read it.
            page_bytes=block_size * 8,
            retention=(
                WindowRetention(kind=RETENTION_SLIDING, window_tokens=window_tokens)
                if i < windowed
                else None
            ),
        )
        for i in range(num_groups)
    ] + [
        StateGroupMeta(
            group_idx=num_groups + i,
            layer_names=[f"m{i}"],
            block_size=block_size,
            per_block_bytes=0,
            page_size_bytes=0,
        )
        for i in range(num_state_groups)
    ]


def install_capture_ledger(conn: Any) -> None:
    """Give a hand-built ConnectorWorker the capture ledger its slot
    bookkeeping lives in (the real one builds it in __init__)."""
    from kv_cache_manager.py_connector.vllm.connector_worker import _CaptureLedger

    conn._captures = _CaptureLedger()
    conn._staged = conn._captures.staged
    conn._in_flight = conn._captures.in_flight
    conn._block_uris = {}
    conn._verdict_pending = {}
    conn._tp_size = getattr(conn, "_tp_size", 1)


def make_connector(
    manager_block_size: int = 16,
    vllm_block_size: Optional[int] = None,
    num_groups: int = 1,
    num_state_groups: int = 0,
    tp_size: int = 1,
) -> ConnectorWorker:
    """Build a bare ConnectorWorker (no __init__) with the minimal state used by
    the pure translation logic under test (block index translation, transfer
    group building).

    ``num_state_groups`` appends that many mamba-style (non-attention) groups
    after the ``num_groups`` attention groups, which is what turns on the
    hybrid-only logic (spec groups, per-block state completeness, hit
    truncation)."""
    conn = ConnectorWorker.__new__(ConnectorWorker)
    conn._manager_block_size = manager_block_size
    # Extra harness attributes the worker itself never declares.
    conn._vllm_block_size = (  # ty: ignore[unresolved-attribute]
        vllm_block_size or manager_block_size
    )
    conn._tp_size = tp_size
    conn._tp_rank = 0
    conn._device = "cpu"
    conn._group_metas = _make_group_metas(
        num_groups,
        num_state_groups,
        conn._vllm_block_size,  # ty: ignore[unresolved-attribute]
    )
    conn._num_groups = len(conn._group_metas)
    conn._state_group_idxs = [  # ty: ignore[unresolved-attribute]
        m.group_idx for m in conn._group_metas if isinstance(m, StateGroupMeta)
    ]
    return conn


class FakeLocationQueries:
    """Test double for LocationQueryManager: same produce/consume contract.

    ``locations`` is the manager's answer (None = in flight). The match
    hook's clamped result (store_result) is what the allocation consumes;
    the offset recorded at get time travels with it."""

    def __init__(self, locations: Optional[list] = None) -> None:
        self.locations = locations
        self.in_flight = locations is None
        self._stored: Any = None
        self.last_computed_blocks = 0

    def get_locations_for_query(self, request: Any, computed_blocks: int) -> Any:
        self.last_computed_blocks = computed_blocks
        # list(self.locations) is only reached when not in flight.
        return None if self.in_flight else list(self.locations)  # ty: ignore[invalid-argument-type]

    def store_result(self, req_id, locations):
        self._stored = list(locations)

    def consume_locations(self, req_id):
        if self.in_flight or self._stored is None:
            return None
        result, self._stored = self._stored, None
        return result, self.last_computed_blocks

    def invalidate(self, req_id):
        self._stored = None


def make_connector_scheduler(
    manager_block_size: int = 16,
    vllm_block_size: Optional[int] = None,
    num_groups: int = 1,
    num_state_groups: int = 0,
    tp_size: int = 1,
    locations=None,
    windowed: int = 0,
    window_tokens: int = 1024,
    has_eagle_group: bool = False,
) -> ConnectorScheduler:
    """Build a bare ConnectorScheduler (no __init__) with the scheduler-loop state
    build_connector_meta and friends need, plus a FakeLocationQueries
    answering ``locations`` (None means "still in flight").

    ``windowed`` turns the first that many attention groups into windowed ones
    (retention metadata included), which is what enables the capture requests
    and the windowed hit clamp."""
    from unittest.mock import MagicMock

    core = ConnectorScheduler.__new__(ConnectorScheduler)
    core._manager_block_size = manager_block_size
    core._vllm_block_size = vllm_block_size or manager_block_size
    core._tp_size = tp_size
    core._group_metas = _make_group_metas(
        num_groups, num_state_groups, core._vllm_block_size, windowed, window_tokens
    )
    core._num_groups = len(core._group_metas)
    core._state_group_idxs = [
        m.group_idx for m in core._group_metas if isinstance(m, StateGroupMeta)
    ]
    core._windowed_metas = [
        m
        for m in core._group_metas
        if isinstance(m, AttentionGroupMeta) and m.retention is not None
    ]
    core._has_eagle_group = has_eagle_group
    declared, core._spec_group_names = spec_group_plan(core._group_metas, tp_size)
    core._narrowest_spec_group = (
        min(declared, key=lambda group: (len(group["spec_names"]), group["name"]))[
            "name"
        ]
        if declared
        else None
    )
    core._window_claims = {}
    core._window_claims_lock = threading.Lock()
    core._announced_sessions = {}
    core._announced_sessions_counters = {}
    core._announced_sessions_lock = _threading.Lock()
    core._pending_release_captures = []
    core._pending_release_captures_lock = _threading.Lock()
    core._epoch = 0
    core._tracked = {}
    core._load_failed = set()
    core._load_attempted = set()
    core._waiting_to_load_requests = []
    core._waiting_to_save_requests_lock = _threading.Lock()
    core._waiting_to_save_requests = []
    core._waiting_to_finish_requests = []
    core._canceled_save_request_ids_lock = _threading.Lock()
    core._canceled_save_request_ids = []
    core._http_executor = MagicMock()
    core._manager_client = MagicMock()
    core._coordinator_client = MagicMock()
    core._location_query_manager = (  # ty: ignore[invalid-assignment]
        FakeLocationQueries(locations)
    )
    return core
