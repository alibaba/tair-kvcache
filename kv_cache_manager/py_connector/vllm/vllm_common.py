"""Shared vocabulary between the scheduler side and the worker side.

Everything here is role-agnostic: the data model both cores speak
(GroupMeta), the spec naming scheme shared with the manager
registration, the KV layout normalization, the hybrid capability gate and
the kv_cache_config parsing. The two cores (scheduler_core / worker_core)
and the thin connector shell (v1_connector) build on this module; nothing
here may import them.

Every layout the transfer path accepts is one latent row per token; the
refusals below say why the rest are refused. Compressed MLA
(compress_ratio > 1) is a documented extension point, not a missing check:
supporting it needs three dimensions this data model does not carry yet --
rows per block (spec.storage_block_size), tokens per row (compress_ratio) and
slots per block -- because vLLM addresses a compressed row by its storage
block number (see get_compressed_slot_mapping), never by token.
"""

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, List, NamedTuple, Optional, Tuple

import torch

from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    MLAAttentionSpec,
    MambaSpec,
)

try:
    # vLLM >= 0.22 names; vllm 0.14 has none of them. The sentinels are only
    # ever used behind a null check (isinstance(x, None) would raise).
    from vllm.v1.kv_cache_interface import (  # ty: ignore[unresolved-import]
        KVQuantMode,
        SlidingWindowMLASpec,
        UniformTypeKVCacheSpecs,
    )
except ImportError:  # pragma: no cover - older vLLM eras
    KVQuantMode = None  # ty: ignore[invalid-assignment]
    SlidingWindowMLASpec = None  # ty: ignore[invalid-assignment]
    UniformTypeKVCacheSpecs = None  # ty: ignore[invalid-assignment]

from kv_cache_manager.py_connector.common.logger import logger
from kv_cache_manager.py_connector.vllm.transfer_types import (
    KVLayout,
)

if TYPE_CHECKING:
    from vllm.v1.kv_cache_interface import KVCacheConfig

# kv_quant_mode values (vLLM's KVQuantMode). Compared as integers so this
# module does not depend on the enum being importable; the names below are
# only used in refusal messages.
_QUANT_NONE = 0
_QUANT_FP8_PER_TENSOR = 1
# INT8 / FP8 / INT4 per-token-head scales: the scales sit in the page budget
# but outside the rows the gather kernel copies.
_QUANT_PER_TOKEN_HEAD = (2, 3, 4)
_QUANT_NVFP4 = 5
_QUANT_MODE_NAMES = {
    _QUANT_NONE: "NONE",
    _QUANT_FP8_PER_TENSOR: "FP8_PER_TENSOR",
    2: "INT8_PER_TOKEN_HEAD",
    3: "FP8_PER_TOKEN_HEAD",
    4: "INT4_PER_TOKEN_HEAD",
    _QUANT_NVFP4: "NVFP4",
}
# cache_dtype_str values that carry no quantization of their own: the cache
# keeps the model dtype (float16/bfloat16 are legal --kv-cache-dtype values,
# so they must stay accepted).
_PLAIN_CACHE_DTYPES = (None, "auto", "float16", "bfloat16")
# Non-packed per-tensor fp8 cache dtypes: compact pages, layer-level scales.
_PLAIN_FP8_CACHE_DTYPES = ("fp8", "fp8_e4m3")
# The accepted cache_dtype_str values, spelled out for refusal messages: derived
# from the tuples above so the message cannot drift from the gate.
_KNOWN_CACHE_DTYPES = (
    "/".join(str(dtype) for dtype in _PLAIN_CACHE_DTYPES)
    + ", "
    + ", ".join(f'"{dtype}"' for dtype in _PLAIN_FP8_CACHE_DTYPES)
    + ', or "fp8_ds_mla"'
)

# Spec group names advertised at registration and used per key in
# start_write_cache. See build_spec_groups for the semantics.
#
# NOTE: the wire strings are frozen protocol: the manager keys on the
# "full" prefix (meta_searcher.cc) and its tests / the optimizer client
# emit these literals. Only the Python-side names below are free to move;
# "attn" reads as attention-only coverage, "full" as attention + every
# recurrent state group (the union, i.e. *all* specs).
ATTN_ONLY_SPEC_GROUP = "attn"
ALL_SPEC_GROUP = "full"


def spec_name(tp_rank: int, meta: "GroupMeta") -> str:
    """Wire name for one (tp rank, transfer bucket).

    Buckets of one vLLM group share its group_idx (they share the block
    table) and differ by spec_suffix: "" for the first bucket, "_b{k}" for
    the rest. The name is a unique key into the manager's name -> bytes map
    and never parses back, so the suffixes are additive on the wire.
    """
    return f"tp{tp_rank}_g{meta.group_idx}{meta.spec_suffix}"


def build_spec_groups(group_metas: List["GroupMeta"], tp_size: int) -> List[dict]:
    """LocationSpecGroups describing which specs a block may carry.

    Hybrid (mamba "align") models write a *sparse* set of recurrent states:
    vLLM materializes a state only at segment boundaries, so the interior
    manager blocks of a request have attention KV but no state. Declaring
    two groups lets ``start_write_cache`` say, per block, which specs that
    block will actually hold:

    * ``full`` -- *all* specs: attention KV + every recurrent state;
    * ``attn`` -- attention specs only (no state was materialized).

    The manager then stores exactly the advertised specs, reports the real
    per-block coverage in ``getCacheLocation``, and later lets a
    complementary write fill in a block's missing state specs.

    Full-attention models have nothing to be sparse about: they declare no
    groups at all, which keeps their requests byte-identical to before (and
    compatible with managers that predate spec groups).
    """
    state_groups = [m for m in group_metas if isinstance(m, StateGroupMeta)]
    if not state_groups:
        return []
    attn_specs = sorted(
        spec_name(rank, meta)
        for rank in range(tp_size)
        for meta in group_metas
        if isinstance(meta, AttentionGroupMeta)
    )
    all_specs = sorted(
        spec_name(rank, meta) for rank in range(tp_size) for meta in group_metas
    )
    return [
        {"name": ATTN_ONLY_SPEC_GROUP, "spec_names": attn_specs},
        {"name": ALL_SPEC_GROUP, "spec_names": all_specs},
    ]


@dataclass(frozen=True)
class GroupMeta:
    """Static description of one *transfer bucket*, derived from KVCacheConfig
    (see parse_groups). Available in both scheduler and worker roles (before
    tensors exist). Kind-specific subclasses carry the kind-specific sizing.

    A bucket is one manager location: one wire name, one per-block byte size,
    one staging pool. A plain kv_cache_group maps to exactly one bucket; a
    UniformTypeKVCacheSpecs group maps to one per page layout it packs."""

    group_idx: int
    layer_names: List[str]
    # The bucket's block table granularity in tokens (spec.block_size).
    block_size: int
    # Bytes stored per manager block for the whole bucket.
    per_block_bytes: int
    # Bucket suffix in the wire name: "" for a group's first bucket (the only
    # one unless vLLM packs several page layouts into one group).
    spec_suffix: str = ""


@dataclass(frozen=True)
class AttentionGroupMeta(GroupMeta):
    """Attention bucket: token-granular KV, re-blockable to the manager block
    size. Sizing derives from the *compact* page size (see parse_groups)."""

    # Compact page bytes (spec.real_page_size_bytes): the alignment tail of a
    # padded page is never copied. Keyword-only because the base class already
    # carries the defaulted spec_suffix.
    page_bytes: int = field(kw_only=True)


@dataclass(frozen=True)
class StateGroupMeta(GroupMeta):
    """MambaSpec group: one opaque state per block, verbatim byte copies.
    page_size_bytes is per layer; per_block_bytes = page_size_bytes * layers."""

    # Bytes per block per state layer (spec.page_size_bytes).
    page_size_bytes: int = 0


def state_kv_view(
    cache: torch.Tensor | list[torch.Tensor] | tuple[torch.Tensor, ...],
    page_size_bytes: int,
) -> torch.Tensor:
    """View state pages as bytes, preserving layer offsets and block strides.

    Older vLLM supplies typed conv/SSM views sharing storage. Current vLLM
    supplies a per-layer [B, 1, 1, C] byte view into a shared allocation;
    C excludes page padding, and consecutive blocks need not be contiguous.

    Invalid layouts raise before set_ can resize the shared storage. Checks
    are explicit so the same safety contract applies under Python -O.
    """
    if not isinstance(page_size_bytes, int):
        raise TypeError(f"state page size must be an integer, got {page_size_bytes!r}")
    if page_size_bytes <= 0:
        raise ValueError(f"state page size must be positive, got {page_size_bytes}")
    byte_cache = isinstance(cache, torch.Tensor)
    if isinstance(cache, torch.Tensor):
        states = (cache,)
    elif isinstance(cache, (list, tuple)) and cache:
        states = cache
    else:
        raise TypeError(
            "state cache must be a byte tensor or a nonempty sequence of state tensors"
        )

    for index, state in enumerate(states):
        if not isinstance(state, torch.Tensor):
            raise TypeError(f"state component {index} must be a tensor")
        if state.layout != torch.strided or state.device.type == "meta":
            raise ValueError(
                f"state component {index} must have materialized strided storage"
            )
        if state.dim() == 0:
            raise ValueError(f"state component {index} must have a block dimension")

    first = states[0]
    if byte_cache:
        if first.dtype not in (torch.int8, torch.uint8):
            raise ValueError(f"state cache must be a byte tensor, got {first.dtype}")
        if first.dim() != 4 or first.shape[1:3] != (1, 1):
            raise ValueError(
                f"state cache must have shape [B, 1, 1, C], got {tuple(first.shape)}"
            )
        if first.stride(-1) != 1 or not 0 < first.shape[-1] <= page_size_bytes:
            raise ValueError(
                f"state content must be contiguous and fit page {page_size_bytes}: "
                f"shape={tuple(first.shape)}, stride={first.stride()}"
            )

    storage = first.untyped_storage()
    num_blocks = first.shape[0]
    if num_blocks <= 0:
        raise ValueError(
            f"state cache must have a positive block count, got {num_blocks}"
        )
    offset = first.storage_offset() * first.element_size()
    block_stride = first.stride(0) * first.element_size()
    if block_stride < page_size_bytes:
        raise ValueError(
            f"state block stride {block_stride} < page size {page_size_bytes}"
        )
    for index, state in enumerate(states):
        if (
            state.device != first.device
            or state.untyped_storage().data_ptr() != storage.data_ptr()
        ):
            raise ValueError(
                f"state component {index}: state tensors do not share storage"
            )
        if (
            state.shape[0] != num_blocks
            or state.stride(0) * state.element_size() != block_stride
        ):
            raise ValueError(
                f"state component {index} must share block count {num_blocks} "
                f"and byte stride {block_stride}"
            )
        # Bound the full strided extent, not merely the starting address or
        # numel: gaps between dimensions are part of the physical page.
        state_offset = state.storage_offset() * state.element_size()
        span_elements = 0
        if state.numel():
            span_elements = 1 + sum(
                (size - 1) * stride
                for size, stride in zip(state.shape[1:], state.stride()[1:])
            )
        state_end = state_offset + span_elements * state.element_size()
        if state_offset < offset or state_end > offset + page_size_bytes:
            raise ValueError(
                f"state component {index} byte range [{state_offset}, {state_end}) "
                f"lies outside page [{offset}, {offset + page_size_bytes})"
            )
    end = offset + (num_blocks - 1) * block_stride + page_size_bytes
    if storage.nbytes() < end:
        raise ValueError(f"state storage {storage.nbytes()} < required end {end}")
    return torch.empty(0, dtype=torch.uint8, device=first.device).set_(
        storage, offset, (num_blocks, page_size_bytes), (block_stride, 1)
    )



def _quant_mode(spec: Any) -> int:
    return int(getattr(spec, "kv_quant_mode", _QUANT_NONE) or _QUANT_NONE)



def _quant_mode_name(spec: Any) -> str:
    """Readable kv_quant_mode for messages: the enum's name where the era has
    one, else the canonical name of the raw value (stub / plain int)."""
    mode = getattr(spec, "kv_quant_mode", _QUANT_NONE)
    name = getattr(mode, "name", None)
    if isinstance(name, str):
        return name
    return _QUANT_MODE_NAMES.get(_quant_mode(spec), f"mode {_quant_mode(spec)}")


def _check_mla_structure(origin: str, spec: "MLAAttentionSpec") -> int:
    """Refuse the compress_ratio / model_version combinations vLLM's own size
    algebra cannot represent, reading raw fields only (never
    storage_block_size / real_page_size_bytes / page_size_bytes): a ratio
    below 1 divides by zero, one that does not divide block_size would be
    floored silently, and an unknown model_version has no algebra at all.
    Returns the validated compress_ratio.

    Supporting a compressed layout (c > 1) is a documented extension point,
    see the module docstring."""
    compress_ratio = getattr(spec, "compress_ratio", 1)
    if compress_ratio < 1:
        raise NotImplementedError(
            f"{origin}: MLAAttentionSpec compress_ratio={compress_ratio} is "
            f"invalid (must be >= 1); refusing to size the KV cache from it"
        )
    if compress_ratio > 1 and spec.block_size % compress_ratio:
        raise NotImplementedError(
            f"{origin}: MLAAttentionSpec block_size={spec.block_size} is not "
            f"divisible by compress_ratio={compress_ratio}; refusing to "
            f"silently floor the storage layout"
        )
    model_version = getattr(spec, "model_version", None)
    if model_version not in (None, "deepseek_v4"):
        raise NotImplementedError(
            f"{origin}: MLAAttentionSpec model_version={model_version!r} is not "
            f'a layout this connector knows (expected None or "deepseek_v4")'
        )
    return compress_ratio


def _check_mla_variant(
    origin: str, spec: "MLAAttentionSpec", *, calculate_kv_scales: bool
) -> None:
    """Gate one MLAAttentionSpec to the token-granular layouts the transfer
    path supports, by the (cache_dtype_str, kv_quant_mode) pair, after the
    structural check (_check_mla_structure) that guarantees the ratio is
    usable.

    Accepted (one latent row per token, self-contained bytes):

    * "fp8_ds_mla" (DeepSeek V3.2/V4 main layers): a packed 656/584 B layout
      whose fp8 scale lives *inside* each row, so a verbatim round trip keeps
      it. kv_quant_mode is FP8_PER_TENSOR there -- that alone must not refuse
      it (a naive quant gate would make every real V3.2 main layer unreachable);
    * None/auto/float16/bfloat16 with kv_quant_mode NONE: the plain latent
      cache (GLM, indexer layers, bf16 V3.2 main);
    * "fp8"/"fp8_e4m3" with kv_quant_mode FP8_PER_TENSOR: layer-level scales
      live outside the cached bytes, so the round trip is exact -- but only
      while vLLM does not *calibrate* those scales at runtime
      (calculate_kv_scales): a reload would then decode with a stale scale.

    Refused: compression (a row spans several tokens, and DeepSeek V4 reads
    those rows together with its sliding-window and compressor-state groups,
    so a partial transfer would silently corrupt reuse), per-token-head /
    NVFP4 modes (their scales sit in the page budget but outside the
    transferred rows), unknown cache_dtype_str, and (cds, kv_quant_mode) pairs
    vLLM cannot produce -- guessing a scale layout is worse than refusing.
    """
    compress_ratio = _check_mla_structure(origin, spec)
    if compress_ratio > 1:
        raise NotImplementedError(
            f"{origin}: MLAAttentionSpec compress_ratio={compress_ratio} stores "
            f"one latent row per {compress_ratio} tokens; DeepSeek V4 reads "
            f"those compressed rows together with its sliding-window and "
            f"compressor-state caches, which live in separate vLLM groups, and "
            f"the connector must serve a request's KV groups as a whole -- "
            f"compressed MLA KV is not supported by TairKvCacheConnector; "
            f"supporting it needs a rows-per-block (storage_block_size) "
            f"dimension this transfer model does not have"
        )
    cache_dtype_str = getattr(spec, "cache_dtype_str", None)
    mode = _quant_mode(spec)
    if cache_dtype_str == "fp8_ds_mla":
        if mode != _QUANT_FP8_PER_TENSOR:
            raise NotImplementedError(
                f"{origin}: MLAAttentionSpec cache_dtype_str={cache_dtype_str!r} "
                f"with kv_quant_mode={_quant_mode_name(spec)} is an inconsistent "
                f"spec; the connector refuses to guess the scale layout"
            )
        return
    if mode in _QUANT_PER_TOKEN_HEAD or mode == _QUANT_NVFP4:
        raise NotImplementedError(
            f"{origin}: MLAAttentionSpec kv_quant_mode={_quant_mode_name(spec)} "
            f"keeps per-token-head scales in the page budget but outside the "
            f"transferable rows, and no MLA backend materializes them; "
            f"per-token-head / NVFP4 quantized MLA KV is not supported by "
            f"TairKvCacheConnector"
        )
    if cache_dtype_str in _PLAIN_CACHE_DTYPES:
        if mode != _QUANT_NONE:
            raise NotImplementedError(
                f"{origin}: MLAAttentionSpec cache_dtype_str={cache_dtype_str!r} "
                f"with kv_quant_mode={_quant_mode_name(spec)} is an inconsistent "
                f"spec (a non-quantized KV cache dtype must carry "
                f"kv_quant_mode=NONE); the connector refuses to guess the scale "
                f"layout"
            )
        return
    if cache_dtype_str in _PLAIN_FP8_CACHE_DTYPES:
        if mode != _QUANT_FP8_PER_TENSOR:
            raise NotImplementedError(
                f"{origin}: MLAAttentionSpec cache_dtype_str={cache_dtype_str!r} "
                f"with kv_quant_mode={_quant_mode_name(spec)} is an inconsistent "
                f"spec; the connector refuses to guess the scale layout"
            )
        if calculate_kv_scales:
            raise NotImplementedError(
                f"{origin}: cache_dtype_str={cache_dtype_str} with "
                f"calculate_kv_scales=True: the per-tensor scales are calibrated "
                f"at runtime and are not part of the cached bytes, so a reload "
                f"would decode with the wrong scale; re-run with "
                f"--calculate-kv-scales off or use bfloat16 KV"
            )
        return
    raise NotImplementedError(
        f"{origin}: MLAAttentionSpec cache_dtype_str={cache_dtype_str!r} is not a "
        f"layout this connector knows (expected {_KNOWN_CACHE_DTYPES})"
    )


def _check_attention_spec_supported(
    origin: str, spec: Any, *, calculate_kv_scales: bool
) -> None:
    """Gate one attention spec to the layouts the transfer path supports.

    SlidingWindowMLASpec comes first because it is *not* a FullAttentionSpec
    subclass: V4 keeps its attention sliding-window cache and its compressor
    state in those specs (the state is fp32, one row per window step), and
    both hold windowed or partial state, not the full prefix. V4 couples them
    with its compressed MLA group, so transferring the MLA group while
    skipping these would serve hits with uninitialized layers.

    MLA variants are decided by the spec's own fields (_check_mla_variant).
    The other FullAttentionSpec flavours keep the merged-window refusal: vLLM
    merges SWA / chunked-attention layers into FullAttentionSpec keeping
    sliding_window / attention_chunk_size set, and those blocks hold windowed
    KV, not the full prefix -- publishing them as prefix caches would corrupt
    reuse.
    """
    if SlidingWindowMLASpec is not None and isinstance(spec, SlidingWindowMLASpec):
        raise NotImplementedError(
            f"{origin}: SlidingWindowMLASpec sliding_window="
            f"{getattr(spec, 'sliding_window', None)} holds windowed or partial "
            f"state, not the full prefix, and DeepSeek V4 couples it with the "
            f"compressed MLA group; skipping it would leave those layers "
            f"uninitialized after a hit -- sliding-window / compressor-state MLA "
            f"KV is not supported by TairKvCacheConnector"
        )
    if not isinstance(spec, FullAttentionSpec):
        raise NotImplementedError(
            f"Unsupported kv cache spec {type(spec).__name__} in {origin}"
        )
    if isinstance(spec, MLAAttentionSpec):
        _check_mla_variant(origin, spec, calculate_kv_scales=calculate_kv_scales)
    for window_field in ("sliding_window", "attention_chunk_size"):
        if getattr(spec, window_field, None) is not None:
            raise NotImplementedError(
                f"{origin}: FullAttentionSpec has {window_field}="
                f"{getattr(spec, window_field)}; sliding-window / "
                f"chunked attention KV is not full-prefix and is "
                f"not yet supported by TairKvCacheConnector"
            )


def _compact_page_bytes(spec: Any, origin: str) -> int:
    """Raw KV bytes of one page: the compact page size, i.e. the bytes the
    gather kernel copies. spec.page_size_bytes returns page_size_padded when
    set, which includes an allocation-alignment tail the kernel never copies
    -- sizing locations/staging buffers with it would break the staging
    view() and waste storage."""
    page_bytes = getattr(spec, "real_page_size_bytes", None)
    if page_bytes is None:
        if getattr(spec, "page_size_padded", None) is not None:
            raise NotImplementedError(
                f"{origin}: page_size_padded={spec.page_size_padded} but this "
                f"vLLM exposes no real_page_size_bytes to recover the compact "
                f"page size; padded attention layouts are unsupported here"
            )
        page_bytes = spec.page_size_bytes
    return page_bytes


class _BucketKey(NamedTuple):
    """Transfer-shape key of one sub spec: layers can share a bucket only if
    their class, element type, page size and page stride all agree."""

    class_name: str
    dtype_name: str
    page_bytes: int
    # spec.page_size_padded as vLLM set it (None = no alignment padding).
    pad: Optional[int]


def _bucket_key(spec: Any, origin: str) -> _BucketKey:
    return _BucketKey(
        class_name=type(spec).__name__,
        dtype_name=str(getattr(spec, "dtype", None)),
        page_bytes=_compact_page_bytes(spec, origin),
        pad=getattr(spec, "page_size_padded", None),
    )


def _bucket_order(key: _BucketKey) -> Tuple[int, str, str, int]:
    """Total order over the buckets of one group: biggest page first (so the
    main layer of a packed group keeps the bare group name), then class,
    dtype and pad. Only spec-local fields enter, so the suffixes -- and with
    them the wire names -- are a pure function of the bucket *set*: layer
    registration order, or a different layer order on another rank, cannot
    move them."""
    return (
        -key.page_bytes,
        key.class_name,
        key.dtype_name,
        key.pad if key.pad is not None else -1,
    )


def _attention_meta(
    group_idx: int,
    layer_names: List[str],
    spec: Any,
    manager_block_size: int,
    *,
    suffix: str,
    origin: str,
) -> AttentionGroupMeta:
    """Size one attention bucket from its spec. The gate already refused
    compress_ratio > 1, so one token is one storage row and the per-token
    byte size is exact: compact_page_bytes / block_size."""
    page_bytes = _compact_page_bytes(spec, origin)
    if page_bytes % spec.block_size != 0:
        raise NotImplementedError(
            f"{origin}: compact page size {page_bytes} is not a multiple of "
            f"block_size {spec.block_size}; refusing to floor the per-token "
            f"byte size"
        )
    return AttentionGroupMeta(
        group_idx=group_idx,
        layer_names=list(layer_names),
        block_size=spec.block_size,
        per_block_bytes=(page_bytes // spec.block_size)
        * manager_block_size
        * len(layer_names),
        spec_suffix=suffix,
        page_bytes=page_bytes,
    )


def _uniform_group_metas(
    idx: int, wrapper: Any, manager_block_size: int, *, calculate_kv_scales: bool
) -> List[GroupMeta]:
    """Split one UniformTypeKVCacheSpecs group into transfer buckets.

    vLLM packs every MLA layer whose specs differ only in page size into one
    group sharing a single block table (V3.2: a 656 B/token main spec plus a
    132 B/token indexer spec), so the group maps to one manager location per
    bucket. Gateway first, bucket second: a refused sub spec refuses the whole
    instance (never a partial transfer with silently missing layers).
    """
    buckets: Dict[_BucketKey, List[Tuple[str, Any]]] = {}
    for layer_name, spec in wrapper.kv_cache_specs.items():
        origin = f"group {idx} (UniformTypeKVCacheSpecs, layer {layer_name})"
        _check_attention_spec_supported(
            origin, spec, calculate_kv_scales=calculate_kv_scales
        )
        buckets.setdefault(_bucket_key(spec, origin), []).append((layer_name, spec))
    ordered = sorted(buckets.items(), key=lambda item: _bucket_order(item[0]))
    return [
        _attention_meta(
            idx,
            [layer_name for layer_name, _ in layers],
            layers[0][1],
            manager_block_size,
            suffix="" if i == 0 else f"_b{i}",
            origin=f"group {idx} (UniformTypeKVCacheSpecs)",
        )
        for i, (_, layers) in enumerate(ordered)
    ]


def _parse_group(
    idx: int, group: Any, manager_block_size: int, *, calculate_kv_scales: bool
) -> List[GroupMeta]:
    """One vLLM kv_cache_group -> its transfer buckets (a plain group = one)."""
    spec = group.kv_cache_spec
    layers = list(group.layer_names)
    if isinstance(spec, MambaSpec):
        return [
            StateGroupMeta(
                group_idx=idx,
                layer_names=layers,
                block_size=spec.block_size,
                per_block_bytes=spec.page_size_bytes * len(layers),
                page_size_bytes=spec.page_size_bytes,
            )
        ]
    if UniformTypeKVCacheSpecs is not None and isinstance(
        spec, UniformTypeKVCacheSpecs
    ):
        return _uniform_group_metas(
            idx, spec, manager_block_size, calculate_kv_scales=calculate_kv_scales
        )
    origin = f"group {idx}"
    _check_attention_spec_supported(
        origin, spec, calculate_kv_scales=calculate_kv_scales
    )
    return [
        _attention_meta(idx, layers, spec, manager_block_size, suffix="", origin=origin)
    ]


def _check_unique_names(metas: List[GroupMeta]) -> None:
    """The manager keys locations by spec name, so two buckets sharing a name
    would silently merge (or overwrite each other). The rank prefix is not
    part of the identity: every rank registers the same name set."""
    names = [f"g{m.group_idx}{m.spec_suffix}" for m in metas]
    assert len(names) == len(set(names)), f"wire spec name collision: {names}"


def parse_groups(
    kv_cache_config: "KVCacheConfig",
    manager_block_size: int,
    *,
    calculate_kv_scales: bool,
) -> List[GroupMeta]:
    """Derive the transferable GroupMeta (bucket) list from vLLM's
    KVCacheConfig
    (https://github.com/vllm-project/vllm/blob/v0.26.0/vllm/v1/kv_cache_interface.py#L952:
    kv_cache_groups holds one KVCacheGroupSpec per block table, each with its
    kv_cache_spec -- FullAttentionSpec at L227, MambaSpec at L690).

    The input is the *calling process's* view of the config, and the roles do
    not see the same thing: vLLM folds every UniformTypeKVCacheSpecs group for
    the scheduler (generate_scheduler_kv_cache_config keeps one sub spec per
    group), so only the worker's view carries the bucket structure. The
    scheduler view must never feed the registered location specs -- see
    v1_connector: registration is worker-only.

    calculate_kv_scales is a mandatory keyword: defaulting it would let a
    runtime-calibrated scale layout pass as a plain fp8 one.
    """
    metas: List[GroupMeta] = []
    for idx, group in enumerate(kv_cache_config.kv_cache_groups):
        if getattr(group, "is_eagle_group", False):
            logger.warning(
                "skip eagle group %d (%d layers)", idx, len(group.layer_names)
            )
            continue
        metas.extend(
            _parse_group(
                idx, group, manager_block_size, calculate_kv_scales=calculate_kv_scales
            )
        )
    if not metas:
        # Every group was skipped (all-EAGLE config or an empty group list):
        # nothing to transfer, refuse explicitly instead of asserting.
        raise NotImplementedError("no usable kv cache groups (all groups skipped?)")
    if not any(isinstance(m, AttentionGroupMeta) for m in metas):
        # Pure-mamba / attention-free models have no attention KV to
        # transfer; the register_kv_caches path would fail obscurely later
        # (it requires an attention layer tensor). Refuse before init.
        raise NotImplementedError(
            "pure-mamba / attention-free models are not supported: "
            "TairKvCacheConnector transfers full-attention or hybrid "
            "(attention + mamba) KV caches only"
        )
    _check_unique_names(metas)
    return metas


def attn_kv_views(ref: torch.Tensor) -> tuple:
    """Normalize one attention layer's paged KV cache into per-pointer views.

    vLLM's flash_attn backend changed ``get_kv_cache_shape`` twice; the three
    layouts (see KVLayout for the per-version source links) are detected from
    the tensor shape itself (never from version strings):

    * 3-D ``(num_blocks, block, head_size)`` -- MLA latent cache
      (``num_kv_heads == 1``: one latent vector per token, no K/V split).
      One transfer pointer per layer.
    * 4-D ``(num_blocks, H, block, 2*D)``  -- K/V packed into the content dim
      (vLLM >= 0.26.0). One transfer pointer per layer.
    * 5-D ``(num_blocks, 2, block, H, D)`` -- N-first split K/V
      (vLLM 0.23.0 - 0.25.x). Two pointers per layer: ``t[:, 0]`` / ``t[:, 1]``.
    * 5-D ``(2, num_blocks, block, H, D)`` -- KV-first split K/V
      (vLLM <= 0.22.1). Two pointers per layer: ``t[0]`` / ``t[1]``.

    Returns ``(views, layout)``. Every view has the logical shape
    ``(num_blocks, kernel_block_size, heads, content_dim)`` matching the NHD
    memory order, so all downstream math (per-token dim, token-major check,
    block stride, data_ptr) is layout-independent -- the layout travels along
    only for traceability. Unrecognized layouts raise.
    """
    if ref.dim() == 3:
        # MLA: (num_blocks, block, head_size) is already token-major with a
        # single implicit head; unsqueeze gives the shared (n, b, 1, d) shape
        # without copying. Padded MLA pages (block stride != b*d) flow through
        # the same strided path as the split layouts.
        return [ref.unsqueeze(2)], KVLayout.MLA_3D
    if ref.dim() == 4:
        # Packed content dim; permute to token-major logical order. The permuted
        # view shares storage, data_ptr() is the storage base.
        return [ref.permute(0, 2, 1, 3)], KVLayout.PACKED_4D
    if ref.dim() == 5:
        kv_first = ref.shape[0] == 2
        n_first = ref.shape[1] == 2
        if kv_first and n_first:
            raise NotImplementedError(
                f"ambiguous kv layout {tuple(ref.shape)}: cannot tell the K/V "
                f"dim from a num_blocks dim of size 2"
            )
        if kv_first:
            return [ref[0], ref[1]], KVLayout.SPLIT_KV_5D_KV_FIRST
        if n_first:
            return [ref[:, 0], ref[:, 1]], KVLayout.SPLIT_KV_5D_N_FIRST
    raise NotImplementedError(
        f"unrecognized kv cache layout {tuple(ref.shape)}; expected the packed "
        f"4-D (vllm >= 0.26.0) or one of the split K/V 5-D layouts "
        f"(vllm <= 0.25.x)"
    )


def _hybrid_external_load_supported() -> Optional[bool]:
    """vLLM <= 0.22.x cannot combine mamba align mode with a KV connector:
    ``Scheduler._mamba_block_aligned_split`` asserts
    ``num_external_computed_tokens == 0`` ("External KV connector is not
    verified yet"), so the first external match would crash the scheduler.
    Probe the installed vLLM for that blocking assert (a capability check,
    not a version-string comparison).

    Returns:
        True   -- supported (assert absent, or the method was removed by a
                  newer vLLM: the assert went away with it);
        False  -- unsupported (the blocking assert is present);
        None   -- the method exists but its source is unavailable (frozen /
                  bytecode-only install), so the assert cannot be ruled out.
    """
    try:
        from vllm.v1.core.sched.scheduler import Scheduler

        method = Scheduler._mamba_block_aligned_split
    except (ImportError, AttributeError):
        # No such method: the blocking assert was removed/refactored away.
        return True
    try:
        import inspect

        src = inspect.getsource(method)
    except Exception:
        return None  # method exists but cannot be inspected
    return "External KV connector is not verified yet" not in src


def ensure_hybrid_supported(force: bool = False) -> None:
    """Fail fast with a clear message when a hybrid (mamba) model is served on
    a vLLM whose scheduler rejects external KV loads (vllm <= 0.22.x).

    When the probe is inconclusive (method present but source unavailable) the
    gate fails closed: a wrong guess would crash the scheduler on the first
    external match. ``force`` (extra_config ``force_hybrid_support``) bypasses
    the inconclusive case for source-restricted environments."""
    supported = _hybrid_external_load_supported()
    if supported:
        return
    if supported is None:
        if force:
            logger.warning(
                "force_hybrid_support=true: skipping the hybrid external-load "
                "capability probe; if this vLLM's scheduler still asserts "
                "'External KV connector is not verified yet' the first "
                "external match will crash it"
            )
            return
        raise NotImplementedError(
            "TairKvCacheConnector: cannot verify that this vLLM supports "
            "hybrid (mamba) models with an external KV connector -- "
            "Scheduler._mamba_block_aligned_split exists but its source is "
            "unavailable, so the vllm <= 0.22.x blocking assert cannot be "
            "ruled out. If you know this vLLM is >= 0.23.0, set "
            'kv_connector_extra_config {"force_hybrid_support": true} to '
            "bypass this check."
        )
    raise NotImplementedError(
        "TairKvCacheConnector: this vLLM version cannot combine hybrid "
        "(mamba) models with an external KV connector -- its scheduler "
        "asserts num_external_computed_tokens == 0 in "
        "_mamba_block_aligned_split ('External KV connector is not "
        "verified yet'). Upgrade to vLLM >= 0.23.0 for hybrid model "
        "support; full-attention models are unaffected."
    )
