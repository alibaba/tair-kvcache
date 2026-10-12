"""Shared vocabulary between the scheduler side and the worker side.

Everything here is role-agnostic: the data model both cores speak
(GroupMeta), the spec naming scheme shared with the manager
registration, the KV layout normalization, the hybrid capability gate and
the kv_cache_config parsing. The two cores (scheduler_core / worker_core)
and the thin connector shell (v1_connector) build on this module; nothing
here may import them.

A compressed MLA cache (tokens_per_state > 1, the DeepSeek V4.1 main/indexer
layers) stores one latent row per several tokens: vLLM addresses such a row by
(storage block, row within it) and never by token. The data model carries that
as RowGranularMeta -- tokens per row, rows per page, bytes per row -- plus the
two arithmetic preconditions the mapping needs (mbs % tokens_per_state == 0 so
a manager block stays a whole number of rows, block_size % tokens_per_state ==
0 so a block-table entry does too). vLLM renamed the field on the way to 0.31
(MLAAttentionSpec.compress_ratio -> AttentionSpec.tokens_per_state), so every
read goes through the era-safe _tokens_per_state below; the layouts this file
still refuses (fractional pooling, nvfp4_ds_mla, non-replayed windowed MLA) are
refused by name, not by accident.
"""

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, List, NamedTuple, Optional, Tuple

import torch

from vllm.v1.kv_cache_interface import (
    ChunkedLocalAttentionSpec,
    FullAttentionSpec,
    MLAAttentionSpec,
    MambaSpec,
    SinkFullAttentionSpec,
    SlidingWindowSpec,
)

try:
    # Optional compatibility symbols, one import statement each: vLLM 0.14
    # ships UniformTypeKVCacheSpecs but not KVQuantMode / SlidingWindowMLASpec,
    # and a shared statement would null all three on that era. The sentinels
    # are only ever used behind a null check (isinstance(x, None) would raise).
    from vllm.v1.kv_cache_interface import KVQuantMode  # ty: ignore[unresolved-import]
except ImportError:  # pragma: no cover - vllm 0.14 has no KVQuantMode
    KVQuantMode = None  # ty: ignore[invalid-assignment]
try:
    from vllm.v1.kv_cache_interface import SlidingWindowMLASpec  # ty: ignore[unresolved-import]
except ImportError:  # pragma: no cover - vllm 0.14 has no SlidingWindowMLASpec
    SlidingWindowMLASpec = None  # ty: ignore[invalid-assignment]
try:
    from vllm.v1.kv_cache_interface import UniformTypeKVCacheSpecs  # ty: ignore[unresolved-import]
except ImportError:  # pragma: no cover - older vLLM eras
    UniformTypeKVCacheSpecs = None  # ty: ignore[invalid-assignment]
try:
    from vllm.v1.kv_cache_interface import HiddenStateCacheSpec  # ty: ignore[unresolved-import]
except ImportError:  # pragma: no cover - vllm <= 0.21 has no HiddenStateCacheSpec
    HiddenStateCacheSpec = None  # ty: ignore[invalid-assignment]
try:
    from vllm.v1.kv_cache_interface import RSWASpec  # ty: ignore[unresolved-import]
except ImportError:  # pragma: no cover - vllm <= 0.25 has no RSWASpec
    RSWASpec = None  # ty: ignore[invalid-assignment]
try:
    # vllm 0.22 - 0.27 only: a FullAttentionSpec whose page size counts TQ slots
    # instead of head x dtype. Dropped in 0.28.
    from vllm.v1.kv_cache_interface import TQFullAttentionSpec  # ty: ignore[unresolved-import]
except ImportError:  # pragma: no cover - other vLLM eras
    TQFullAttentionSpec = None  # ty: ignore[invalid-assignment]
try:
    from vllm.v1.kv_cache_interface import CircularBufferSpec  # ty: ignore[unresolved-import]
except ImportError:  # pragma: no cover - vllm <= 0.28 has no compression ring
    CircularBufferSpec = None  # ty: ignore[invalid-assignment]

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
# fp8_e5m2 maps to FP8_PER_TENSOR and uint8 storage like the e4m3 variant
# (vLLM 0.26 lists it; the ROCm aiter backends accept it).
_PLAIN_FP8_CACHE_DTYPES = ("fp8", "fp8_e4m3", "fp8_e5m2")
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
#: The windowed twins: every registered spec minus the windowed ones, i.e. what
#: a block whose window was already freed still carries (see spec_group_plan).
NO_WINDOW_SPEC_GROUP = "full_no_window"
NO_WINDOW_ATTN_SPEC_GROUP = "attn_no_window"


def spec_name(tp_rank: int, meta: "GroupMeta") -> str:
    """Wire name for one (tp rank, transfer bucket).

    Buckets of one vLLM group share its group_idx (they share the block
    table) and differ by spec_suffix: "" for the first bucket, "_b{k}" for
    the rest. The name is a unique key into the manager's name -> bytes map
    and never parses back, so the suffixes are additive on the wire.
    """
    return f"tp{tp_rank}_g{meta.group_idx}{meta.spec_suffix}"


def windowed_block_has_data(
    windowed_metas: List["AttentionGroupMeta"],
    block_ids_per_group: List[List[int]],
    manager_block: int,
    manager_block_size: int,
) -> bool:
    """Does every windowed group hold real slots for this manager block?

    vLLM's windowed managers write the *null block* (id 0) into every block-table
    entry they free, so a manager block is describable by this one predicate:
    all of its group blocks are still real (the bytes are in HBM and readable)
    or some are null (the data does not exist any more -- reading the slots
    yields the shared zero page, never the block's KV).

    Both roles read it from the *same* snapshot: the scheduler in the loop
    iteration that announces the block (which fixes the announced spec coverage)
    and the worker in the same step's capture (which fixes what is really
    copied). Sharing the predicate is what keeps the announcement and the
    capture from ever disagreeing.
    """
    if not windowed_metas:
        return True
    first = manager_block * manager_block_size
    last = first + manager_block_size - 1
    for meta in windowed_metas:
        table = block_ids_per_group[meta.group_idx]
        group_block_size = meta.block_size
        for logical in range(first // group_block_size, last // group_block_size + 1):
            if logical >= len(table) or table[logical] == 0:
                return False
    return True


def spec_group_plan(
    group_metas: List["GroupMeta"], tp_size: int
) -> Tuple[List[dict], Dict[Tuple[bool, bool], str]]:
    """The declared location spec groups and the per-block selection table.

    A block's *coverage* is the pair (every state group holds a state, every
    windowed group holds window data); what a block really carries is the specs
    its coverage implies, and ``location_spec_group_names`` announces exactly
    that set per block. The table maps each reachable coverage to a declared
    group name.

    Why it exists: a windowed block whose data is already gone must not be
    published *as if* it had some -- the load side only trusts the manager's
    per-block spec coverage. The state dimension has always worked this way
    (attn/full); this adds the window dimension (``*_no_window`` variants), so
    a block without windowed data is announced with the specs it does have.

    A coverage missing from the table means no declared group describes it,
    which happens only when the instance has no non-windowed spec at all (there
    is nothing to announce for such a block, and nothing else can publish it
    either -- an all-abstain block is not published).
    """
    if not any(isinstance(meta, AttentionGroupMeta) for meta in group_metas):
        return [], {}
    windowed_specs = {
        spec_name(rank, meta)
        for rank in range(tp_size)
        for meta in group_metas
        if isinstance(meta, AttentionGroupMeta) and meta.retention is not None
    }
    state_specs = {
        spec_name(rank, meta)
        for rank in range(tp_size)
        for meta in group_metas
        if isinstance(meta, StateGroupMeta)
    }
    all_specs = {
        spec_name(rank, meta) for rank in range(tp_size) for meta in group_metas
    }
    attn_specs = all_specs - state_specs
    no_window = all_specs - windowed_specs
    no_window_attn = attn_specs - windowed_specs
    if not windowed_specs and not state_specs:
        # Nothing sparse: no groups at all, so single-group and hybrid-free
        # models keep byte-identical requests (and old managers keep working).
        return [], {}
    declared: List[dict] = []

    def declare(name: str, names: set) -> None:
        if names:
            declared.append({"name": name, "spec_names": sorted(names)})

    # Once a block's coverage is announced per block, the "has everything"
    # group is needed whatever dimensions are sparse; the state dimension adds
    # its attention-only twin, the window dimension its no-window variants
    # (whose sets can be empty -- a group is declared only when it names specs,
    # and a coverage with no declareable group is one that carries nothing, so
    # nothing gets published for it either).
    declare(ALL_SPEC_GROUP, all_specs)
    if state_specs:
        declare(ATTN_ONLY_SPEC_GROUP, attn_specs)
    if windowed_specs:
        declare(NO_WINDOW_SPEC_GROUP, no_window)
        if state_specs:
            declare(NO_WINDOW_ATTN_SPEC_GROUP, no_window_attn)
    table: Dict[Tuple[bool, bool], str] = {}
    for states_complete in (True, False):
        for window_present in (True, False):
            if window_present:
                name = ALL_SPEC_GROUP if states_complete else ATTN_ONLY_SPEC_GROUP
            else:
                name = (
                    NO_WINDOW_SPEC_GROUP
                    if states_complete
                    else NO_WINDOW_ATTN_SPEC_GROUP
                )
            if any(group["name"] == name for group in declared):
                table[(states_complete, window_present)] = name
    return declared, table


def _group_coverage(meta: "GroupMeta") -> Tuple[Any, bool]:
    """The coverage dimensions one vLLM group's folded view can lose.

    The engine hands the scheduler one sub spec per UniformTypeKVCacheSpecs
    group (generate_scheduler_kv_cache_config keeps the first member), so the
    scheduler's masks are built from that one member: a window retention and a
    recurrent state are the two dimensions a bucket is transferred under that
    the fold can drop. Everything else (page size, geometry, tokens per row) is
    the *worker's* business and folds harmlessly.

    The state half is defensive -- vLLM's uniform bases keep mamba specs in
    their own group today -- but the predicate is total on purpose.
    """
    return (getattr(meta, "retention", None), isinstance(meta, StateGroupMeta))


def _check_coverage_survives_folding(group_metas: List["GroupMeta"]) -> None:
    """Refuse a vLLM group whose buckets disagree about their coverage.

    Buckets that agree are safe: the folded member carries every dimension the
    group has, so the scheduler's window / state masks describe exactly what the
    worker's buckets hold. Buckets that disagree (a windowed R-SWA layer packed
    with a full MLA one, or two windows of different widths) are not: the
    scheduler computes its masks from the one member it can see, so it would
    report window data (or a state) as present while the worker's masking knows
    it is gone. The observable failure depends on which side the folded member
    lands on: at least one path is loud (a windowed bucket whose capture was
    withheld fails its announced block instead of publishing it), but the
    other direction can reach the manager with stale or zero-page bytes, so
    the combination is refused at registration, before any request, rather
    than measured case by case. A packed group whose buckets do agree
    (V4.1's compressed group, which declares nothing at all) is untouched.
    """
    seen: Dict[int, Tuple[Any, bool]] = {}
    for meta in group_metas:
        coverage = _group_coverage(meta)
        previous = seen.setdefault(meta.group_idx, coverage)
        if previous != coverage:
            raise NotImplementedError(
                f"vLLM group {meta.group_idx} carries transfer buckets with "
                f"different coverage: {previous} and {coverage} "
                f"(layer {meta.layer_names[0]!r}). The engine folds a packed "
                f"group to one member spec for the scheduler, whose window and "
                f"state masks would then miss this bucket's coverage and let "
                f"the manager store stale or zero-page bytes as cacheable. "
                f"TairKvCacheConnector cannot serve this combination"
            )


def build_spec_groups(group_metas: List["GroupMeta"], tp_size: int) -> List[dict]:
    """LocationSpecGroups describing which specs a block may carry.

    Two kinds of sparsity make a block's spec coverage narrower than the
    instance's full set:

    * hybrid (mamba "align") models materialize a recurrent state only at
      segment boundaries, so interior manager blocks have attention KV but no
      state (``full`` / ``attn``);
    * windowed models free the blocks outside the window while the request
      lives, so a block saved late carries no windowed KV at all
      (``*_no_window``, see spec_group_plan).

    Declaring the groups lets ``start_write_cache`` say, per block, which specs
    that block will really hold. The manager then stores exactly the advertised
    specs, reports the real per-block coverage in ``getCacheLocation``, and
    later lets a complementary write fill in a block's missing specs.

    Models with nothing to be sparse about declare no groups at all, which
    keeps their requests byte-identical to before (and compatible with managers
    that predate spec groups).

    A group whose buckets *disagree* about coverage is refused outright
    (``_check_coverage_survives_folding``): the engine folds the group to one
    member for the scheduler, so a disagreement would mask the wrong blocks.
    Buckets that agree fold without loss -- including a packed group, whose
    extra buckets keep both their registration *and* their announcement name
    (the group's member list comes from the worker's registration, not from the
    scheduler's folded view). A packed group with no sparsity declares nothing
    at all, which leaves the manager on the worker's full registration: V4.1's
    compressed group runs exactly that path.
    """
    _check_coverage_survives_folding(group_metas)
    return spec_group_plan(group_metas, tp_size)[0]


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


#: Retention kinds of a windowed attention bucket (see WindowRetention.kind).
RETENTION_SLIDING = "sliding"
RETENTION_CHUNKED_LOCAL = "chunked_local"
RETENTION_RSWA_GAP = "rswa_gap"


@dataclass(frozen=True)
class WindowRetention:
    """What a windowed attention group keeps, and therefore what the connector
    must copy eagerly and must never restore into a freed slot.

    vLLM's windowed managers free the blocks outside the window while the
    request is alive (``remove_skipped_blocks``) and write the *null block* into
    their block-table slots, so those positions are neither a token prefix nor
    readable: a block's bytes exist only in the pass that produced it (eager
    capture) and a null slot must never be written (the null block is the
    shared zero page). The window the connector has to be able to serve is:

    * ``sliding`` (SlidingWindowSpec): [H - sliding_window, H), extended by
      ``extra_retained_tokens`` on the trailing edge (spec decoding);
    * ``chunked_local`` (ChunkedLocalAttentionSpec): the chunk containing H;
    * ``rswa_gap`` (RSWASpec): [H - rswa_window, H) plus the whole prefill, with
      only the gap in between freed.
    """

    kind: str
    # spec.sliding_window / spec.attention_chunk_size / spec.rswa_window.
    window_tokens: int
    # spec.extra_retained_tokens: trailing tokens kept (but not attended) so a
    # multi-module spec-decoding store lag can still rebuild the window.
    extra_retained_tokens: int = 0


@dataclass(frozen=True)
class RowGranularMeta:
    """How a compressed attention bucket's rows map onto tokens and pages.

    vLLM stores one latent row per ``tokens_per_row`` tokens when the spec's
    tokens_per_state (``compress_ratio`` on vllm <= 0.28) is above 1 -- the
    DeepSeek V4.1 main and indexer layers -- and addresses such a row by
    (storage block, row within it), never by token. One kernel block (SM90: 64
    tokens, SM100 family: 128) is one page, so a page holds ``rows_per_page``
    rows of ``bytes_per_row`` bytes each (per layer).

    The two arithmetic preconditions are asserted where the meta is built: the
    manager block size must be a whole number of rows (mbs % tokens_per_row)
    and so must a block-table entry (block_size % tokens_per_row, checked by
    the gate), or the block table would address half a row.
    """

    # spec.tokens_per_state / spec.compress_ratio (validated int >= 2).
    tokens_per_row: int
    # Rows in one compact page (vLLM's num_states): spec.block_size // tps.
    rows_per_page: int
    # Compact bytes of one row of one layer (spec's num_heads x state cell).
    bytes_per_row: int


@dataclass(frozen=True)
class AttentionGroupMeta(GroupMeta):
    """Attention bucket: token-granular KV, re-blockable to the manager block
    size. Sizing derives from the *compact* page size (see parse_groups)."""

    # Compact page bytes (spec.real_page_size_bytes): the alignment tail of a
    # padded page is never copied. Keyword-only because the base class already
    # carries the defaulted spec_suffix.
    page_bytes: int = field(kw_only=True)
    # None = the bucket's blocks hold the whole token prefix and are never
    # freed while the request lives (the contract the transfer path had before
    # windowed groups arrived). Not None = the windowed contract above; every
    # windowed-only branch (eager capture, null disposition, hit clamping)
    # keys on this field so full-attention groups keep their exact old path.
    retention: Optional[WindowRetention] = field(default=None, kw_only=True)
    # None = one storage row per token (every bucket before V4.1). Not None =
    # the compressed contract: the address math and the staging geometry work
    # in rows, not tokens (see RowGranularMeta).
    row_granular: Optional[RowGranularMeta] = field(default=None, kw_only=True)

    # The page element type (spec.dtype): one instance may mix them (V3.2 bf16
    # main + uint8 indexer), and the staging pool keys by it, so the bucket, not
    # the instance, is the granularity. Equals the kv_cache tensor dtype by
    # construction: both come from kv_cache_dtype_str_to_dtype on the same
    # cache_dtype (gpu_model_runner for the tensors, the attention layer for the
    # spec). None only for a spec object that carries no dtype at all.
    dtype: Any = None


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


def _has_quant_mode(spec: Any) -> bool:
    """vLLM 0.14 specs predate kv_quant_mode: the dtype string alone defines
    the layout there, so the (cds, mode) consistency rules cannot apply."""
    return hasattr(spec, "kv_quant_mode")


def _quant_mode_name(spec: Any) -> str:
    """Readable kv_quant_mode for messages: the enum's name where the era has
    one, else the canonical name of the raw value (stub / plain int)."""
    mode = getattr(spec, "kv_quant_mode", _QUANT_NONE)
    name = getattr(mode, "name", None)
    if isinstance(name, str):
        return name
    return _QUANT_MODE_NAMES.get(_quant_mode(spec), f"mode {_quant_mode(spec)}")


def _tokens_per_state(spec: Any) -> Tuple[Any, str]:
    """Tokens covered by one stored row, plus the field that carried it.

    vLLM <= 0.28 spells the compression as ``MLAAttentionSpec.compress_ratio``;
    from 0.29 it is the AttentionSpec-wide ``tokens_per_state`` (int, or a
    Fraction < 1 for pooled states). An engine spec carries exactly one of the
    two and both default to 1, so reading *both* and trusting whichever is not
    the default is the era-safe read: a compression can never hide behind the
    era's field name. A spec that carries both (only a test double) is read
    from ``tokens_per_state`` first -- the newer spelling.

    Missing attributes read as 1: vLLM 0.14 predates both fields and has no
    compressed MLA either. The value is used by the gate (validated) and by the
    bucketing (one geometry per (page, head geometry, tokens per row) triple).
    """
    tokens_per_state = getattr(spec, "tokens_per_state", 1)
    if tokens_per_state != 1:
        return tokens_per_state, "tokens_per_state"
    compress_ratio = getattr(spec, "compress_ratio", 1)
    if compress_ratio != 1:
        return compress_ratio, "compress_ratio"
    return 1, "tokens_per_state"


def _mla_layout_triple(spec: Any) -> str:
    """The (cache_dtype_str, geometry, tokens_per_state) triple of a refusal
    message: what the layout *is*, not merely that it is unknown."""
    tokens_per_state, field = _tokens_per_state(spec)
    return (
        f"cache_dtype_str={getattr(spec, 'cache_dtype_str', None)!r}, "
        f"head_size={getattr(spec, 'head_size', None)}, "
        f"block_size={getattr(spec, 'block_size', None)}, "
        f"{field}={tokens_per_state}"
    )


def _check_mla_structure(origin: str, spec: "MLAAttentionSpec") -> Tuple[int, str]:
    """Refuse the tokens-per-state / model_version combinations vLLM's own size
    algebra cannot represent, reading raw fields only (never
    storage_block_size / real_page_size_bytes / page_size_bytes): a fractional
    ratio stores more states than tokens (and divides by zero on the way), and a
    ratio that does not divide block_size would be floored silently. Returns the
    validated (tokens per stored row, the field it came from).

    ``tokens_per_state > 1`` is a *supported* layout (see RowGranularMeta), so
    the only refusals left here are ratios no row granularity can describe.
    """
    tokens_per_state, field = _tokens_per_state(spec)
    if isinstance(tokens_per_state, bool) or not isinstance(tokens_per_state, int):
        # whisper_causal's Fraction(1, block_pool_size) pools several states
        # into one token: the row unit is smaller than a token, so neither the
        # manager block nor the block table stays a whole number of rows.
        raise NotImplementedError(
            f"{origin}: MLAAttentionSpec {field}={tokens_per_state} must be a "
            f"whole number >= 1 of tokens per stored row (a fractional ratio "
            f"pools states the row-granular transfer model cannot address); "
            f"refusing to size it ({_mla_layout_triple(spec)})"
        )
    if tokens_per_state < 1:
        raise NotImplementedError(
            f"{origin}: MLAAttentionSpec {field}={tokens_per_state} is "
            f"invalid (must be >= 1); refusing to size the KV cache from it"
        )
    if tokens_per_state > 1 and spec.block_size % tokens_per_state:
        raise NotImplementedError(
            f"{origin}: MLAAttentionSpec block_size={spec.block_size} is not "
            f"divisible by {field}={tokens_per_state}; refusing to "
            f"silently floor the storage layout"
        )
    model_version = getattr(spec, "model_version", None)
    if model_version not in (None, "deepseek_v4"):
        raise NotImplementedError(
            f"{origin}: MLAAttentionSpec model_version={model_version!r} is not "
            f'a layout this connector knows (expected None or "deepseek_v4")'
        )
    return tokens_per_state, field


def _check_mla_variant(
    origin: str, spec: "MLAAttentionSpec", *, calculate_kv_scales: bool
) -> None:
    """Gate one MLAAttentionSpec to the latent layouts the transfer path
    supports, by the (cache_dtype_str, kv_quant_mode) pair, after the
    structural check (_check_mla_structure) that guarantees the ratio is
    usable.

    Accepted (self-contained bytes; one row per token, or one row per
    ``tokens_per_state`` tokens in the row-granular case, see
    RowGranularMeta):

    * "fp8_ds_mla" (DeepSeek V3.2/V4 main layers): a packed 656/584 B layout,
      compressed or not, whose fp8 scale lives *inside* each row, so a verbatim
      round trip keeps it. kv_quant_mode is FP8_PER_TENSOR there -- that alone
      must not refuse it (a naive quant gate would make every real V3.2 main
      layer unreachable);
    * None/auto/float16/bfloat16 with kv_quant_mode NONE: the plain latent
      cache (GLM, indexer layers, bf16 V3.2 main);
    * "fp8"/"fp8_e4m3"/"fp8_e5m2" with kv_quant_mode FP8_PER_TENSOR:
      layer-level scales live outside the cached bytes, so the round trip is
      exact -- but only while vLLM does not *calibrate* those scales at
      runtime (calculate_kv_scales): a reload would then decode with a stale
      scale. vLLM 0.14 specs carry no kv_quant_mode field at all; there the
      dtype string alone defines the layout, so the consistency rules are
      skipped, not failed.

    Refused by name: "nvfp4_ds_mla" (the SM100-only NVFP4 record layout, 288 B
    per row for the V4.1 record and 352 B for the V3.2 one -- a layout the
    connector would have to size from a page algebra it does not implement),
    per-token-head quant modes (their scales sit in the page budget but outside
    the transferred rows), unknown cache_dtype_str, and (cds, kv_quant_mode)
    pairs vLLM cannot produce -- guessing a scale layout is worse than
    refusing.
    """
    _check_mla_structure(origin, spec)
    cache_dtype_str = getattr(spec, "cache_dtype_str", None)
    mode = _quant_mode(spec)
    if cache_dtype_str == "nvfp4_ds_mla":
        raise NotImplementedError(
            f"{origin}: MLAAttentionSpec is the SM100-only NVFP4 record layout "
            f"({_mla_layout_triple(spec)}): its rows are an NVFP4 cell with an "
            f"in-row scale, 288 B for the V4.1 record and 352 B for the V3.2 "
            f"one, which the connector neither sizes nor validates; "
            f"nvfp4_ds_mla KV is not supported by TairKvCacheConnector"
        )
    if cache_dtype_str == "fp8_ds_mla":
        if _has_quant_mode(spec) and mode != _QUANT_FP8_PER_TENSOR:
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
        if _has_quant_mode(spec) and mode != _QUANT_FP8_PER_TENSOR:
            raise NotImplementedError(
                f"{origin}: MLAAttentionSpec cache_dtype_str={cache_dtype_str!r} "
                f"with kv_quant_mode={_quant_mode_name(spec)} is an inconsistent "
                f"spec; the connector refuses to guess the scale layout"
            )
        # MLA-only scope (r1/S5): non-MLA full-attention fp8 KV has the same
        # calibration risk but does not pass through this gate; flagging it in
        # the PR body instead of widening this connector change.
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


#: Spec classes whose blocks hold the whole token prefix: vLLM allocates them
#: for every token of the request and never frees or rewrites one while the
#: request lives, so a verbatim token-prefix cache is exactly their content.
#:
#: * FullAttentionSpec -- including a *promoted* local-attention layer, i.e.
#:   sliding_window / attention_chunk_size left set by
#:   kv_cache_utils._promote_local_kv_cache_specs: its two callers are
#:   unify_hybrid_kv_cache_specs (--disable-hybrid-kv-cache-manager) and
#:   _try_get_full_allocation_fallback_groups (the allocation-only fallback
#:   taken when page sizes cannot be unified, with the hybrid manager *on*).
#:   Both allocate for all tokens and let the window narrow only the compute.
#:   That is the documented escape hatch for windowed models (GPU KV is not
#:   saved by it).
#: * SinkFullAttentionSpec (OpenPangu) -- the sink K/V is a per-layer constant
#:   that vLLM populates into every request's own blocks (maybe_populate_sink),
#:   so the blocks are the full prefix and the byte round trip is lossless.
#: * TQFullAttentionSpec (vllm 0.22 - 0.27) -- a page-size-only variant.
#:
#: Matched by *exact* type: a class this file has not been taught is refused
#: even when it inherits from one of these (a subclass is a new claim about
#: layout and retention -- RSWASpec itself is exactly such a subclass). The
#: same rule governs _MLA_SPEC_TYPES below. A user-registered custom spec
#: subclass (see vllm.v1.kv_cache_spec_registry) is refused for the same
#: reason: it must be declared here deliberately first.
#:
#: Era-safe: a class this vLLM lacks reads as None and drops out of the tuple.
_FULL_PREFIX_SPEC_TYPES: Tuple[type, ...] = tuple(
    cls
    for cls in (FullAttentionSpec, SinkFullAttentionSpec, TQFullAttentionSpec)
    if cls is not None
)

#: The accepted MLA-family classes: the latent layout itself, plus the
#: extract_hidden_states marker, which adds no layout of its own. Measured
#: across the vllm 0.14 - 0.31 trees (envs/): HiddenStateCacheSpec is the only
#: MLAAttentionSpec subclass any era ships, so exact-type matching refuses
#: nothing a real engine produces.
_MLA_SPEC_TYPES: Tuple[type, ...] = tuple(
    cls for cls in (MLAAttentionSpec, HiddenStateCacheSpec) if cls is not None
)

#: Scratch classes the *engine* rebuilds per request, so a connector may drop
#: them even when vLLM advertises them for transfer (see parse_groups):
#:
#: CircularBufferSpec -- "one block per request holding the raw keys of the
#: token group that is still being compressed" (0.31 docstring), claimed for
#: the request's lifetime by CircularBufferManager ("Claims the ring's single
#: block per request; prefix caching disabled"), i.e. the compression machinery
#: refills it every request. An external hit restores the *main* MLA group's
#: prefix rows and the compressor rebuilds the ring around them.
#:
#: KpoolTailSpec is deliberately NOT here: same manager family, but its data
#: path has not been shown to be rebuilt per request, so it stays refused when
#: transferable (fail-closed).
#:
#: Matched by *exact* type, like every other acceptance rule here: a subclass of
#: the ring is a new claim about the block's content and gets no exemption.
_COMPRESSION_SCRATCH_TYPES: Tuple[type, ...] = tuple(
    cls for cls in (CircularBufferSpec,) if cls is not None
)


def _retention_of(spec: Any) -> Optional[WindowRetention]:
    """The window retention of an already-vetted attention spec.

    Only the three windowed classes carry one -- and only as their *exact*
    type, for the same reason the accept set is exact-type matched (an
    undeclared subclass is a new claim about what stays alive). Called by the
    gate's accept branches and by the parse path, so both agree by
    construction.
    """
    if type(spec) is SlidingWindowSpec:
        return WindowRetention(
            kind=RETENTION_SLIDING,
            window_tokens=spec.sliding_window,
            extra_retained_tokens=getattr(spec, "extra_retained_tokens", 0) or 0,
        )
    if (
        ChunkedLocalAttentionSpec is not None
        and type(spec) is ChunkedLocalAttentionSpec
    ):
        return WindowRetention(
            kind=RETENTION_CHUNKED_LOCAL,
            window_tokens=spec.attention_chunk_size,
        )
    if RSWASpec is not None and type(spec) is RSWASpec:
        return WindowRetention(kind=RETENTION_RSWA_GAP, window_tokens=spec.rswa_window)
    return None


def _check_attention_spec_supported(
    origin: str, spec: Any, *, calculate_kv_scales: bool
) -> Optional[WindowRetention]:
    """Accept or refuse one attention spec, by *class* rather than by field,
    and return how the accepted bucket retains its blocks.

    The rules below cover the whole attention zoo of vllm 0.14 - 0.31 in the
    order they must be tested (a subclass before its parent). The field
    whitelist this replaces (sliding_window / attention_chunk_size) is exactly
    how a spec that keeps its window under another name stayed accepted:
    RSWASpec.rswa_window is not in it, so an R-SWA instance passed the gate and
    only failed much later -- or silently published freed gap blocks.

    Refused, because their blocks stop being the full token prefix while the
    request is alive (vLLM frees or rewrites them) *and* no data-plane
    mechanism for them exists yet:

    * a windowed/replayed MLA layout that vLLM does *not* rebuild itself
      (SlidingWindowMLASpec with bounded_replay off): it needs the windowed
      retention contract, which this connector implements for the plain
      windowed classes only. The replayed form never reaches this gate -- it is
      skipped while parsing, because the engine regenerates those rows;
    * an undeclared subclass of *any* accepted class, windowed ones included;
    * MLAAttentionSpec with a window field: only reachable through the promote
      path, which drops the replay policy; its retention semantics belong to
      the windowed-MLA work.

    Accepted with a retention (the windowed contract, see WindowRetention --
    these three carry the eager-capture / null-disposition / hit-clamp
    machinery this file's users implement):

    * SlidingWindowSpec -> ``sliding``;
    * ChunkedLocalAttentionSpec -> ``chunked_local``;
    * RSWASpec -> ``rswa_gap`` (the hole of vllm <= 0.31's field whitelist).

    Accepted with ``None`` retention (full prefix, unchanged contract): the
    full-prefix classes (_FULL_PREFIX_SPEC_TYPES) and the MLA latent layouts
    (_MLA_SPEC_TYPES, which includes the HiddenStateCacheSpec marker of
    extract_hidden_states -- it changes no layout of its own), compressed
    (tokens_per_state > 1, or compress_ratio on vllm <= 0.28) ones included:
    their rows are whole and the row-granular mapping is exact (RowGranularMeta).

    Acceptance is by *exact* type, never by family: a subclass of an accepted
    class is a fresh claim about byte layout and retention (RSWASpec is
    precisely such a subclass) and is refused until this file names it. That is
    also why the window test below is no longer the only line of defence -- a
    spec whose window hides under a new field name cannot reach any accept
    branch at all.

    Everything else -- CrossAttentionSpec, EncoderOnlyAttentionSpec, and any
    class this file has not been taught -- fails closed. The group-level
    receipts are matched as strictly in _parse_group: an undeclared
    MambaSpec / UniformTypeKVCacheSpecs subclass reaches this gate too and is
    refused here by name.
    """
    if SlidingWindowMLASpec is not None and isinstance(spec, SlidingWindowMLASpec):
        raise NotImplementedError(
            f"{origin}: SlidingWindowMLASpec sliding_window="
            f"{getattr(spec, 'sliding_window', None)} with bounded_replay="
            f"{bool(getattr(spec, 'bounded_replay', False))} holds windowed or "
            f"partial state, not the full prefix, and vLLM does not rebuild "
            f"those rows on a hit: it needs the windowed retention contract, "
            f"which this connector implements for the plain windowed classes "
            f"only. The replayed form (bounded_replay on, the 0.31 default) is "
            f"skipped while parsing instead -- sliding-window / compressor-state "
            f"MLA KV is not supported by TairKvCacheConnector in the "
            f"non-replayed form"
        )
    if isinstance(spec, SlidingWindowSpec):
        if type(spec) is not SlidingWindowSpec:
            # KpoolTailSpec (a ring tail, skipped before the gate when the era
            # declares it is not cacheable) and anything user-registered: a
            # windowed subclass is a new retention claim, so it must be named
            # here deliberately first -- the RSWA lesson.
            raise NotImplementedError(
                f"{origin}: {type(spec).__name__} sliding_window="
                f"{getattr(spec, 'sliding_window', None)} is a SlidingWindowSpec "
                f"subclass this connector has not been taught; it transfers the "
                f"windowed layouts of {SlidingWindowSpec.__name__} only"
            )
        return _retention_of(spec)
    if ChunkedLocalAttentionSpec is not None and isinstance(
        spec, ChunkedLocalAttentionSpec
    ):
        if type(spec) is not ChunkedLocalAttentionSpec:
            raise NotImplementedError(
                f"{origin}: {type(spec).__name__} attention_chunk_size="
                f"{getattr(spec, 'attention_chunk_size', None)} is a "
                f"{ChunkedLocalAttentionSpec.__name__} subclass this connector "
                f"has not been taught"
            )
        return _retention_of(spec)
    if RSWASpec is not None and isinstance(spec, RSWASpec):
        if type(spec) is not RSWASpec:
            raise NotImplementedError(
                f"{origin}: {type(spec).__name__} rswa_window="
                f"{getattr(spec, 'rswa_window', None)} is an RSWASpec subclass "
                f"this connector has not been taught"
            )
        return _retention_of(spec)
    if isinstance(spec, MLAAttentionSpec):
        if type(spec) not in _MLA_SPEC_TYPES:
            # HiddenStateCacheSpec is the only declared one (a marker that adds
            # no layout); anything else is a new layout/retention claim -- the
            # RSWA shape one level down -- and must be named here first.
            raise NotImplementedError(
                f"{origin}: {type(spec).__name__} is an MLAAttentionSpec subclass "
                f"this connector has not been taught; it transfers the declared "
                f"latent layouts only ({', '.join(cls.__name__ for cls in _MLA_SPEC_TYPES)})"
            )
        # HiddenStateCacheSpec deliberately falls through the same checks as
        # MLAAttentionSpec: the marker changes no layout, so the window test and
        # the latent-layout gate apply to it verbatim (it is accepted because
        # extract_hidden_states writes plain latent bytes).
        for window_field in ("sliding_window", "attention_chunk_size"):
            if getattr(spec, window_field, None) is not None:
                raise NotImplementedError(
                    f"{origin}: {type(spec).__name__} has {window_field}="
                    f"{getattr(spec, window_field)}; a windowed MLA layout "
                    f"holds windowed or replayed state, not the full "
                    f"prefix, and is not supported by "
                    f"TairKvCacheConnector yet"
                )
        _check_mla_variant(origin, spec, calculate_kv_scales=calculate_kv_scales)
        return None
    if type(spec) in _FULL_PREFIX_SPEC_TYPES:
        tokens_per_state, field = _tokens_per_state(spec)
        if tokens_per_state != 1:
            # Pooled / compressed *full attention*: one row holds several states
            # per token (Whisper's block pooling sets Fraction(1, n)) or spans
            # several tokens. _attention_meta sizes one row per token, so the
            # page would not match the tensor and registration fails much later
            # with a bare block-size assertion -- refuse here, by field name.
            # MLA keeps its own structural check above (that one will learn to
            # transfer multi-token rows; this model does not).
            raise NotImplementedError(
                f"{origin}: {type(spec).__name__} {field}={tokens_per_state} is "
                f"not one state per token; the transfer model sizes one row per "
                f"token, so pooled or compressed full-attention KV would be "
                f"mis-sized. Refusing the instance (not supported by "
                f"TairKvCacheConnector yet)"
            )
        return None
    raise NotImplementedError(
        f"Unsupported kv cache spec {type(spec).__name__} in {origin}: "
        f"TairKvCacheConnector serves token-prefix KV (full attention, windowed "
        f"attention, MLA latent, and hybrid attention + mamba state) only; "
        f"there is no transfer contract for this class"
    )


def _is_scratch_spec(spec: Any) -> bool:
    """Is this group's KV per-request scratch that no prefix-cache hit can serve?

    vLLM's ring-buffer and host-tier caches (CircularBufferSpec, KpoolTailSpec,
    HiSparseHot/ResidentSpec) hold a page or a raw buffer per request rather
    than token-addressed prefix KV, and they say so: ``prefix_cacheable`` is
    False (vLLM never hashes their blocks, so no hit is ever expected for them)
    *and* the spec either takes no slot mapping row or keeps no per-layer views
    for a connector to transfer. Nothing in them is a token prefix, so the
    whole group is skipped -- the shape of an EAGLE drafter group -- instead of
    refusing the instance that carries it.

    Both halves matter: a *cacheable* spec must never be skipped (that would be
    a silent hole), and a spec that is not cacheable but is still transferable
    through the ordinary contract is refused by the gate rather than dropped
    (SlidingWindowMLASpec with bounded_replay off).

    The flags arrived with the specs they describe (prefix_cacheable and
    CircularBufferSpec in 0.29, uses_slot_mapping / has_layer_views and the
    HiSparse / kpool-tail caches in 0.30), so an older era is missing both the
    flag and the spec it would judge: a missing attribute reads as the
    permissive default and nothing is skipped.
    """
    if getattr(spec, "prefix_cacheable", True):
        return False
    return not getattr(spec, "uses_slot_mapping", True) or not getattr(
        spec, "has_layer_views", True
    )


def _is_replayed_window(spec: Any) -> bool:
    """Does vLLM itself rebuild this group's rows on a prefix-cache hit?

    A SlidingWindowMLASpec with ``bounded_replay`` (the 0.31 default, which is
    what DeepSeek V4.1 configures) keeps no reusable window state: the engine
    replays the window tokens and the forward pass regenerates the rows, and it
    pads exactly those slots in the prefix-cacheable groups. The two flags say
    the same thing (``prefix_replay_tokens`` is the window it replays), and both
    default to "not replayed" -- so an era without them refuses the spec instead
    of skipping it (fail-closed, vllm 0.26 has the class but not the policy).
    """
    if not (
        SlidingWindowMLASpec is not None and isinstance(spec, SlidingWindowMLASpec)
    ):
        return False
    return bool(getattr(spec, "prefix_replay_tokens", 0)) or bool(
        getattr(spec, "bounded_replay", False)
    )


def _skip_reason(spec: Any) -> Optional[str]:
    """Why this group's KV is not transferable prefix data (None = transfer it).

    Two shapes of "no prefix cache can serve this", both of which make vLLM
    keep the group out of its own reuse machinery:

    * the engine rebuilds the rows (a replayed window, see _is_replayed_window);
    * per-request scratch (see _is_scratch_spec).

    A wrapper (UniformTypeKVCacheSpecs) is judged member by member: skipping it
    drops *every* member, so a group with even one cacheable, transferable
    member must go through the gate (which then names the refused member)
    rather than disappear.

    The returned string is the operator-facing log line; the *decision* is the
    absence of a string, so the two cannot drift.
    """
    members = (
        list(spec.kv_cache_specs.values())
        if UniformTypeKVCacheSpecs is not None
        and isinstance(spec, UniformTypeKVCacheSpecs)
        else [spec]
    )
    if members and all(_is_replayed_window(member) for member in members):
        return (
            "vLLM replays this window on a hit and rebuilds the group's rows "
            "(SlidingWindowMLASpec bounded_replay): the bytes are neither "
            "stable nor needed"
        )
    if _is_scratch_spec(spec):
        return "per-request scratch, no prefix cache to serve from it"
    return None


def _is_engine_rebuilt_scratch(spec: Any) -> bool:
    """Is this engine-rebuilt scratch, i.e. may a *transferable* group be dropped?

    The one proven case is the compression ring: CircularBufferSpec (and a
    UniformTypeKVCacheSpecs whose members are all rings, the shape V4.1 packs)
    holds the raw keys of the in-flight token group for one request, and the
    compression machinery refills it there -- an external hit restores the main
    MLA group and the ring is rebuilt around it. Everything else that is
    scratch *and* transferable is refused instead (see parse_groups): dropping
    a group the engine hands to connectors and does *not* rebuild would resume
    an external hit with uninitialized state.

    Exact types again, wrapper included: a registered custom spec that
    *subclasses* the ring (or packs a subclass) is a new claim about that
    block's content, so it must not inherit the exemption -- it falls through
    to the transferable-scratch refusal above.

    Eras that ship no ring class answer False, so the refusal stays in place.
    """
    if not _COMPRESSION_SCRATCH_TYPES:
        return False
    # Exact types, like every other spec receipt in this file: a subclass of a
    # ring (or of the packed wrapper) is a new claim about its own layout, and
    # dropping it here would bypass the gate that refuses undeclared classes.
    members = getattr(spec, "kv_cache_specs", None)
    if UniformTypeKVCacheSpecs is not None and type(spec) is UniformTypeKVCacheSpecs:
        if not isinstance(members, dict):
            return False
        return bool(members) and all(
            type(member) in _COMPRESSION_SCRATCH_TYPES for member in members.values()
        )
    return type(spec) in _COMPRESSION_SCRATCH_TYPES


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
    their class, element type, page size, page stride, row granularity and
    window retention all agree (a packed group may legally mix e.g. full and
    R-SWA layers, two R-SWA layers may carry different windows, and V4.1 packs
    compressed (tokens_per_state 2) next to uncompressed layers: one bucket
    keeps one geometry *and* one retention *and* one row size)."""

    class_name: str
    dtype_name: str
    page_bytes: int
    # spec.page_size_padded as vLLM set it (None = no alignment padding).
    pad: Optional[int]
    # Head geometry: equal page bytes do not imply equal tensor shapes
    # (2 heads x 64 and 1 head x 128 share a page), and one bucket must
    # keep one tensor shape.
    num_kv_heads: int
    head_size: int
    head_size_v: int
    # None = full prefix; not None = the bucket's windowed contract.
    retention: Optional[WindowRetention]
    # spec.tokens_per_state (compress_ratio on vllm <= 0.28); 1 = one row per
    # token. Equal page bytes do not imply equal row granularity: today the
    # SWA layer of a V4-style model (tps=1, block 32, 32 rows x 584 B) and a
    # compressed main layer (tps=2, block 64, 32 rows x 584 B) happen to
    # differ in class and alignment padding as well, but both separators are
    # incidental -- same-class layers with tps 1 and 2, or specs without
    # alignment padding, would collide on every other key while their
    # address math differs. This dimension is the semantic guardrail; the
    # load-bearing construction is pinned in TestBucketKeyKeepsRowGranularityApart.
    tokens_per_state: int


def _bucket_key(spec: Any, origin: str) -> _BucketKey:
    head_size = getattr(spec, "head_size", 0)
    tokens_per_state, _field = _tokens_per_state(spec)
    return _BucketKey(
        class_name=type(spec).__name__,
        dtype_name=str(getattr(spec, "dtype", None)),
        page_bytes=_compact_page_bytes(spec, origin),
        pad=getattr(spec, "page_size_padded", None),
        num_kv_heads=getattr(spec, "num_kv_heads", 0),
        head_size=head_size,
        head_size_v=getattr(spec, "head_size_v", None) or head_size,
        retention=_retention_of(spec),
        tokens_per_state=tokens_per_state if isinstance(tokens_per_state, int) else 1,
    )


def _bucket_order(
    key: _BucketKey,
) -> Tuple[int, str, str, int, int, int, int, str, int, int, int]:
    """Total order over the buckets of one group: biggest page first (so the
    main layer of a packed group keeps the bare group name), then class,
    dtype, head geometry, pad, window retention and row granularity. Only
    spec-local fields enter, so the suffixes -- and with them the wire names --
    are a pure function of the bucket *set*: layer registration order, or a
    different layer order on another rank, cannot move them. For a model with
    no windowed and no compressed layer every retention is None and every
    tokens_per_state is 1, so old bucket sets order exactly as before."""
    retention = key.retention
    return (
        -key.page_bytes,
        key.class_name,
        key.dtype_name,
        key.num_kv_heads,
        key.head_size,
        key.head_size_v,
        key.pad if key.pad is not None else -1,
        retention.kind if retention is not None else "",
        retention.window_tokens if retention is not None else -1,
        retention.extra_retained_tokens if retention is not None else -1,
        key.tokens_per_state,
    )


def _attention_meta(
    group_idx: int,
    layer_names: List[str],
    spec: Any,
    manager_block_size: int,
    *,
    suffix: str,
    origin: str,
    retention: Optional[WindowRetention] = None,
) -> AttentionGroupMeta:
    """Size one attention bucket from its spec.

    One compact page holds the whole block-table entry, ``block_size //
    tokens_per_state`` rows of ``page_bytes // rows`` bytes each, so the byte
    size of a manager block is exact in *rows*:

        per_block_bytes = bytes_per_row * (mbs // tps) * layers

    With tokens_per_state == 1 (every bucket before the V4.1 compressed rows)
    this is exactly the historical ``page_bytes // block_size * mbs * layers``
    -- one token per row, one row per token. The two divisions are assertions,
    not rounds: a page that is not a whole number of rows, or a manager block
    that is not, would silently floor into a layout nobody allocated.

    ``retention`` is the gate's verdict for this bucket (None = full prefix)."""
    page_bytes = _compact_page_bytes(spec, origin)
    tokens_per_row, _field = _tokens_per_state(spec)
    if not isinstance(tokens_per_row, int) or tokens_per_row < 1:
        # The gate refuses these; keep the sizing honest if it is ever bypassed.
        raise NotImplementedError(
            f"{origin}: tokens_per_state={tokens_per_row} is not a whole "
            f"number of tokens per row; cannot size the block"
        )
    if spec.block_size % tokens_per_row:
        raise NotImplementedError(
            f"{origin}: block_size={spec.block_size} is not divisible by "
            f"tokens_per_state={tokens_per_row}; a block-table entry would not "
            f"hold a whole number of rows"
        )
    rows_per_page = spec.block_size // tokens_per_row
    if page_bytes % rows_per_page != 0:
        raise NotImplementedError(
            f"{origin}: compact page size {page_bytes} is not a multiple of "
            f"the {rows_per_page} rows a page holds; refusing to floor the "
            f"per-row byte size"
        )
    if manager_block_size % tokens_per_row:
        raise NotImplementedError(
            f"{origin}: manager block size {manager_block_size} is not "
            f"divisible by tokens_per_state={tokens_per_row}; a manager block "
            f"would not hold a whole number of rows"
        )
    bytes_per_row = page_bytes // rows_per_page
    rows_per_block = manager_block_size // tokens_per_row
    return AttentionGroupMeta(
        group_idx=group_idx,
        layer_names=list(layer_names),
        block_size=spec.block_size,
        per_block_bytes=bytes_per_row * rows_per_block * len(layer_names),
        spec_suffix=suffix,
        page_bytes=page_bytes,
        dtype=getattr(spec, "dtype", None),
        retention=retention,
        row_granular=(
            None
            if tokens_per_row == 1
            else RowGranularMeta(
                tokens_per_row=tokens_per_row,
                rows_per_page=rows_per_page,
                bytes_per_row=bytes_per_row,
            )
        ),
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
    # A packed group whose members agree on their coverage (same window
    # retention, same page semantics) folds losslessly for the scheduler -- its
    # one representative sub spec carries the same retention -- and each bucket
    # then transfers on its own identity (wire name, staging pool, staged
    # captures). A packed group whose members *disagree* on coverage is the
    # case the scheduler's folded view cannot describe, and this branch cannot
    # bail out either: if the representative is the full-attention member, the
    # scheduler emits no capture and no hit clamp although the worker's windowed
    # bucket needs staged bytes (its claimed saves then fail); with a windowed
    # representative the announcement can describe the wrong retention. Refuse
    # the divergence here, where the whole member list is visible -- the folding
    # guard (the folding change stacked on this one) enforces the same predicate
    # at registration, for metas built without going through this function.
    coverages = {key.retention for key, _layers in ordered}
    if len(coverages) > 1:
        raise NotImplementedError(
            f"group {idx} (UniformTypeKVCacheSpecs) mixes {len(ordered)} "
            f"transfer buckets with different coverage "
            f"{sorted(str(c) for c in coverages)}: vLLM folds the group to one "
            f"representative sub spec for the scheduler, so the connector could "
            f"not announce or clamp every bucket's window. Keep the windowed "
            f"layers together (one retention per packed group) or let them form "
            f"their own group."
        )
    return [
        _attention_meta(
            idx,
            [layer_name for layer_name, _ in layers],
            layers[0][1],
            manager_block_size,
            suffix="" if i == 0 else f"_b{i}",
            origin=f"group {idx} (UniformTypeKVCacheSpecs)",
            retention=key.retention,
        )
        for i, (key, layers) in enumerate(ordered)
    ]


def _parse_group(
    idx: int, group: Any, manager_block_size: int, *, calculate_kv_scales: bool
) -> List[GroupMeta]:
    """One vLLM kv_cache_group -> its transfer buckets (a plain group = one).

    Both group-level receipts are matched by *exact* type: a MambaSpec subclass
    is a new claim about what a block's state covers (the transfer path relies
    on "one state per block end", see _state_block_ids), and a
    UniformTypeKVCacheSpecs subclass need not be a page-size pack at all. An
    undeclared subclass therefore falls through to the attention gate, which
    refuses it by name (vLLM 0.14 - 0.31 ships no subclass of either).
    """
    spec = group.kv_cache_spec
    layers = list(group.layer_names)
    if type(spec) is MambaSpec:
        return [
            StateGroupMeta(
                group_idx=idx,
                layer_names=layers,
                block_size=spec.block_size,
                per_block_bytes=spec.page_size_bytes * len(layers),
                page_size_bytes=spec.page_size_bytes,
            )
        ]
    if UniformTypeKVCacheSpecs is not None and type(spec) is UniformTypeKVCacheSpecs:
        return _uniform_group_metas(
            idx, spec, manager_block_size, calculate_kv_scales=calculate_kv_scales
        )
    origin = f"group {idx}"
    retention = _check_attention_spec_supported(
        origin, spec, calculate_kv_scales=calculate_kv_scales
    )
    return [
        _attention_meta(
            idx,
            layers,
            spec,
            manager_block_size,
            suffix="",
            origin=origin,
            retention=retention,
        )
    ]


def _check_unique_names(metas: List[GroupMeta]) -> None:
    """The manager keys locations by spec name, so two buckets sharing a name
    would silently merge (or overwrite each other). The rank prefix is not
    part of the identity: every rank registers the same name set."""
    names = [f"g{m.group_idx}{m.spec_suffix}" for m in metas]
    assert len(names) == len(set(names)), f"wire spec name collision: {names}"


def _transferable_groups(kv_cache_config: Any) -> List[Tuple[int, Any]]:
    """The (vLLM group index, group) pairs this connector may transfer.

    vLLM 0.29+ projects the group list for connectors: KVCacheGroupSpec.
    enable_kv_transfer=False marks groups the engine rebuilds itself (ephemeral
    accelerator-side replicas of a transferable source group, e.g. HiSparse's
    resident/hot tiers), and KVCacheConfig.transfer_group_ids / transfer_groups
    are the authoritative projection. A group the engine does not hand to
    connectors must not be registered here even when its spec looks cacheable
    (_is_scratch_spec judges the class, not the layout's role).

    The projection decides *membership only*: the index we keep is the raw vLLM
    group index, because that is the index space of every block-id payload this
    connector receives (KVCacheBlocks.get_block_ids() and
    CachedRequestData.new_block_ids are both unfiltered -- see
    v1/core/kv_cache_manager.get_block_ids and its new_block_ids caller), and
    the ledger plus every consumer index by it. Renumbering to the projected
    position would pair our metas with another group's block table.

    Eras without the attribute (vLLM <= 0.28) expose every group; the flag
    arrived with the specs it describes.
    """
    groups = kv_cache_config.kv_cache_groups
    transfer_ids = getattr(kv_cache_config, "transfer_group_ids", None)
    if transfer_ids is None:
        return [(idx, group) for idx, group in enumerate(groups)]
    return [(idx, groups[idx]) for idx in transfer_ids]


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

    Groups that carry nothing the manager may cache are skipped instead of
    refused -- EAGLE/MTP drafters (vLLM marks them), per-request scratch and
    engine-replayed windows (both by _skip_reason, which also gives the log
    line), plus the groups the engine keeps out of the transfer contract (see
    _transferable_groups). A skipped group keeps its vLLM block-table index, so
    its number is simply absent from the wire names.

    The two vLLM contracts differ, though, and a scratch group that the engine
    *does* expose for transfer (vLLM 0.29+ default: enable_kv_transfer=True) is
    refused rather than dropped: the engine rebuilds only the groups it owns,
    so an external hit would otherwise resume with that state uninitialized.
    A UniformTypeKVCacheSpecs group is bucketed by *member*, in the worker's
    view, because vLLM's merge decision does not look at the row granularity:
    the V4.1 packed group holds compressed (tokens_per_state 2) and
    uncompressed layers at once, which are different transfer geometries (see
    RowGranularMeta and _BucketKey) and therefore different buckets. Buckets
    that agree on their coverage (window retention included) fold losslessly
    for the scheduler, which only ever sees one representative sub spec; a
    packed group whose members *disagree* is refused by the folding guard
    (_check_coverage_survives_folding).
    The one exception is scratch the engine itself rebuilds per request -- the
    compression ring (see _is_engine_rebuilt_scratch) -- which is skipped even
    when transferable.
    """
    metas: List[GroupMeta] = []
    transferable = _transferable_groups(kv_cache_config)
    declares_projection = (
        getattr(kv_cache_config, "transfer_group_ids", None) is not None
    )
    transferable_ids = {idx for idx, _ in transferable}
    for idx, group in enumerate(kv_cache_config.kv_cache_groups):
        if idx not in transferable_ids:
            logger.warning(
                "skip %s group %d (%d layers): the engine does not transfer it "
                "(enable_kv_transfer=False) and rebuilds it after a load",
                type(group.kv_cache_spec).__name__,
                idx,
                len(group.layer_names),
            )
    for idx, group in transferable:
        if getattr(group, "is_eagle_group", False):
            logger.warning(
                "skip eagle group %d (%d layers)", idx, len(group.layer_names)
            )
            continue
        skip_reason = _skip_reason(group.kv_cache_spec)
        if skip_reason is not None:
            if (
                _is_scratch_spec(group.kv_cache_spec)
                and declares_projection
                and not _is_engine_rebuilt_scratch(group.kv_cache_spec)
            ):
                raise NotImplementedError(
                    f"group {idx}: {type(group.kv_cache_spec).__name__} is "
                    f"per-request scratch (prefix_cacheable=False) yet the engine "
                    f"exposes it for transfer (enable_kv_transfer=True): its state "
                    f"must be restored on an external hit and the engine does not "
                    f"rebuild a group it hands to connectors. TairKvCacheConnector "
                    f"has no transfer layout for it, so the instance is refused "
                    f"rather than resumed with uninitialized state"
                )
            if _is_engine_rebuilt_scratch(group.kv_cache_spec):
                logger.warning(
                    "skip %s group %d (%d layers): per-request compression ring, "
                    "refilled by the engine for every request (not prefix KV)",
                    type(group.kv_cache_spec).__name__,
                    idx,
                    len(group.layer_names),
                )
            else:
                logger.warning(
                    "skip %s group %d (%d layers): the class declares itself out "
                    "of the prefix cache (prefix_cacheable=False and no slot "
                    "mapping or no layer views), so the manager has nothing to "
                    "serve; if this class is meant to be cached, name it in "
                    "vllm_common.py instead",
                    type(group.kv_cache_spec).__name__,
                    idx,
                    len(group.layer_names),
                )
            logger.warning(
                "skip %s group %d (%d layers): %s; if this class is meant to "
                "be cached, name it in vllm_common.py instead",
                type(group.kv_cache_spec).__name__,
                idx,
                len(group.layer_names),
                skip_reason,
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
