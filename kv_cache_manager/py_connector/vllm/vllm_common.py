"""Shared vocabulary between the scheduler side and the worker side.

Everything here is role-agnostic: the data model both cores speak
(GroupMeta), the spec naming scheme shared with the manager
registration, the KV layout normalization, the hybrid capability gate and
the kv_cache_config parsing. The two cores (scheduler_core / worker_core)
and the thin connector shell (v1_connector) build on this module; nothing
here may import them.
"""

from dataclasses import dataclass, field
from typing import List, Optional

import torch

from vllm.v1.kv_cache_interface import FullAttentionSpec, MambaSpec

from kv_cache_manager.py_connector.common.logger import logger
from kv_cache_manager.py_connector.vllm.transfer_types import (
    KVLayout,
)

# Spec group names advertised at registration and used per key in
# start_write_cache. See build_spec_groups for the semantics.
ATTN_SPEC_GROUP = "attn"
FULL_SPEC_GROUP = "full"


def spec_name(tp_rank: int, group_idx: int) -> str:
    """Location spec name for one (tp rank, kv cache group) shard."""
    return f"tp{tp_rank}_g{group_idx}"


def build_spec_groups(group_metas: List["GroupMeta"], tp_size: int) -> List[dict]:
    """LocationSpecGroups describing which specs a block may carry.

    Hybrid (mamba "align") models write a *sparse* set of recurrent states:
    vLLM materializes a state only at segment boundaries, so the interior
    manager blocks of a request have attention KV but no state. Declaring
    two groups lets ``start_write_cache`` say, per block, which specs that
    block will actually hold:

    * ``full`` -- every group's spec (attention KV + all states);
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
        spec_name(rank, meta.group_idx)
        for rank in range(tp_size)
        for meta in group_metas if isinstance(meta, AttentionGroupMeta))
    all_specs = sorted(
        spec_name(rank, meta.group_idx)
        for rank in range(tp_size) for meta in group_metas)
    return [{"name": ATTN_SPEC_GROUP, "spec_names": attn_specs},
            {"name": FULL_SPEC_GROUP, "spec_names": all_specs}]


@dataclass(frozen=True)
class GroupMeta:
    """Static description of one kv_cache_group, derived from KVCacheConfig
    (see parse_groups). Available in both scheduler and worker roles (before
    tensors exist). Kind-specific subclasses carry the kind-specific sizing."""

    group_idx: int
    layer_names: List[str]
    # The group's block table granularity in tokens (spec.block_size).
    block_size: int
    # Bytes stored per manager block for the whole group.
    per_block_bytes: int


@dataclass(frozen=True)
class AttentionGroupMeta(GroupMeta):
    """FullAttentionSpec group: token-granular KV, re-blockable to the
    manager block size. Sizing derives from the *compact* page size (see
    parse_groups)."""


@dataclass(frozen=True)
class StateGroupMeta(GroupMeta):
    """MambaSpec group: one opaque state per block, verbatim byte copies.
    page_size_bytes is per layer; per_block_bytes = page_size_bytes * layers."""

    # Bytes per block per state layer (spec.page_size_bytes).
    page_size_bytes: int = 0


def parse_groups(kv_cache_config, manager_block_size: int) -> List[GroupMeta]:
    """Derive the transferable GroupMeta list from vLLM's KVCacheConfig
    (https://github.com/vllm-project/vllm/blob/v0.26.0/vllm/v1/kv_cache_interface.py#L952:
    kv_cache_groups holds one KVCacheGroupSpec per block table, each with its
    kv_cache_spec -- FullAttentionSpec at L227, MambaSpec at L690)."""
    metas = []
    for idx, group in enumerate(kv_cache_config.kv_cache_groups):
        if getattr(group, "is_eagle_group", False):
            logger.warning("skip eagle group %d (%d layers)", idx, len(group.layer_names))
            continue
        spec = group.kv_cache_spec
        if isinstance(spec, MambaSpec):
            metas.append(StateGroupMeta(
                group_idx=idx,
                layer_names=list(group.layer_names),
                block_size=spec.block_size,
                per_block_bytes=spec.page_size_bytes * len(group.layer_names),
                page_size_bytes=spec.page_size_bytes,
            ))
        elif isinstance(spec, FullAttentionSpec):
            # FullAttentionSpec doubles as the merged spec of hybrid
            # SWA/chunked-attention models (vLLM merges window layers into
            # it, keeping sliding_window/attention_chunk_size set). Those
            # blocks hold windowed KV, not the full prefix -- publishing
            # them as prefix caches would corrupt reuse. Refuse explicitly.
            for window_field in ("sliding_window", "attention_chunk_size"):
                if getattr(spec, window_field, None) is not None:
                    raise NotImplementedError(
                        f"group {idx}: FullAttentionSpec has {window_field}="
                        f"{getattr(spec, window_field)}; sliding-window / "
                        f"chunked attention KV is not full-prefix and is "
                        f"not yet supported by TairKvCacheConnector")
            # Attention KV is token-granular; scale from the spec's page size
            # to the manager block size. Use the *compact* page size:
            # spec.page_size_bytes returns page_size_padded when set, which
            # includes an allocation-alignment gap the gather kernel never
            # copies -- sizing locations/staging buffers with it would break
            # the staging view() and waste storage. real_page_size_bytes is
            # exactly the raw KV bytes (2 * block * heads * head_dim * dtype).
            compact_page_bytes = getattr(spec, "real_page_size_bytes", None)
            if compact_page_bytes is None:
                if getattr(spec, "page_size_padded", None) is not None:
                    raise NotImplementedError(
                        f"group {idx}: page_size_padded="
                        f"{spec.page_size_padded} but this vLLM exposes no "
                        f"real_page_size_bytes to recover the compact page "
                        f"size; padded attention layouts are unsupported here")
                compact_page_bytes = spec.page_size_bytes
            per_token_bytes = compact_page_bytes // spec.block_size
            metas.append(AttentionGroupMeta(
                group_idx=idx,
                layer_names=list(group.layer_names),
                block_size=spec.block_size,
                per_block_bytes=per_token_bytes * manager_block_size * len(group.layer_names),
            ))
        else:
            raise NotImplementedError(
                f"Unsupported kv cache spec {type(spec).__name__} in group {idx}")
    assert metas, "no usable kv cache groups"
    return metas


def attn_kv_views(ref: torch.Tensor) -> tuple:
    """Normalize one attention layer's paged KV cache into per-pointer views.

    vLLM's flash_attn backend changed ``get_kv_cache_shape`` twice; the three
    layouts (see KVLayout for the per-version source links) are detected from
    the tensor shape itself (never from version strings):

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
                f"dim from a num_blocks dim of size 2")
        if kv_first:
            return [ref[0], ref[1]], KVLayout.SPLIT_KV_5D_KV_FIRST
        if n_first:
            return [ref[:, 0], ref[:, 1]], KVLayout.SPLIT_KV_5D_N_FIRST
    raise NotImplementedError(
        f"unrecognized kv cache layout {tuple(ref.shape)}; expected the packed "
        f"4-D (vllm >= 0.26.0) or one of the split K/V 5-D layouts "
        f"(vllm <= 0.25.x)")


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


def ensure_hybrid_supported(force: bool = False):
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
                "external match will crash it")
            return
        raise NotImplementedError(
            "TairKvCacheConnector: cannot verify that this vLLM supports "
            "hybrid (mamba) models with an external KV connector -- "
            "Scheduler._mamba_block_aligned_split exists but its source is "
            "unavailable, so the vllm <= 0.22.x blocking assert cannot be "
            "ruled out. If you know this vLLM is >= 0.23.0, set "
            "kv_connector_extra_config {\"force_hybrid_support\": true} to "
            "bypass this check.")
    raise NotImplementedError(
        "TairKvCacheConnector: this vLLM version cannot combine hybrid "
        "(mamba) models with an external KV connector -- its scheduler "
        "asserts num_external_computed_tokens == 0 in "
        "_mamba_block_aligned_split ('External KV connector is not "
        "verified yet'). Upgrade to vLLM >= 0.23.0 for hybrid model "
        "support; full-attention models are unaffected.")
