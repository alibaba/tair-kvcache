"""Worker-side transfer data model, one type per KV cache group kind.

The vLLM connector treats every ``kv_cache_group`` as an independent
transfer unit. Attention groups and state (mamba/linear) groups transfer
dissimilar data through dissimilar mechanics -- token-granular gather /
scatter through the Triton kernel vs per-block opaque byte copies -- so
they are modelled as two explicit subclasses of :class:`TransferGroup`
and dispatched with ``isinstance``. The dispatch is the extension point
for future group kinds (e.g. sliding-window attention would add its own
subclass instead of overloading a boolean).

Nothing here is consumed by the sglang connector; per the placement rule
("common only when shared") these types live under ``vllm/``.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional

import torch


class KVLayout(Enum):
    """The flash_attn paged-KV tensor layouts this connector understands.

    Detected from the tensor *shape* (never from version strings) in
    ``vllm_common.attn_kv_views``. One layout per vLLM era:

    * vLLM <= 0.22.1 returned ``(2, num_blocks, block, H, D)``:
      https://github.com/vllm-project/vllm/blob/v0.22.1/vllm/v1/attention/backends/flash_attn.py#L149
    * vLLM 0.23.0 - 0.25.x returned ``(num_blocks, 2, block, H, D)``:
      https://github.com/vllm-project/vllm/blob/v0.23.0/vllm/v1/attention/backends/flash_attn.py#L149
    * vLLM >= 0.26.0 returns the packed 4-D ``(num_blocks, H, block, 2D)``:
      https://github.com/vllm-project/vllm/blob/v0.26.0/vllm/v1/attention/backends/flash_attn.py#L141

    The saved byte layout differs between the split-K/V and packed eras, so
    KV cache is not portable across vLLM major upgrades; instance_id
    isolation prevents such mixing in practice.
    """

    SPLIT_KV_5D_KV_FIRST = "split_kv_5d_kv_first"   # (2, num_blocks, block, H, D)
    SPLIT_KV_5D_N_FIRST = "split_kv_5d_n_first"     # (num_blocks, 2, block, H, D)
    PACKED_4D = "packed_4d"                         # (num_blocks, H, block, 2D)


@dataclass(frozen=True)
class TransferGroup:
    """Base: fields every kv cache group carries regardless of kind."""

    group_idx: int
    # Location spec name registered with the manager, e.g. "tp0_g3".
    spec_name: str
    layer_names: List[str]
    # The group's own block table granularity in tokens (spec.block_size).
    block_size: int
    # Bytes stored per manager block for this whole group (all its layers).
    per_block_bytes: int
    layer_num: int


@dataclass(frozen=True)
class AttentionTransferGroup(TransferGroup):
    """Token-granular KV, moved through the strided gather/scatter kernel."""

    # Which vLLM-era layout the pointers below were normalized from.
    kv_layout: KVLayout
    # int64 tensor of transfer-pointer bases on the compute device. For the
    # packed layout one pointer per layer [L0, L1, ...]; for split K/V layouts
    # two per layer [K0, V0, K1, V1, ...] -- each view's data_ptr() is its own
    # base, so the kernel never adds a K->V offset.
    kvcache_ptr_tensor_gpu: torch.Tensor
    # Number of transfer pointers (staging buffer rows per block):
    # layer_num for the packed layout, 2 * layer_num for split K/V.
    num_kv_ptrs: int
    # heads * content dim per pointer (content dim is 2*D packed, D split).
    per_token_dim: int
    # Tokens per kernel page (may differ from block_size on padded pages).
    kernel_block_size: int
    # Element stride between kernel pages of one pointer; 0 => the pages are
    # contiguous and the kernel uses flat indexing.
    block_stride: int


@dataclass(frozen=True)
class StateTransferGroup(TransferGroup):
    """Per-block opaque state bytes, copied verbatim (mamba/linear/gdn)."""

    # Per layer, (num_blocks, page_size_bytes) uint8 views into the state
    # storage (all of a layer's state tensors share one storage).
    block_view_tensors: List[torch.Tensor] = field(default_factory=list)
    # Bytes per block per state layer (spec.page_size_bytes).
    page_size_bytes: int = 0


@dataclass(frozen=True)
class TransferPlan:
    """One group's slice of a save (or load): what to move, from/to where.

    Built by the worker from the manager's locations plus the request's
    block tables; consumed by the transfer tasks. ``uris`` is positionally
    aligned with the manager blocks and may contain ``None`` where a block
    carries no data for this group's spec (hybrid sparse coverage) -- what a
    hole means is the task's disposition logic (see data_transfer)."""

    group: TransferGroup
    uris: List
    # Attention groups: flat token slots per manager block (gather/scatter).
    token_indices: Optional[List[List[int]]] = None
    # State groups: source/target state block id per manager block.
    block_ids: Optional[List[int]] = None


@dataclass(frozen=True)
class KVCacheInfo:
    """Worker-side registered KV cache description (all groups)."""

    tp_rank: int
    world_size: int
    groups: List[TransferGroup]
    device: torch.device
    dtype: torch.dtype
