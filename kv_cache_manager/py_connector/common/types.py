from dataclasses import dataclass, field
from typing import List, Optional

import torch


@dataclass
class TransferGroup:
    """One KV cache group = one independent transfer unit.

    A vLLM model exposes ``kv_cache_config.kv_cache_groups``: for pure attention
    models there is a single ``FullAttentionSpec`` group; hybrid models expose
    several ``MambaSpec`` groups plus one ``FullAttentionSpec`` group. Every
    group has its own block table (``block_ids`` is a tuple indexed by group)
    in units of its own ``block_size``, and its own storage strategy, so the
    connector treats each group as a self-contained transfer unit.
    """

    group_idx: int
    # Location spec name registered with the manager, e.g. "tp0_g3".
    spec_name: str
    # True for attention layers (token-granular strided gather/scatter);
    # False for mamba/linear/gdn state layers (per-block opaque byte copy).
    is_attention: bool
    layer_names: List[str]
    # The group's own block table granularity in tokens (spec.block_size).
    block_size: int
    # Bytes stored per manager block for this whole group (all its layers).
    per_block_bytes: int
    layer_num: int = 0

    # --- Attention-only fields (is_attention == True) ---
    # int64 tensor of transfer-pointer bases on the compute device. For packed
    # K/V layouts (vllm >= 0.26.0) one pointer per layer [L0, L1, ...]; for
    # split K/V layouts (vllm <= 0.25.x) two per layer [K0, V0, K1, V1, ...].
    kvcache_ptr_tensor_gpu: Optional[torch.Tensor] = None
    # Number of transfer pointers (rows of the staging buffer view per block).
    # layer_num for packed layouts, 2 * layer_num for split K/V layouts.
    num_kv_ptrs: int = 0
    per_token_dim: int = 0          # heads * content dim per pointer
    kernel_block_size: int = 0      # tokens per kernel page
    kv_stride: int = 0              # K->V element offset added by the kernel; 0
                                    # because every pointer is its own base
    block_stride: int = 0           # element stride between kernel pages of one
                                    # pointer, 0 => contiguous flat layout

    # --- State-only fields (is_attention == False) ---
    # Per layer (num_blocks, page_size_bytes) uint8 views into the state storage.
    block_view_tensors: List[torch.Tensor] = field(default_factory=list)
    page_size_bytes: int = 0        # bytes per block per state layer


@dataclass
class KVCacheInfo:
    """Worker-side registered KV cache description (all groups)."""

    tp_rank: int
    world_size: int
    groups: List[TransferGroup]
    device: torch.device
    dtype: torch.dtype
