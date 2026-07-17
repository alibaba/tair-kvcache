from enum import Enum
from typing import Tuple, Dict, Optional, Any, List

import attrs
import torch


@attrs.define(frozen=True)
class HybridCacheInfo:
    """Info for non-attention (mamba/gdn/linear) layers in a hybrid model."""
    layer_names: List[str]
    # (num_hybrid_layers, ) int64 tensor of base ptrs on CPU (for offset computation)
    ptr_tensor_cpu: torch.Tensor
    # List of raw byte-view tensors on GPU: each is (num_blocks, page_size_bytes) uint8
    # These are views into the hybrid state tensors' untyped_storage
    block_view_tensors: List[torch.Tensor]
    # byte size per block per layer (all hybrid layers share the same page_size_bytes)
    page_size_bytes: int
    # number of hybrid layers
    layer_num: int


@attrs.define(frozen=True)
class KVCacheInfo:
    tp_rank: int
    world_size: int
    kvcaches: Dict[str, torch.Tensor]
    kvcache_ptr_tensor_cpu: torch.Tensor
    kvcache_ptr_tensor_gpu: torch.Tensor
    all_kvcache_ptr_tensor_gpu: torch.Tensor
    layer_num: int
    local_token_num: int
    per_manager_block_shape: Tuple[int, ...]
    per_manager_block_byte_size: int
    per_token_per_layer_dim_size: int
    device: torch.device
    dtype: torch.dtype
    # Hybrid (mamba/gdn/linear) layer info, None for pure attention models
    hybrid_info: Optional[HybridCacheInfo] = None
    # Stride parameters for strided memory layout (0 = contiguous/flat indexing)
    kv_stride: int = 0  # stride between K and V (for V pointers)
    block_stride: int = 0  # stride between blocks
    local_block_size: int = 0  # actual block size in tensor (0 = use manager block size)
