from enum import Enum
from typing import Tuple, Dict, Optional, Any

import attrs
import torch


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
