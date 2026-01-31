from typing import List, Optional, Union

import torch
import triton
import triton.language as tl


def pytorch_dtype_to_triton_dtype(torch_dtype):
    """
    Converts a PyTorch dtype to a Triton language dtype.
    """
    # Define the mapping between torch dtypes and triton.language dtypes
    dtype_mapping = {
        torch.float32: tl.float32,
        torch.float64: tl.float64,
        torch.float16: tl.float16,
        torch.bfloat16: tl.bfloat16,
        torch.int8: tl.int8,
        torch.int16: tl.int16,
        torch.int32: tl.int32,
        torch.int64: tl.int64,
        torch.uint8: tl.uint8,
        torch.float8_e4m3fn: tl.float8e4nv,
        torch.bool: tl.int1
    }

    triton_dtype = dtype_mapping.get(torch_dtype)
    if triton_dtype is None:
        raise ValueError(f"Unsupported PyTorch dtype: {torch_dtype}")
    return triton_dtype


@triton.jit
def kv_cache_batch_gather_kernel(
        kv_cache_ptrs_ptr,  # 指针数组基址 [num_layers * kv_count]
        dst_ptr,  # 输出缓冲区 (pinned host memory)
        block_token_indices_ptr,  # [total_blocks, num_tokens_per_block]
        dst_block_indices_ptr,  # [total_blocks]
        total_blocks: int,  # 需要处理的总block数
        NUM_TOKENS_PER_BLOCK: tl.constexpr,
        NUM_DIMS_PER_TOKEN: tl.constexpr,
        NUM_KVCACHE_PTRS: tl.constexpr,  # num_layers * kv_count
        BLOCK_SIZE: tl.constexpr,  # 隐藏维度分块大小
        DTYPE: tl.constexpr = tl.float16,
):
    NUM_DIMS_PER_BLOCK = NUM_TOKENS_PER_BLOCK * NUM_DIMS_PER_TOKEN

    pid = tl.program_id(0)
    grid_size = tl.num_programs(0)  # 实际grid大小 (如3)

    # 每个grid处理多个block: 采用grid-stride loop
    for block_idx in tl.range(pid, total_blocks, grid_size):
        # 1. 加载当前block在dst中的索引
        dst_block_idx = tl.load(dst_block_indices_ptr + block_idx)

        # 2. 预计算当前block在dst中的基础偏移 (元素为单位)
        block_offset = (
                dst_block_idx
                * NUM_KVCACHE_PTRS
                * NUM_TOKENS_PER_BLOCK
                * NUM_DIMS_PER_TOKEN
        )

        # 3. 遍历所有KV缓存指针 (k/v for each layer)
        for ptr_idx in tl.range(NUM_KVCACHE_PTRS):
            # 3.1 加载当前层的KV缓存基地址
            kvcache_ptr = tl.load(kv_cache_ptrs_ptr + ptr_idx).to(tl.pointer_type(DTYPE))

            # 3.2 计算当前层在dst中的基础偏移
            layer_offset = block_offset + ptr_idx * NUM_DIMS_PER_BLOCK
            dst_layer_ptr = dst_ptr + layer_offset

            # 4. host dram上的单个block[token,dim]摊平成1维处理 (保证store时的memory coalesce)
            for off in tl.range(0, NUM_DIMS_PER_BLOCK, BLOCK_SIZE):
                # 创建分块内的偏移
                offsets = off + tl.arange(0, BLOCK_SIZE)
                mask = offsets < NUM_DIMS_PER_BLOCK

                # 计算offset对应的token idx和dim idx
                token_idx_in_block = offsets // NUM_DIMS_PER_TOKEN
                dim_idx_in_token = offsets % NUM_DIMS_PER_TOKEN

                # 计算对应的源地址
                token_gather_mask = token_idx_in_block < NUM_TOKENS_PER_BLOCK
                global_token_idx = tl.load(
                    block_token_indices_ptr + block_idx * NUM_TOKENS_PER_BLOCK + token_idx_in_block,
                    mask=token_gather_mask,
                    other=0
                )

                # 从HBM的KV缓存加载数据
                # 计算源指针: [BLOCK_SIZE]
                src_ptrs = kvcache_ptr + global_token_idx * NUM_DIMS_PER_TOKEN + dim_idx_in_token
                load_mask = mask & token_gather_mask
                data = tl.load(src_ptrs, mask=load_mask, other=0.0)
                # 大块连续写入 host memory (PCIe优化)
                dst_ptrs = dst_layer_ptr + offsets
                tl.store(dst_ptrs, data, mask=load_mask)


def batch_gather_kv_caches(
        # List of KV cache tensors ptr (each shape [2, total_token_in_kvcache, hidden_size])
        kv_caches_ptrs_tensor: torch.Tensor,
        # Shape [block_num, num_layers * kv_num, num_tokens_per_block, dim_size_per_token_per_layer]
        dst_tensor: torch.Tensor,
        block_token_indices: List[int],  # List of token positions to gather
        dst_block_indices: List[int],  # List of dst block indices
        num_tokens_per_block: int,
        dim_size_per_token_per_layer: int,
        sm_count: int = 3
):
    # 配置参数
    total_blocks = len(dst_block_indices)
    total_kv_caches_ptr = kv_caches_ptrs_tensor.size(0)
    grid = (sm_count,)  # 限制SM数量

    device = kv_caches_ptrs_tensor.device
    block_token_indices_tensor = torch.tensor(block_token_indices, dtype=torch.int32, device="cpu").to(device,
                                                                                                       non_blocking=True)
    dst_block_indices_tensor = torch.tensor(dst_block_indices, dtype=torch.int32, device="cpu").to(device,
                                                                                                   non_blocking=True)

    kv_cache_batch_gather_kernel[grid](
        kv_caches_ptrs_tensor,
        dst_tensor,
        block_token_indices_tensor,
        dst_block_indices_tensor,
        total_blocks=total_blocks,
        NUM_TOKENS_PER_BLOCK=num_tokens_per_block,
        NUM_DIMS_PER_TOKEN=dim_size_per_token_per_layer,
        NUM_KVCACHE_PTRS=total_kv_caches_ptr,
        BLOCK_SIZE=2048,
        DTYPE=pytorch_dtype_to_triton_dtype(dst_tensor.dtype),
        num_warps=32,
    )
    # TODO autotune num_warps and BLOCK_SIZE


@triton.jit
def kv_cache_batch_scatter_kernel(
        kv_cache_ptrs_ptr,  # 指针数组基址 [num_layers * kv_count]
        src_ptr,  # 源缓冲区 (pinned host memory)
        block_token_indices_ptr,  # [total_blocks, num_tokens_per_block]
        src_block_indices_ptr,  # [total_blocks]
        total_blocks: int,  # 需要处理的总block数
        NUM_TOKENS_PER_BLOCK: tl.constexpr,
        NUM_DIMS_PER_TOKEN: tl.constexpr,
        NUM_KVCACHE_PTRS: tl.constexpr,  # num_layers * kv_count
        BLOCK_SIZE: tl.constexpr,  # 隐藏维度分块大小
        DTYPE: tl.constexpr = tl.float16,
):
    NUM_DIMS_PER_BLOCK = NUM_TOKENS_PER_BLOCK * NUM_DIMS_PER_TOKEN

    pid = tl.program_id(0)
    grid_size = tl.num_programs(0)  # 实际grid大小 (如3)

    # 每个grid处理多个block: 采用grid-stride loop
    for block_idx in tl.range(pid, total_blocks, grid_size):
        # 1. 加载当前block在src中的索引
        src_block_idx = tl.load(src_block_indices_ptr + block_idx)

        # 2. 预计算当前block在src中的基础偏移 (元素为单位)
        block_offset = (
                src_block_idx
                * NUM_KVCACHE_PTRS
                * NUM_TOKENS_PER_BLOCK
                * NUM_DIMS_PER_TOKEN
        )

        # 3. 遍历所有KV缓存指针 (k/v for each layer)
        for ptr_idx in range(NUM_KVCACHE_PTRS):
            # 3.1 加载当前层的KV缓存基地址
            kvcache_ptr = tl.load(kv_cache_ptrs_ptr + ptr_idx).to(tl.pointer_type(DTYPE))

            # 3.2 计算当前层在src中的基础偏移
            layer_offset = block_offset + ptr_idx * NUM_DIMS_PER_BLOCK
            src_layer_ptr = src_ptr + layer_offset

            # 4. host dram上的单个block[token,dim]摊平成1维处理 (保证load时的memory coalesce)
            for off in range(0, NUM_DIMS_PER_BLOCK, BLOCK_SIZE):
                # 创建分块内的偏移
                offsets = off + tl.arange(0, BLOCK_SIZE)
                mask = offsets < NUM_DIMS_PER_BLOCK

                # 计算offset对应的token idx和dim idx
                token_idx_in_block = offsets // NUM_DIMS_PER_TOKEN
                dim_idx_in_token = offsets % NUM_DIMS_PER_TOKEN

                # 计算对应的目的地址
                token_gather_mask = token_idx_in_block < NUM_TOKENS_PER_BLOCK
                global_token_idx = tl.load(
                    block_token_indices_ptr + block_idx * NUM_TOKENS_PER_BLOCK + token_idx_in_block,
                    mask=token_gather_mask,
                    other=0
                )

                load_mask = mask & token_gather_mask
                # 大块连续读取host memory (PCIe优化)
                new_src_ptrs = src_layer_ptr + offsets
                data = tl.load(new_src_ptrs, mask=load_mask, other=0.0)

                # 向HBM的KV缓存写入数据
                # 计算目的指针: [BLOCK_SIZE]
                dst_ptrs = kvcache_ptr + global_token_idx * NUM_DIMS_PER_TOKEN + dim_idx_in_token
                tl.store(dst_ptrs, data, mask=load_mask)


def batch_scatter_kv_caches(
        # List of KV cache tensors ptr (each shape [2, total_token_in_kvcache, hidden_size])
        kv_caches_ptrs_tensor: torch.Tensor,
        # Shape [block_num, num_layers * kv_num, num_tokens_per_block, dim_size_per_token_per_layer]
        src_tensor: torch.Tensor,  # 注意：src_tensor在PCIE连接的host DRAM上 (pinned memory)
        block_token_indices: List[int],  # List of token positions to scatter to
        src_block_indices: List[int],  # List of src block indices
        num_tokens_per_block: int,
        dim_size_per_token_per_layer: int,
        sm_count: int = 3
):
    # 配置参数
    total_blocks = len(src_block_indices)
    total_kv_caches_ptr = kv_caches_ptrs_tensor.size(0)
    grid = (sm_count,)  # 限制SM数量

    device = kv_caches_ptrs_tensor.device

    # 将host数据异步复制到device（非阻塞）
    block_token_indices_tensor = torch.tensor(
        block_token_indices, dtype=torch.int32, device="cpu"
    ).to(device, non_blocking=True)

    src_block_indices_tensor = torch.tensor(
        src_block_indices, dtype=torch.int32, device="cpu"
    ).to(device, non_blocking=True)

    # 启动内核
    kv_cache_batch_scatter_kernel[grid](
        kv_caches_ptrs_tensor,
        src_tensor,  # 直接传递host内存指针
        block_token_indices_tensor,
        src_block_indices_tensor,
        total_blocks=total_blocks,
        NUM_TOKENS_PER_BLOCK=num_tokens_per_block,
        NUM_DIMS_PER_TOKEN=dim_size_per_token_per_layer,
        NUM_KVCACHE_PTRS=total_kv_caches_ptr,
        BLOCK_SIZE=2048,
        DTYPE=pytorch_dtype_to_triton_dtype(src_tensor.dtype),
        num_warps=32,
    )
