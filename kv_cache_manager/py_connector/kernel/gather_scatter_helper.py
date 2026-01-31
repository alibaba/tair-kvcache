from typing import List, Optional, Union
import threading

import torch
import triton
import triton.language as tl
from triton.language import static_print


@triton.jit
def kv_cache_scatter_kernel(
        # Pointers to KV cache tensors (flattened into one array)
        kv_cache_ptrs_ptr,  # pointer to array of pointers (size num_layers)
        source_ptr,  # pointer to source tensor (num_layers, 2, num_tokens, hidden_size)
        token_indices_ptr,  # pointer to token indices
        num_tokens_in_block,  # length of token_indices
        hidden_size,  # hidden dimension size
        total_token_in_kvcache,  # sequence length in KV cache
        num_layers,  # number of layers
        BLOCK_SIZE: tl.constexpr,
):
    # Get the current program's layer index and token position
    layer_idx = tl.program_id(0)
    token_pos = tl.program_id(1)

    # Boundary check
    if layer_idx >= num_layers or token_pos >= num_tokens_in_block:
        return

    # Get the token index to write to
    token_idx = tl.load(token_indices_ptr + token_pos)

    # Get the pointer to the current layer's KV cache
    kv_cache_ptr = tl.cast(tl.load(kv_cache_ptrs_ptr + layer_idx), source_ptr.dtype)

    # Check if token_idx is valid
    if token_idx >= total_token_in_kvcache:
        return

    # Calculate source offsets (for both K and V)
    source_offset_k = (layer_idx * num_tokens_in_block * 2 + 0 * num_tokens_in_block + token_pos) * hidden_size
    source_offset_v = (1 * num_tokens_in_block + layer_idx * num_tokens_in_block * 2 + token_pos) * hidden_size

    # Calculate target offsets (for both K and V)
    target_offset_k = (0 * total_token_in_kvcache + token_idx) * hidden_size
    target_offset_v = (1 * total_token_in_kvcache + token_idx) * hidden_size
    # Copy data in blocks
    for i in range(0, hidden_size, BLOCK_SIZE):
        offset = i + tl.arange(0, BLOCK_SIZE)
        mask = offset < hidden_size

        # Load from source
        val_k = tl.load(source_ptr + source_offset_k + offset, mask=mask)
        val_v = tl.load(source_ptr + source_offset_v + offset, mask=mask)

        # Store to KV cache
        tl.store(kv_cache_ptr + target_offset_k + offset, val_k, mask=mask)
        tl.store(kv_cache_ptr + target_offset_v + offset, val_v, mask=mask)


@triton.jit
def kv_cache_gather_kernel(
        # Pointers to KV cache tensors (flattened into one array)
        kv_cache_ptrs_ptr,  # pointer to array of pointers (size num_layers)
        dst_ptr,  # pointer to dst tensor (num_layers, 2, num_tokens, hidden_size)
        token_indices_ptr,  # pointer to token indices
        num_tokens_in_block,  # length of token_indices
        hidden_size,  # hidden dimension size
        total_token_in_kvcache,  # sequence length in KV cache
        num_layers,  # number of layers
        BLOCK_SIZE: tl.constexpr,
):
    # Get the current program's layer index and token position
    layer_idx = tl.program_id(0)
    token_pos = tl.program_id(1)

    # Boundary check
    if layer_idx >= num_layers or token_pos >= num_tokens_in_block:
        return

    # Get the token index to write to
    token_idx = tl.load(token_indices_ptr + token_pos)

    # Get the pointer to the current layer's KV cache
    kv_cache_ptr = tl.cast(tl.load(kv_cache_ptrs_ptr + layer_idx), dst_ptr.dtype)

    # Check if token_idx is valid
    if token_idx >= total_token_in_kvcache:
        return

    # Calculate dst offsets (for both K and V)
    dst_offset_k = (layer_idx * num_tokens_in_block * 2 + 0 * num_tokens_in_block + token_pos) * hidden_size
    dst_offset_v = (layer_idx * num_tokens_in_block * 2 + 1 * num_tokens_in_block + token_pos) * hidden_size

    # Calculate kvcache offsets (for both K and V)
    kvcache_offset_k = (0 * total_token_in_kvcache + token_idx) * hidden_size
    kvcache_offset_v = (1 * total_token_in_kvcache + token_idx) * hidden_size
    # Copy data in blocks
    for i in range(0, hidden_size, BLOCK_SIZE):
        offset = i + tl.arange(0, BLOCK_SIZE)
        mask = offset < hidden_size

        val_k = tl.load(kv_cache_ptr + kvcache_offset_k + offset, mask=mask)
        val_v = tl.load(kv_cache_ptr + kvcache_offset_v + offset, mask=mask)

        tl.store(dst_ptr + dst_offset_k + offset, val_k, mask=mask)
        tl.store(dst_ptr + dst_offset_v + offset, val_v, mask=mask)


def scatter_kv_caches(
        kv_caches_ptrs: torch.Tensor,
        # List of KV cache tensors ptr (each shape [2, total_token_in_kvcache, hidden_size])
        total_token_in_kvcache: int,  # total token in kv cache
        src_tensor: torch.Tensor,  # Shape [num_layers, 2, num_tokens_in_block, hidden_size]
        token_indices: List[int],  # List of token positions to update
):
    assert len(kv_caches_ptrs) == src_tensor.shape[0], "Number of layers mismatch"
    num_layers = len(kv_caches_ptrs)
    num_tokens_in_block = len(token_indices)
    hidden_size = src_tensor.shape[-1]

    # Prepare device tensors
    device = kv_caches_ptrs.device
    token_indices_tensor = torch.tensor(token_indices, dtype=torch.int32, device="cpu").to(device, non_blocking=True)

    # Calculate grid size
    grid = (num_layers, num_tokens_in_block)

    # Launch kernel
    BLOCK_SIZE = 128  # Tunable parameter
    kv_cache_scatter_kernel[grid](
        kv_caches_ptrs,
        src_tensor,
        token_indices_tensor,
        num_tokens_in_block,
        hidden_size,
        total_token_in_kvcache,
        num_layers,
        BLOCK_SIZE=BLOCK_SIZE,
    )


def gather_kv_caches(
        kv_caches_ptrs: torch.Tensor,
        # List of KV cache tensors ptr (each shape [2, total_token_in_kvcache, hidden_size])
        total_token_in_kvcache: int,  # total token in kv cache
        dst_tensor: torch.Tensor,  # Shape [num_layers, 2, num_tokens_in_block, hidden_size]
        token_indices: List[int],  # List of token positions to update
):
    assert kv_caches_ptrs.shape[0] == dst_tensor.shape[0], "Number of layers mismatch"
    num_layers = kv_caches_ptrs.shape[0]
    num_tokens_in_block = len(token_indices)
    hidden_size = dst_tensor.shape[-1]
    # Prepare device tensors
    device = kv_caches_ptrs.device
    token_indices_tensor = torch.tensor(token_indices, dtype=torch.int32, device="cpu").to(device, non_blocking=True)

    # Calculate grid size
    grid = (num_layers, num_tokens_in_block)

    # Launch kernel
    BLOCK_SIZE = 128  # Tunable parameter
    kv_cache_gather_kernel[grid](
        kv_caches_ptrs,
        dst_tensor,
        token_indices_tensor,
        num_tokens_in_block,
        hidden_size,
        total_token_in_kvcache,
        num_layers,
        BLOCK_SIZE=BLOCK_SIZE,
    )


class CopyBufferAllocator:
    def __init__(self, device, dtype, shape: List[int], max_count: int):
        if max_count <= 0:
            raise ValueError("max_count must be positive")

        self.max_count = max_count
        self.device = device
        self.dtype = dtype

        # 使用 Condition 变量实现线程同步与阻塞等待
        self._cond = threading.Condition()
        need_pin = self.device.type == "cpu"
        self._raw_buffer = torch.empty([max_count] + shape, dtype=dtype, device=device, pin_memory=need_pin)
        self._raw_buffer.zero_()

        self._free_idx_list = [i for i in range(max_count)]
        self._free_buffer_list = [self._raw_buffer[i] for i in range(max_count)]
        self.shape = self._free_buffer_list[0].shape
        self._inuse_count: int = 0

    # def alloc_buffer(self, count: int) -> Optional[List[torch.Tensor]]:
    #     """非阻塞分配：若资源不足，立即返回 None"""
    #     if count <= 0:
    #         raise ValueError("count must be positive")
    #     if count > self.max_count:
    #         raise ValueError(f"count ({count}) exceeds max_count ({self.max_count})")
    #
    #     with self._cond:
    #         if self._inuse_count + count <= self.max_count:
    #             self._inuse_count += count
    #             # 取后 count 个（避免频繁移动列表头部，提升效率）
    #             result = self._free_buffer_list[-count:]
    #             del self._free_buffer_list[-count:]
    #             return result
    #         return None

    def alloc_buffer_idx_blocking(self, count: int) -> List[int]:
        """阻塞式分配：若资源不足，阻塞等待直到满足需求"""
        if count <= 0:
            raise ValueError("count must be positive")
        if count > self.max_count:
            raise ValueError(f"count ({count}) exceeds max_count ({self.max_count})")

        with self._cond:
            # 循环等待——防止虚假唤醒
            while self._inuse_count + count > self.max_count:
                self._cond.wait()

            self._inuse_count += count
            result = self._free_idx_list[-count:]
            del self._free_idx_list[-count:]
            return result

    def get_buffer_by_idx(self, indices :List[int]):
        result = []
        for idx in indices:
            result.append(self._free_buffer_list[idx])
        return result

    def free_buffer(self, buffers: List[int]):

        if not buffers:
            return  # 空列表，直接返回

        with self._cond:
            for idx in buffers:
                if idx >= self.max_count or idx < 0:
                    raise ValueError(f"Buffer idx {idx} wrong")

            n = len(buffers)
            if self._inuse_count < n:
                raise RuntimeError(f"Trying to free {n} buffers, but only {self._inuse_count} are in use")

            self._inuse_count -= n
            self._free_idx_list.extend(buffers)

            # 唤醒所有等待者（或用 notify_all 更稳妥，因可能有多个等待不同 count 的线程）
            self._cond.notify_all()


def generate_test_data(num_layers=64, total_token_in_kvcache=1024, num_tokens_in_block=128, hidden_size=4096,
                       device="cuda"):
    # 生成KV caches (不连续的tensor列表)
    kv_caches = []
    for _ in range(num_layers):
        # 每个cache是[2, total_token_in_kvcache, hidden_size]
        cache = torch.randn(2, total_token_in_kvcache, hidden_size, device=device)
        kv_caches.append(cache)

    # 生成source tensor [num_layers, 2, num_tokens_in_block, hidden_size]
    source_tensor = torch.randn(num_layers, 2, num_tokens_in_block, hidden_size, device=device)

    # 生成token indices (确保不重复且在有效范围内)
    token_indices = torch.randperm(total_token_in_kvcache)[:num_tokens_in_block].tolist()

    return kv_caches, source_tensor, token_indices


def reference_impl(kv_caches: List[torch.Tensor], source_tensor: torch.Tensor, token_indices: List[int]):
    """参考实现，用于验证正确性"""
    for layer_idx in range(len(kv_caches)):
        kv_caches[layer_idx][:, token_indices, :] = source_tensor[layer_idx]


def main():
    """测试算子实现的正确性"""
    torch.manual_seed(42)
    num_layers = 4  # 测试时减少层数以加快速度
    total_token_in_kvcache = 128
    num_tokens_in_block = 32
    hidden_size = 512

    # 生成测试数据
    kv_caches, source_tensor, token_indices = generate_test_data(
        num_layers=num_layers,
        total_token_in_kvcache=total_token_in_kvcache,
        num_tokens_in_block=num_tokens_in_block,
        hidden_size=hidden_size
    )

    # 创建用于参考实现的拷贝
    kv_caches_ref = [cache.clone() for cache in kv_caches]

    # 运行参考实现
    reference_impl(kv_caches_ref, source_tensor, token_indices)

    kv_cache_ptrs = torch.tensor([cache.data_ptr() for cache in kv_caches], device="cuda", dtype=torch.int64)
    # 运行我们的实现
    scatter_kv_caches(kv_cache_ptrs, total_token_in_kvcache, source_tensor, token_indices)

    # 验证结果
    for layer_idx in range(num_layers):
        torch.testing.assert_close(
            kv_caches[layer_idx][0],
            kv_caches_ref[layer_idx][0],
            msg=f"Layer {layer_idx} key mismatch"
        )
        torch.testing.assert_close(
            kv_caches[layer_idx][1],
            kv_caches_ref[layer_idx][1],
            msg=f"Layer {layer_idx} value mismatch"
        )
    print("Correctness test passed!")


if __name__ == "__main__":
    main()
