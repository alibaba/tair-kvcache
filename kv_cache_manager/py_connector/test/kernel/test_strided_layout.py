"""
Unit tests for strided KV cache layout support.

These tests verify that the gather/scatter kernels correctly handle
strided memory layouts where K and V are interleaved within blocks.

Key scenarios:
1. Contiguous layout (baseline) - flat indexing
2. Strided layout - block-strided with separate K/V strides
3. V-pointer handling - ensure no double-offset bug
"""

import torch
import pytest

from kv_cache_manager.py_connector.kernel.batch_gather_scatter_helper import (
    batch_gather_kv_caches,
    batch_scatter_kv_caches,
)


def create_strided_kv_cache(num_blocks, block_size, num_heads, head_dim, device="cuda"):
    """
    Create a strided KV cache tensor with shape [2, num_blocks, block_size, num_heads, head_dim].
    
    Returns:
        cache: The full tensor [2, num_blocks, block_size, num_heads, head_dim]
        k_ptr: Pointer to K section (cache[0].data_ptr())
        v_ptr: Pointer to V section (cache[1].data_ptr())
        kv_stride: Stride between K and V (cache.stride(0))
        block_stride: Stride between blocks (cache.stride(1))
    """
    cache = torch.randn(
        2, num_blocks, block_size, num_heads, head_dim,
        device=device, dtype=torch.bfloat16
    )
    
    k_ptr = cache[0].data_ptr()
    v_ptr = cache[1].data_ptr()
    kv_stride = cache.stride(0)  # Distance from K to V
    block_stride = cache.stride(1)  # Distance between blocks
    
    return cache, k_ptr, v_ptr, kv_stride, block_stride


def test_batch_gather_strided_layout():
    """
    Test batch_gather_kv_caches with strided memory layout.
    
    This test verifies:
    1. Strided offset calculation is correct
    2. V-pointer is NOT double-offset (the bug we fixed)
    3. Gathered data matches expected values from strided cache
    """
    torch.manual_seed(42)
    
    # Test parameters - use small sizes for faster testing
    num_layers = 2
    num_blocks = 8
    block_size = 16  # local_block_size
    num_heads = 4
    head_dim = 32
    hidden_size = num_heads * head_dim  # 128
    
    # Create strided KV caches for each layer
    kv_caches = []
    kv_ptrs = []
    kv_strides = []
    block_strides = []
    
    for layer_idx in range(num_layers):
        cache, k_ptr, v_ptr, kv_stride, block_stride = create_strided_kv_cache(
            num_blocks, block_size, num_heads, head_dim
        )
        kv_caches.append(cache)
        kv_ptrs.extend([k_ptr, v_ptr])  # [K0, V0, K1, V1, ...]
        kv_strides.append(kv_stride)
        block_strides.append(block_stride)
    
    # Verify all layers have the same strides (they should)
    assert all(s == kv_strides[0] for s in kv_strides), "All layers must have same kv_stride"
    assert all(s == block_strides[0] for s in block_strides), "All layers must have same block_stride"
    
    kv_stride = kv_strides[0]
    block_stride = block_strides[0]
    
    # Create pointer tensor
    kv_ptrs_tensor = torch.tensor(kv_ptrs, device="cuda", dtype=torch.int64)
    
    # Create destination tensor
    total_blocks = 4
    num_tokens_per_block = block_size  # Gather full blocks
    dst_tensor = torch.zeros(
        total_blocks, num_layers * 2, num_tokens_per_block, hidden_size,
        device="cpu", dtype=torch.bfloat16, pin_memory=True
    )
    
    # Generate block token indices
    # Each block maps to a contiguous range in the cache
    block_token_indices = []
    for block_idx in range(total_blocks):
        # Map to physical block in cache
        cache_block_idx = block_idx % num_blocks
        start_token = cache_block_idx * block_size
        for token_offset in range(block_size):
            block_token_indices.append(start_token + token_offset)
    
    dst_block_indices = list(range(total_blocks))
    
    # Call batch_gather_kv_caches with strided parameters
    batch_gather_kv_caches(
        kv_ptrs_tensor,
        dst_tensor,
        block_token_indices,
        dst_block_indices,
        num_tokens_per_block,
        hidden_size,
        kv_stride=kv_stride,
        block_stride=block_stride,
        local_block_size=block_size,
    )
    
    # Verify results by comparing with reference implementation
    kv_caches_cpu = [cache.cpu() for cache in kv_caches]
    
    for block_idx in range(total_blocks):
        cache_block_idx = block_idx % num_blocks
        dst_block_idx = dst_block_indices[block_idx]
        
        for token_offset in range(block_size):
            token_idx_in_cache = cache_block_idx * block_size + token_offset
            
            for layer_idx in range(num_layers):
                # Gathered K
                gathered_k = dst_tensor[dst_block_idx, layer_idx * 2, token_offset, :]
                expected_k = kv_caches_cpu[layer_idx][0, cache_block_idx, token_offset, :, :].flatten()
                torch.testing.assert_close(
                    gathered_k, expected_k,
                    msg=f"K mismatch: block={block_idx}, layer={layer_idx}, token={token_offset}"
                )
                
                # Gathered V - this is where the double-offset bug would show up
                gathered_v = dst_tensor[dst_block_idx, layer_idx * 2 + 1, token_offset, :]
                expected_v = kv_caches_cpu[layer_idx][1, cache_block_idx, token_offset, :, :].flatten()
                torch.testing.assert_close(
                    gathered_v, expected_v,
                    msg=f"V mismatch (possible double-offset): block={block_idx}, layer={layer_idx}, token={token_offset}"
                )
    
    print("✓ Strided layout gather test passed")


def test_batch_scatter_strided_layout():
    """
    Test batch_scatter_kv_caches with strided memory layout.
    
    This test verifies:
    1. Strided offset calculation is correct for scatter
    2. V-pointer is NOT double-offset
    3. Scattered data is written to correct locations in strided cache
    """
    torch.manual_seed(42)
    
    # Test parameters
    num_layers = 2
    num_blocks = 8
    block_size = 16
    num_heads = 4
    head_dim = 32
    hidden_size = num_heads * head_dim
    
    # Create strided KV caches
    kv_caches = []
    kv_ptrs = []
    
    for layer_idx in range(num_layers):
        cache, k_ptr, v_ptr, kv_stride, block_stride = create_strided_kv_cache(
            num_blocks, block_size, num_heads, head_dim
        )
        # Zero out the cache to make verification easier
        cache.zero_()
        kv_caches.append(cache)
        kv_ptrs.extend([k_ptr, v_ptr])
    
    kv_stride = kv_caches[0].stride(0)
    block_stride = kv_caches[0].stride(1)
    
    kv_ptrs_tensor = torch.tensor(kv_ptrs, device="cuda", dtype=torch.int64)
    
    # Create source tensor with known data
    total_blocks = 4
    num_tokens_per_block = block_size
    src_tensor = torch.randn(
        total_blocks, num_layers * 2, num_tokens_per_block, hidden_size,
        device="cpu", dtype=torch.bfloat16, pin_memory=True
    )
    
    # Generate block token indices
    block_token_indices = []
    for block_idx in range(total_blocks):
        cache_block_idx = block_idx % num_blocks
        start_token = cache_block_idx * block_size
        for token_offset in range(block_size):
            block_token_indices.append(start_token + token_offset)
    
    src_block_indices = list(range(total_blocks))
    
    # Call batch_scatter_kv_caches
    batch_scatter_kv_caches(
        kv_ptrs_tensor,
        src_tensor,
        block_token_indices,
        src_block_indices,
        num_tokens_per_block,
        hidden_size,
        kv_stride=kv_stride,
        block_stride=block_stride,
        local_block_size=block_size,
    )
    
    # Verify results
    kv_caches_cpu = [cache.cpu() for cache in kv_caches]
    
    for block_idx in range(total_blocks):
        cache_block_idx = block_idx % num_blocks
        src_block_idx = src_block_indices[block_idx]
        
        for token_offset in range(block_size):
            for layer_idx in range(num_layers):
                # Expected K
                expected_k = src_tensor[src_block_idx, layer_idx * 2, token_offset, :]
                actual_k = kv_caches_cpu[layer_idx][0, cache_block_idx, token_offset, :, :].flatten()
                torch.testing.assert_close(
                    actual_k, expected_k,
                    msg=f"K scatter mismatch: block={block_idx}, layer={layer_idx}, token={token_offset}"
                )
                
                # Expected V
                expected_v = src_tensor[src_block_idx, layer_idx * 2 + 1, token_offset, :]
                actual_v = kv_caches_cpu[layer_idx][1, cache_block_idx, token_offset, :, :].flatten()
                torch.testing.assert_close(
                    actual_v, expected_v,
                    msg=f"V scatter mismatch (possible double-offset): block={block_idx}, layer={layer_idx}, token={token_offset}"
                )
    
    print("✓ Strided layout scatter test passed")


def test_v_pointer_no_double_offset():
    """
    Specific test to verify V-pointer is not double-offset.
    
    The bug was: V pointer already points to V's base (cache[1].data_ptr()),
    but the kernel was adding kv_stride again, causing it to read/write
    past V's actual location.
    
    This test creates a scenario where the bug would be obvious:
    - Small cache with distinct K and V values
    - Verify V data is gathered from correct location
    """
    torch.manual_seed(42)
    
    num_layers = 1
    num_blocks = 2
    block_size = 4
    num_heads = 2
    head_dim = 8
    hidden_size = num_heads * head_dim
    
    # Create cache with distinct K and V values
    cache = torch.zeros(2, num_blocks, block_size, num_heads, head_dim, 
                       device="cuda", dtype=torch.bfloat16)
    
    # Fill K with 1.0, V with 2.0
    cache[0].fill_(1.0)  # K
    cache[1].fill_(2.0)  # V
    
    k_ptr = cache[0].data_ptr()
    v_ptr = cache[1].data_ptr()
    kv_stride = cache.stride(0)
    block_stride = cache.stride(1)
    
    kv_ptrs_tensor = torch.tensor([k_ptr, v_ptr], device="cuda", dtype=torch.int64)
    
    # Gather one block
    dst_tensor = torch.zeros(1, 2, block_size, hidden_size, 
                            device="cpu", dtype=torch.bfloat16, pin_memory=True)
    
    block_token_indices = list(range(block_size))  # First block
    dst_block_indices = [0]
    
    batch_gather_kv_caches(
        kv_ptrs_tensor,
        dst_tensor,
        block_token_indices,
        dst_block_indices,
        block_size,
        hidden_size,
        kv_stride=kv_stride,
        block_stride=block_stride,
        local_block_size=block_size,
    )
    
    # Verify K is all 1.0
    gathered_k = dst_tensor[0, 0, :, :]
    assert torch.all(gathered_k == 1.0), f"K should be 1.0, got {gathered_k}"
    
    # Verify V is all 2.0 (this would fail with double-offset bug)
    gathered_v = dst_tensor[0, 1, :, :]
    assert torch.all(gathered_v == 2.0), f"V should be 2.0, got {gathered_v}"
    
    print("✓ V-pointer double-offset test passed")


def test_contiguous_layout_baseline():
    """
    Baseline test for contiguous layout (kv_stride=0, block_stride=0).
    
    This ensures the strided layout fix doesn't break the contiguous path.
    """
    torch.manual_seed(42)
    
    num_layers = 2
    total_tokens = 256
    block_size = 16
    num_heads = 4
    head_dim = 32
    hidden_size = num_heads * head_dim
    
    # Create contiguous KV caches
    kv_caches = []
    kv_ptrs = []
    
    for layer_idx in range(num_layers):
        cache = torch.randn(2, total_tokens, hidden_size, 
                           device="cuda", dtype=torch.bfloat16)
        kv_caches.append(cache)
        kv_ptrs.extend([cache[0].data_ptr(), cache[1].data_ptr()])
    
    kv_ptrs_tensor = torch.tensor(kv_ptrs, device="cuda", dtype=torch.int64)
    
    # Gather test
    total_blocks = 4
    dst_tensor = torch.zeros(total_blocks, num_layers * 2, block_size, hidden_size,
                            device="cpu", dtype=torch.bfloat16, pin_memory=True)
    
    block_token_indices = []
    for block_idx in range(total_blocks):
        start_token = block_idx * block_size
        for token_offset in range(block_size):
            block_token_indices.append(start_token + token_offset)
    
    dst_block_indices = list(range(total_blocks))
    
    batch_gather_kv_caches(
        kv_ptrs_tensor,
        dst_tensor,
        block_token_indices,
        dst_block_indices,
        block_size,
        hidden_size,
        # No stride parameters - use defaults (0)
    )
    
    # Verify
    kv_caches_cpu = [cache.cpu() for cache in kv_caches]
    
    for block_idx in range(total_blocks):
        dst_block_idx = dst_block_indices[block_idx]
        
        for token_offset in range(block_size):
            token_idx = block_idx * block_size + token_offset
            
            for layer_idx in range(num_layers):
                gathered_k = dst_tensor[dst_block_idx, layer_idx * 2, token_offset, :]
                expected_k = kv_caches_cpu[layer_idx][0, token_idx, :]
                torch.testing.assert_close(gathered_k, expected_k)
                
                gathered_v = dst_tensor[dst_block_idx, layer_idx * 2 + 1, token_offset, :]
                expected_v = kv_caches_cpu[layer_idx][1, token_idx, :]
                torch.testing.assert_close(gathered_v, expected_v)
    
    print("✓ Contiguous layout baseline test passed")


if __name__ == "__main__":
    print("Running strided layout tests...")
    print()
    
    test_contiguous_layout_baseline()
    test_v_pointer_no_double_offset()
    test_batch_gather_strided_layout()
    test_batch_scatter_strided_layout()
    
    print()
    print("All strided layout tests passed! ✓")
