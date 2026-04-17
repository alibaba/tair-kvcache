"""Integration tests for hybrid (KV + NSA Indexer) scenarios.

Requires a running KV Cache Manager process and CUDA device.
Mirrors test_linear.py structure but replaces Mamba with Indexer.
"""
import logging
logging.basicConfig(level=logging.DEBUG)

import subprocess
import signal
import time
import os
import atexit
from types import SimpleNamespace

import torch
from sglang.srt.mem_cache.hicache_storage import (
    HiCacheStorageConfig,
    HiCacheStorageExtraInfo,
    PoolName,
    PoolTransfer,
    PoolHitPolicy,
)
from sglang.srt.mem_cache.utils import get_hash_str
from sglang.srt.mem_cache.memory_pool import MHATokenToKVPool
from sglang.srt.mem_cache.memory_pool_host import MHATokenToKVPoolHost
from sglang.srt.distributed import (
    init_distributed_environment,
    initialize_model_parallel,
)
from kv_cache_manager.py_connector.sglang.connector import HiCacheKVCM

logger = logging.getLogger(__name__)


# ---- Distributed init ----
init_distributed_environment(
    world_size=1,
    rank=0,
    distributed_init_method="tcp://127.0.0.1:23458",
    local_rank=0,
    backend="gloo",
)

initialize_model_parallel(
    tensor_model_parallel_size=1,
    pipeline_model_parallel_size=1,
)

# ---- Configuration ----
model_name = "xxx"
max_total_num_tokens = 8192
page_size = 64
kv_cache_dtype = torch.bfloat16
layer_num = 64
head_num, head_dim = 8, 128
device = "cuda"
hicache_ratio = 2
hicache_size = 0
hicache_mem_layout = "page_first_direct"

# Indexer params (mimic NSAIndexerPoolHost)
indexer_layer_num = 64
indexer_index_head_dim = 128
indexer_quant_block_size = 128
indexer_dtype = torch.uint8
indexer_page_num = 256

manager_uri = os.environ.get("KVCM_URI", "http://127.0.0.1:6382")
kvcm_home = os.environ.get("KVCM_HOME", "/home/admin/kv_cache_manager")

proc = None


def _stop_manager():
    global proc
    if proc and proc.poll() is None:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
    if proc:
        while os.path.exists(f"/proc/{proc.pid}"):
            time.sleep(1)
    logger.info("kv_cache_manager fully stopped.")


def _start_manager():
    global proc
    subprocess.run("rm -rf /root/KVCacheManager/logs/*", shell=True, check=False)
    proc = subprocess.Popen([
        f"{kvcm_home}/bin/kv_cache_manager_bin",
        "-c", f"{kvcm_home}/etc/default_server_config.conf",
        "-l", f"{kvcm_home}/etc/default_logger_config.conf"
    ])
    signal.signal(signal.SIGINT, _stop_manager)
    signal.signal(signal.SIGTERM, _stop_manager)
    atexit.register(_stop_manager)
    logger.info(f"kv_cache_manager {proc=}")
    return proc


# ============================================================
# Mock Indexer Pool
# ============================================================

class MockIndexerPoolHost:
    """Lightweight mock of NSAIndexerPoolHost for testing.

    Allocates a flat buffer with the same layout as the real pool
    (page_first_direct) and provides get_page_buffer_meta / get_size_per_token
    so the connector can do zero-copy I/O.
    """

    def __init__(self, page_num, page_size_val, layer_num_val,
                 index_head_dim, quant_block_size, dtype):
        self.page_num = page_num
        self.page_size = page_size_val
        self.layer_num = layer_num_val
        self.index_head_dim = index_head_dim
        self.indexer_quant_block_size = quant_block_size
        self.indexer_dtype = dtype
        self.dtype = dtype
        self.layout = "page_first_direct"

        self.indexer_size_per_token = (
            index_head_dim
            + index_head_dim // quant_block_size * 4
        )
        self.indexer_page_stride_size = (
            self.indexer_size_per_token * page_size_val * dtype.itemsize
        )

        # Flat buffer: [page_num, layer_num, 1, indexer_page_stride_size]
        self.index_k_with_scale_buffer = torch.zeros(
            (page_num, layer_num_val, 1, self.indexer_page_stride_size),
            dtype=dtype,
            device="cpu",
            pin_memory=True,
        )

    def get_size_per_token(self):
        return (
            self.indexer_size_per_token
            * self.layer_num
            * self.indexer_dtype.itemsize
        )

    def get_page_buffer_meta(self, indices):
        """Return (ptr_list, size_list) for zero-copy I/O.

        indices are token-granularity (page_size aligned); each page_size
        tokens map to one page-level pointer.
        """
        indices = indices.tolist()
        ptr_list = []
        page_stride_bytes = (
            self.layer_num
            * self.indexer_page_stride_size
            * self.indexer_dtype.itemsize
        )
        base_ptr = self.index_k_with_scale_buffer.data_ptr()
        for i in range(0, len(indices), self.page_size):
            page_index = int(indices[i]) // self.page_size
            ptr_list.append(base_ptr + page_index * page_stride_bytes)
        return ptr_list, [page_stride_bytes] * len(ptr_list)


# ============================================================
# Setup helpers
# ============================================================

def _create_pools():
    device_pool = MHATokenToKVPool(
        size=max_total_num_tokens,
        page_size=page_size,
        dtype=kv_cache_dtype,
        head_num=head_num,
        head_dim=head_dim,
        layer_num=layer_num,
        device=device,
        enable_memory_saver=False,
    )
    kv_pool_host = MHATokenToKVPoolHost(
        device_pool=device_pool,
        host_to_device_ratio=hicache_ratio,
        host_size=hicache_size,
        page_size=page_size,
        layout=hicache_mem_layout,
    )
    indexer_pool = MockIndexerPoolHost(
        page_num=indexer_page_num,
        page_size_val=page_size,
        layer_num_val=indexer_layer_num,
        index_head_dim=indexer_index_head_dim,
        quant_block_size=indexer_quant_block_size,
        dtype=indexer_dtype,
    )
    return device_pool, kv_pool_host, indexer_pool


def _create_indexer_backend(kv_pool_host, indexer_pool, instance_id, num_blocks=10):
    """Create HiCacheKVCM with KV + Indexer pools registered."""
    entries = [
        SimpleNamespace(
            name=PoolName.KV,
            host_pool=kv_pool_host,
            is_primary_index_anchor=True,
        ),
        SimpleNamespace(
            name=PoolName.INDEXER,
            host_pool=indexer_pool,
            is_primary_index_anchor=False,
        ),
    ]
    host_pool_group = SimpleNamespace(
        entries=entries,
        dtype=kv_pool_host.dtype,
        layer_num=kv_pool_host.layer_num,
        page_size=kv_pool_host.page_size,
        page_num=kv_pool_host.page_num,
        kv_buffer=kv_pool_host.kv_buffer,
    )
    host_pool_group.get_size_per_token = kv_pool_host.get_size_per_token
    host_pool_group.get_page_buffer_meta = kv_pool_host.get_page_buffer_meta

    storage_config = HiCacheStorageConfig(
        tp_rank=0,
        tp_size=1,
        pp_rank=0,
        pp_size=1,
        attn_cp_rank=0,
        attn_cp_size=1,
        is_mla_model=False,  # MHATokenToKVPoolHost uses kv_factor=2 (separate K/V)
        enable_storage_metrics=False,
        is_page_first_layout=True,
        model_name=model_name,
        extra_config={
            "manager_uri": manager_uri,
            "instance_group": "default",
            "instance_id": instance_id,
        },
    )

    storage_backend = HiCacheKVCM(storage_config, {})
    storage_backend.register_mem_pool_host(host_pool_group)

    token_ids = list(range(num_blocks * page_size))
    block_hashes = []
    block_hash = None
    kv_host_indices = []
    for i in range(0, num_blocks * page_size, page_size):
        block_hash = get_hash_str(token_ids[i:i + page_size], block_hash)
        block_hashes.append(block_hash)
        kv_host_indices.extend(range(i, i + page_size))

    # Indexer host_indices: page-aligned token indices (same as KV)
    indexer_host_indices = list(range(num_blocks * page_size))

    return (storage_backend, block_hashes, kv_host_indices, indexer_host_indices)


def _fill_kv_buffer(kv_pool_host, num_blocks):
    for i in range(num_blocks * page_size):
        page_id = i // page_size
        token_id = i % page_size
        kv_pool_host.kv_buffer[:, page_id, :, token_id] = torch.tensor(
            i, dtype=torch.bfloat16
        )


def _fill_indexer_buffer(indexer_pool, num_blocks):
    """Fill indexer buffer with identifiable byte pattern."""
    for p in range(num_blocks):
        indexer_pool.index_k_with_scale_buffer[p] = (p + 1) % 256


# ============================================================
# Tests
# ============================================================

def test_indexer_hybrid(kv_pool_host, indexer_pool):
    """Basic hybrid: set_v2 (Indexer) + set_v1 (KV), then exists_v2, get_v2 with data verification."""
    (storage_backend,
     block_hashes, kv_host_indices, indexer_host_indices) = _create_indexer_backend(
        kv_pool_host, indexer_pool, "indexer_hybrid_0"
    )
    num_blocks = len(block_hashes)

    _fill_kv_buffer(kv_pool_host, num_blocks)
    _fill_indexer_buffer(indexer_pool, num_blocks)

    indexer_transfer = PoolTransfer(
        name=PoolName.INDEXER,
        host_indices=torch.tensor(indexer_host_indices),
        keys=block_hashes,
        hit_policy=PoolHitPolicy.ALL_PAGES,
    )

    # Initially empty
    exist_result = storage_backend.batch_exists_v2(block_hashes, [indexer_transfer])
    assert exist_result.kv_hit_pages == 0, (
        f"Expected 0 KV hits, got {exist_result.kv_hit_pages}"
    )

    # Write Indexer then KV
    set_v2_result = storage_backend.batch_set_v2([indexer_transfer])
    assert PoolName.INDEXER in set_v2_result

    set_v1_result = storage_backend.batch_set_v1(
        block_hashes, torch.tensor(kv_host_indices)
    )
    assert all(set_v1_result), f"batch_set_v1 failed: {set_v1_result}"

    # Both exist
    exist_result = storage_backend.batch_exists_v2(block_hashes, [indexer_transfer])
    assert exist_result.kv_hit_pages == num_blocks, (
        f"Expected {num_blocks} KV hits, got {exist_result.kv_hit_pages}"
    )
    indexer_hits = exist_result.extra_pool_hit_pages.get(PoolName.INDEXER, 0)
    assert indexer_hits == num_blocks, (
        f"Expected {num_blocks} Indexer hits, got {indexer_hits}"
    )

    # Read Indexer back after clearing
    orig_buf = indexer_pool.index_k_with_scale_buffer[:num_blocks].clone()
    indexer_pool.index_k_with_scale_buffer[:num_blocks].zero_()

    get_v2_result = storage_backend.batch_get_v2([indexer_transfer])
    assert PoolName.INDEXER in get_v2_result
    assert all(get_v2_result[PoolName.INDEXER]), (
        f"batch_get_v2 failed: {get_v2_result[PoolName.INDEXER]}"
    )

    for i in range(num_blocks):
        assert torch.equal(
            indexer_pool.index_k_with_scale_buffer[i], orig_buf[i]
        ), f"Indexer data mismatch at page {i}"

    logger.info("test_indexer_hybrid passed!")


def test_indexer_write_kv_only(kv_pool_host, indexer_pool):
    """Write KV only -> batch_exists_v2 shows KV hit but Indexer boundary=0."""
    (storage_backend,
     block_hashes, kv_host_indices, indexer_host_indices) = _create_indexer_backend(
        kv_pool_host, indexer_pool, "indexer_kv_only_0"
    )
    num_blocks = len(block_hashes)
    _fill_kv_buffer(kv_pool_host, num_blocks)

    set_result = storage_backend.batch_set_v1(
        block_hashes, torch.tensor(kv_host_indices)
    )
    assert all(set_result)

    indexer_transfer = PoolTransfer(
        name=PoolName.INDEXER,
        host_indices=torch.tensor(indexer_host_indices),
        keys=block_hashes,
        hit_policy=PoolHitPolicy.ALL_PAGES,
    )
    result = storage_backend.batch_exists_v2(block_hashes, [indexer_transfer])

    assert result.extra_pool_hit_pages.get(PoolName.KV, 0) == num_blocks, (
        f"Expected KV pool hit {num_blocks}, got {result.extra_pool_hit_pages}"
    )
    indexer_boundary = result.extra_pool_hit_pages.get(PoolName.INDEXER, 0)
    assert indexer_boundary == 0, (
        f"Expected Indexer boundary 0 (no Indexer spec written), got {indexer_boundary}"
    )
    # ALL_PAGES: Indexer missing -> final_pages truncated to 0
    assert result.kv_hit_pages == 0, (
        f"Expected final_pages=0, got {result.kv_hit_pages}"
    )

    logger.info("test_indexer_write_kv_only passed!")


def test_indexer_write_indexer_only(kv_pool_host, indexer_pool):
    """Write Indexer only (no KV) -> kv_hit_pages should be 0."""
    (storage_backend,
     block_hashes, kv_host_indices, indexer_host_indices) = _create_indexer_backend(
        kv_pool_host, indexer_pool, "indexer_only_0"
    )
    num_blocks = len(block_hashes)
    _fill_indexer_buffer(indexer_pool, num_blocks)

    indexer_transfer = PoolTransfer(
        name=PoolName.INDEXER,
        host_indices=torch.tensor(indexer_host_indices),
        keys=block_hashes,
        hit_policy=PoolHitPolicy.ALL_PAGES,
    )
    set_v2_result = storage_backend.batch_set_v2([indexer_transfer])
    assert PoolName.INDEXER in set_v2_result

    result = storage_backend.batch_exists_v2(block_hashes, [indexer_transfer])
    assert result.kv_hit_pages == 0, (
        f"Expected kv_hit_pages=0 (KV not written), got {result.kv_hit_pages}"
    )

    logger.info("test_indexer_write_indexer_only passed!")


def test_indexer_kv_first_then_indexer(kv_pool_host, indexer_pool):
    """Write KV first, then Indexer. Both should succeed and verify data."""
    (storage_backend,
     block_hashes, kv_host_indices, indexer_host_indices) = _create_indexer_backend(
        kv_pool_host, indexer_pool, "indexer_cross_order_0"
    )
    num_blocks = len(block_hashes)
    _fill_kv_buffer(kv_pool_host, num_blocks)
    _fill_indexer_buffer(indexer_pool, num_blocks)

    # Write KV first
    set_v1_result = storage_backend.batch_set_v1(
        block_hashes, torch.tensor(kv_host_indices)
    )
    assert all(set_v1_result), f"batch_set_v1 failed: {set_v1_result}"

    # Write Indexer second
    indexer_transfer = PoolTransfer(
        name=PoolName.INDEXER,
        host_indices=torch.tensor(indexer_host_indices),
        keys=block_hashes,
        hit_policy=PoolHitPolicy.ALL_PAGES,
    )
    set_v2_result = storage_backend.batch_set_v2([indexer_transfer])
    assert PoolName.INDEXER in set_v2_result
    assert all(set_v2_result[PoolName.INDEXER]), (
        f"batch_set_v2 failed: {set_v2_result[PoolName.INDEXER]}"
    )

    # Verify both exist
    result = storage_backend.batch_exists_v2(block_hashes, [indexer_transfer])
    assert result.kv_hit_pages == num_blocks
    assert result.extra_pool_hit_pages.get(PoolName.INDEXER, 0) == num_blocks

    # Verify KV data
    index_shift = 2048
    shifted_indices = [v + index_shift for v in kv_host_indices]
    get_result = storage_backend.batch_get_v1(
        block_hashes, torch.tensor(shifted_indices)
    )
    assert all(get_result)

    for i in range(num_blocks * page_size):
        page_id = (i + index_shift) // page_size
        token_id = (i + index_shift) % page_size
        bf16_i = torch.tensor(i, dtype=torch.bfloat16)
        tensor_i = kv_pool_host.kv_buffer[:, page_id, :, token_id]
        assert torch.mean(tensor_i).item() == bf16_i.item(), (
            f"KV data mismatch at i={i}: actual={torch.mean(tensor_i).item()} expected={bf16_i.item()}"
        )

    # Verify Indexer data
    orig_buf = indexer_pool.index_k_with_scale_buffer[:num_blocks].clone()
    indexer_pool.index_k_with_scale_buffer[:num_blocks].zero_()

    get_v2_result = storage_backend.batch_get_v2([indexer_transfer])
    assert all(get_v2_result[PoolName.INDEXER])

    for i in range(num_blocks):
        assert torch.equal(
            indexer_pool.index_k_with_scale_buffer[i], orig_buf[i]
        ), f"Indexer data mismatch at page {i}"

    logger.info("test_indexer_kv_first_then_indexer passed!")


def test_indexer_full_round_trip(kv_pool_host, indexer_pool):
    """Complete write-clear-read-verify cycle for both KV and Indexer."""
    (storage_backend,
     block_hashes, kv_host_indices, indexer_host_indices) = _create_indexer_backend(
        kv_pool_host, indexer_pool, "indexer_full_rt_0"
    )
    num_blocks = len(block_hashes)
    _fill_kv_buffer(kv_pool_host, num_blocks)
    _fill_indexer_buffer(indexer_pool, num_blocks)

    indexer_transfer = PoolTransfer(
        name=PoolName.INDEXER,
        host_indices=torch.tensor(indexer_host_indices),
        keys=block_hashes,
        hit_policy=PoolHitPolicy.ALL_PAGES,
    )

    # Write both
    storage_backend.batch_set_v2([indexer_transfer])
    storage_backend.batch_set_v1(block_hashes, torch.tensor(kv_host_indices))

    # Exists check
    result = storage_backend.batch_exists_v2(block_hashes, [indexer_transfer])
    assert result.kv_hit_pages == num_blocks

    # Save originals
    orig_indexer = indexer_pool.index_k_with_scale_buffer[:num_blocks].clone()

    # Clear both buffers
    index_shift = 3072
    shifted_kv = [v + index_shift for v in kv_host_indices]
    indexer_pool.index_k_with_scale_buffer[:num_blocks].zero_()

    # Read back
    get_v1 = storage_backend.batch_get_v1(block_hashes, torch.tensor(shifted_kv))
    assert all(get_v1)
    get_v2 = storage_backend.batch_get_v2([indexer_transfer])
    assert all(get_v2[PoolName.INDEXER])

    # Verify KV
    for i in range(num_blocks * page_size):
        page_id = (i + index_shift) // page_size
        token_id = (i + index_shift) % page_size
        bf16_i = torch.tensor(i, dtype=torch.bfloat16)
        tensor_i = kv_pool_host.kv_buffer[:, page_id, :, token_id]
        actual = torch.mean(tensor_i).item()
        expected = bf16_i.item()
        if actual != expected:
            print(f"[KV-VERIFY] i={i} page_id={page_id} token_id={token_id} actual={actual} expected={expected} tensor_shape={tensor_i.shape} sample={tensor_i.flatten()[:5]}")
            assert False, f"KV data mismatch at i={i}: actual={actual} expected={expected}"

    # Verify Indexer
    for i in range(num_blocks):
        assert torch.equal(
            indexer_pool.index_k_with_scale_buffer[i], orig_indexer[i]
        ), f"Indexer data mismatch at page {i}"

    logger.info("test_indexer_full_round_trip passed!")


def test_indexer_all_pages_policy(kv_pool_host, indexer_pool):
    """Test ALL_PAGES boundary: only write Indexer for last N blocks,
    ALL_PAGES should truncate at first missing page."""
    (storage_backend,
     block_hashes, kv_host_indices, indexer_host_indices) = _create_indexer_backend(
        kv_pool_host, indexer_pool, "indexer_allpages_0"
    )
    num_blocks = len(block_hashes)
    _fill_kv_buffer(kv_pool_host, num_blocks)
    _fill_indexer_buffer(indexer_pool, num_blocks)

    # Write all KV blocks
    set_v1 = storage_backend.batch_set_v1(
        block_hashes, torch.tensor(kv_host_indices)
    )
    assert all(set_v1)

    # Write Indexer only for last 3 blocks
    trailing_n = 3
    trailing_keys = block_hashes[-trailing_n:]
    trailing_indexer_indices = indexer_host_indices[-trailing_n * page_size:]
    trailing_transfer = PoolTransfer(
        name=PoolName.INDEXER,
        host_indices=torch.tensor(trailing_indexer_indices),
        keys=trailing_keys,
        hit_policy=PoolHitPolicy.ALL_PAGES,
    )
    set_v2 = storage_backend.batch_set_v2([trailing_transfer])
    assert PoolName.INDEXER in set_v2
    assert all(set_v2[PoolName.INDEXER])

    # Query with ALL_PAGES
    query_transfer = PoolTransfer(
        name=PoolName.INDEXER,
        host_indices=torch.tensor(indexer_host_indices),
        keys=block_hashes,
        hit_policy=PoolHitPolicy.ALL_PAGES,
    )
    result = storage_backend.batch_exists_v2(block_hashes, [query_transfer])

    # KV hits all blocks
    assert result.extra_pool_hit_pages.get(PoolName.KV, 0) == num_blocks

    # ALL_PAGES: first 7 blocks lack Indexer spec -> boundary = 0
    indexer_boundary = result.extra_pool_hit_pages.get(PoolName.INDEXER, 0)
    assert indexer_boundary == 0, (
        f"Expected ALL_PAGES boundary 0 (first block lacks Indexer spec), "
        f"got {indexer_boundary}"
    )

    # Now write Indexer for ALL blocks and re-check
    full_transfer = PoolTransfer(
        name=PoolName.INDEXER,
        host_indices=torch.tensor(indexer_host_indices),
        keys=block_hashes,
        hit_policy=PoolHitPolicy.ALL_PAGES,
    )
    set_v2_full = storage_backend.batch_set_v2([full_transfer])
    assert all(set_v2_full[PoolName.INDEXER])

    result_full = storage_backend.batch_exists_v2(block_hashes, [full_transfer])
    assert result_full.kv_hit_pages == num_blocks
    assert result_full.extra_pool_hit_pages.get(PoolName.INDEXER, 0) == num_blocks

    logger.info("test_indexer_all_pages_policy passed!")


def test_indexer_partial_write_buffer_indexing(kv_pool_host, indexer_pool):
    """Verify correct buffer indexing when save_indices is a proper subset.

    Write Indexer for first N blocks, then write ALL 2N blocks. The second call
    triggers a partial write (only the new blocks). With the old buffer slicing
    bug, data from buffer positions 0..N-1 would be written to storage for
    blocks N..2N-1 instead of the correct buffer positions N..2N-1.
    """
    num_blocks = 6
    first_n = 3

    (storage_backend,
     block_hashes, kv_host_indices, indexer_host_indices) = _create_indexer_backend(
        kv_pool_host, indexer_pool, "indexer_partial_idx_0", num_blocks=num_blocks
    )

    _fill_indexer_buffer(indexer_pool, num_blocks)

    # Step 1: Write Indexer for first 3 blocks — all new, save_indices = [0,1,2]
    first_transfer = PoolTransfer(
        name=PoolName.INDEXER,
        host_indices=torch.tensor(indexer_host_indices[:first_n * page_size]),
        keys=block_hashes[:first_n],
        hit_policy=PoolHitPolicy.ALL_PAGES,
    )
    set_result_1 = storage_backend.batch_set_v2([first_transfer])
    assert PoolName.INDEXER in set_result_1
    assert all(set_result_1[PoolName.INDEXER]), (
        f"First batch_set_v2 failed: {set_result_1[PoolName.INDEXER]}"
    )

    # Step 2: Write Indexer for ALL 6 blocks.
    # Manager recognises blocks 0-2 as cached -> save_indices = [3,4,5].
    # Old bug: buffer[0:3] saved to storage for blocks 3-5 (wrong positions).
    full_transfer = PoolTransfer(
        name=PoolName.INDEXER,
        host_indices=torch.tensor(indexer_host_indices),
        keys=block_hashes,
        hit_policy=PoolHitPolicy.ALL_PAGES,
    )
    set_result_2 = storage_backend.batch_set_v2([full_transfer])
    assert PoolName.INDEXER in set_result_2

    # Step 3: Clear buffer and read back ALL blocks
    orig_buf = indexer_pool.index_k_with_scale_buffer[:num_blocks].clone()
    indexer_pool.index_k_with_scale_buffer[:num_blocks].zero_()

    get_result = storage_backend.batch_get_v2([full_transfer])
    assert PoolName.INDEXER in get_result
    assert all(get_result[PoolName.INDEXER]), (
        f"batch_get_v2 failed: {get_result[PoolName.INDEXER]}"
    )

    # Step 4: Verify each page has its own correct data (not swapped).
    # With the old bug, pages 3-5 would contain data from pages 0-2.
    for i in range(num_blocks):
        assert torch.equal(
            indexer_pool.index_k_with_scale_buffer[i], orig_buf[i]
        ), (
            f"Indexer mismatch at page {i}: "
            f"expected fill value {(i + 1) % 256}"
        )

    logger.info("test_indexer_partial_write_buffer_indexing passed!")


if __name__ == "__main__":
    logger.info("Starting hybrid (KV + Indexer) tests")

    try:
        _start_manager()
        _device_pool, _kv_pool_host, _indexer_pool = _create_pools()

        test_indexer_hybrid(_kv_pool_host, _indexer_pool)
        test_indexer_write_kv_only(_kv_pool_host, _indexer_pool)
        test_indexer_write_indexer_only(_kv_pool_host, _indexer_pool)
        test_indexer_kv_first_then_indexer(_kv_pool_host, _indexer_pool)
        test_indexer_full_round_trip(_kv_pool_host, _indexer_pool)
        test_indexer_all_pages_policy(_kv_pool_host, _indexer_pool)
        test_indexer_partial_write_buffer_indexing(_kv_pool_host, _indexer_pool)
        logger.info("All Indexer tests passed successfully!")
    except Exception as e:
        logger.error(f"Test failed with error: {e}")
        raise
    finally:
        _stop_manager()
