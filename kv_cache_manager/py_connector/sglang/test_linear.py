"""Integration tests for hybrid (KV + Mamba/Linear) scenarios.

Requires a running KV Cache Manager process and CUDA device.
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
from sglang.srt.mem_cache.memory_pool import MHATokenToKVPool, MambaPool
from sglang.srt.mem_cache.memory_pool_host import MHATokenToKVPoolHost, MambaPoolHost
from sglang.srt.configs.mamba_utils import Mamba2CacheParams, Mamba2StateShape
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
    distributed_init_method="tcp://127.0.0.1:23457",
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
mamba_num_pages = 256
mamba_layers = [0, 1]
mamba_intermediate_size = 256
mamba_n_groups = 1
mamba_num_heads = 2
mamba_head_dim = 16
mamba_state_size = 16
mamba_conv_kernel = 4

manager_uri = os.environ.get("KVCM_URI", "http://127.0.0.1:6382")
kvcm_home = os.environ.get("KVCM_HOME", "/home/admin/kv_cache_manager")

# Global process reference
proc = None


def _stop_manager():
    """Stop the KV Cache Manager process."""
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
    """Start the KV Cache Manager process."""
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


def _create_pools():
    """Create shared KV + Mamba pools (call once, reuse across tests)."""
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
    mamba_shape = Mamba2StateShape.create(
        tp_world_size=1,
        intermediate_size=mamba_intermediate_size,
        n_groups=mamba_n_groups,
        num_heads=mamba_num_heads,
        head_dim=mamba_head_dim,
        state_size=mamba_state_size,
        conv_kernel=mamba_conv_kernel,
    )
    mamba_cache_params = Mamba2CacheParams(
        shape=mamba_shape,
        layers=mamba_layers,
    )
    mamba_device_pool = MambaPool(
        size=mamba_num_pages,
        spec_state_size=0,
        cache_params=mamba_cache_params,
        mamba_layer_ids=mamba_layers,
        device=device,
    )
    mamba_pool = MambaPoolHost(
        device_pool=mamba_device_pool,
        host_to_device_ratio=hicache_ratio,
        host_size=0,
        layout=hicache_mem_layout,
    )
    return device_pool, kv_pool_host, mamba_pool


def _create_hybrid_backend(kv_pool_host, mamba_pool, instance_id, num_blocks=10):
    """Create a HiCacheKVCM with both KV and Mamba pools registered.

    Uses shared pools from _create_pools().
    Returns (storage_backend, block_hashes, kv_host_indices, mamba_host_indices).
    """
    entries = [
        SimpleNamespace(
            name=PoolName.KV,
            host_pool=kv_pool_host,
            is_primary_index_anchor=True,
        ),
        SimpleNamespace(
            name=PoolName.MAMBA,
            host_pool=mamba_pool,
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
        is_mla_model=False,
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

    # Generate test data
    token_ids = list(range(num_blocks * page_size))
    block_hashes = []
    block_hash = None
    kv_host_indices = []
    for i in range(0, num_blocks * page_size, page_size):
        block_hash = get_hash_str(token_ids[i:i + page_size], block_hash)
        block_hashes.append(block_hash)
        kv_host_indices.extend(range(i, i + page_size))

    mamba_host_indices = list(range(num_blocks))

    return (storage_backend, block_hashes, kv_host_indices, mamba_host_indices)


def _fill_kv_buffer(kv_pool_host, num_blocks):
    """Fill KV buffer with identifiable data."""
    for i in range(num_blocks * page_size):
        page_id = i // page_size
        token_id = i % page_size
        kv_pool_host.kv_buffer[:, page_id, :, token_id] = torch.tensor(
            i, dtype=torch.bfloat16
        )


def _fill_mamba_buffer(mamba_pool):
    """Fill Mamba buffer with identifiable data."""
    for i in range(mamba_pool.page_num):
        mamba_pool.temporal_buffer[i] = float(i + 1)
        for conv_buf in mamba_pool.conv_buffer:
            conv_buf[i] = float(i + 100)


# ============================================================
# Tests
# ============================================================


def test_hybrid(kv_pool_host, mamba_pool):
    """Basic hybrid test (migrated from test.py):
    set_v2 (Mamba) + set_v1 (KV), then exists_v2, get_v2 with data verification."""
    (storage_backend,
     block_hashes, kv_host_indices, mamba_host_indices) = _create_hybrid_backend(kv_pool_host, mamba_pool,"hybrid_0")
    num_blocks = len(block_hashes)

    _fill_kv_buffer(kv_pool_host, num_blocks)
    _fill_mamba_buffer(mamba_pool)

    mamba_transfer = PoolTransfer(
        name=PoolName.MAMBA,
        host_indices=torch.tensor(mamba_host_indices),
        keys=block_hashes,
        hit_policy=PoolHitPolicy.TRAILING_PAGES,
    )

    # Initially empty
    exist_result = storage_backend.batch_exists_v2(block_hashes, [mamba_transfer])
    assert exist_result.kv_hit_pages == 0, (
        f"Expected 0 KV hits, got {exist_result.kv_hit_pages}"
    )

    # Write Mamba then KV
    set_v2_result = storage_backend.batch_set_v2([mamba_transfer])
    assert PoolName.MAMBA in set_v2_result

    set_v1_result = storage_backend.batch_set_v1(
        block_hashes, torch.tensor(kv_host_indices)
    )
    assert all(set_v1_result), f"batch_set_v1 failed: {set_v1_result}"

    # Both exist
    exist_result = storage_backend.batch_exists_v2(block_hashes, [mamba_transfer])
    assert exist_result.kv_hit_pages == num_blocks, (
        f"Expected {num_blocks} KV hits, got {exist_result.kv_hit_pages}"
    )
    mamba_hits = exist_result.extra_pool_hit_pages.get(PoolName.MAMBA, 0)
    assert mamba_hits == num_blocks, (
        f"Expected {num_blocks} Mamba hits, got {mamba_hits}"
    )

    # Read Mamba back after clearing
    orig_temporal = mamba_pool.temporal_buffer[:num_blocks].clone()
    orig_conv = [conv[:num_blocks].clone() for conv in mamba_pool.conv_buffer]

    mamba_pool.temporal_buffer[:num_blocks].zero_()
    for conv_buf in mamba_pool.conv_buffer:
        conv_buf[:num_blocks].zero_()

    get_v2_result = storage_backend.batch_get_v2([mamba_transfer])
    assert PoolName.MAMBA in get_v2_result
    assert all(get_v2_result[PoolName.MAMBA]), (
        f"batch_get_v2 failed: {get_v2_result[PoolName.MAMBA]}"
    )

    for i in range(num_blocks):
        assert torch.allclose(
            mamba_pool.temporal_buffer[i], orig_temporal[i]
        ), f"Temporal data mismatch at page {i}"
        for j, conv_buf in enumerate(mamba_pool.conv_buffer):
            assert torch.allclose(
                conv_buf[i], orig_conv[j][i]
            ), f"Conv {j} data mismatch at page {i}"

    logger.info("test_hybrid passed!")


def test_write_kv_only_then_exists_v2(kv_pool_host, mamba_pool):
    """Write KV only (no Mamba) -> batch_exists_v2 shows KV hit but Mamba boundary=0."""
    (storage_backend,
     block_hashes, kv_host_indices, mamba_host_indices) = _create_hybrid_backend(kv_pool_host, mamba_pool,
        "linear_kv_only_0"
    )
    num_blocks = len(block_hashes)
    _fill_kv_buffer(kv_pool_host, num_blocks)

    # Write KV only
    set_result = storage_backend.batch_set_v1(
        block_hashes, torch.tensor(kv_host_indices)
    )
    assert all(set_result)

    # Check via batch_exists_v2
    mamba_transfer = PoolTransfer(
        name=PoolName.MAMBA,
        host_indices=torch.tensor(mamba_host_indices),
        keys=block_hashes,
        hit_policy=PoolHitPolicy.TRAILING_PAGES,
    )
    result = storage_backend.batch_exists_v2(block_hashes, [mamba_transfer])

    # KV exists but Mamba spec missing -> TRAILING_PAGES boundary=0 -> final_pages=0
    assert result.extra_pool_hit_pages.get(PoolName.KV, 0) == num_blocks, (
        f"Expected KV pool hit {num_blocks}, got {result.extra_pool_hit_pages}"
    )
    mamba_boundary = result.extra_pool_hit_pages.get(PoolName.MAMBA, 0)
    assert mamba_boundary == 0, (
        f"Expected Mamba boundary 0 (no Linear spec written), got {mamba_boundary}"
    )
    assert result.kv_hit_pages == 0, (
        f"Expected final_pages=0, got {result.kv_hit_pages}"
    )

    logger.info("test_write_kv_only_then_exists_v2 passed!")


def test_write_mamba_only_then_exists_v2(kv_pool_host, mamba_pool):
    """Write Mamba only (no KV) -> batch_exists_v2 filters locations by
    KV ("Full") spec presence.  Since only Linear spec was written,
    kv_hit_pages should be 0 and final_pages should be 0.
    """
    (storage_backend,
     block_hashes, kv_host_indices, mamba_host_indices) = _create_hybrid_backend(kv_pool_host, mamba_pool,
        "linear_mamba_only_0"
    )
    num_blocks = len(block_hashes)
    _fill_mamba_buffer(mamba_pool)

    # Write Mamba only
    mamba_transfer = PoolTransfer(
        name=PoolName.MAMBA,
        host_indices=torch.tensor(mamba_host_indices),
        keys=block_hashes,
        hit_policy=PoolHitPolicy.TRAILING_PAGES,
    )
    set_v2_result = storage_backend.batch_set_v2([mamba_transfer])
    assert PoolName.MAMBA in set_v2_result

    # Check via batch_exists_v2
    result = storage_backend.batch_exists_v2(block_hashes, [mamba_transfer])

    # KV not written -> locations lack "Full" spec -> kv_hit_pages = 0
    assert result.kv_hit_pages == 0, (
        f"Expected kv_hit_pages=0 (KV not written), got {result.kv_hit_pages}"
    )

    logger.info("test_write_mamba_only_then_exists_v2 passed!")


def test_kv_then_mamba_cross_order(kv_pool_host, mamba_pool):
    """Write KV first, then Mamba (reverse order of test_hybrid). Both should succeed."""
    (storage_backend,
     block_hashes, kv_host_indices, mamba_host_indices) = _create_hybrid_backend(kv_pool_host, mamba_pool,
        "linear_cross_order_0"
    )
    num_blocks = len(block_hashes)
    _fill_kv_buffer(kv_pool_host, num_blocks)
    _fill_mamba_buffer(mamba_pool)

    # Write KV first
    set_v1_result = storage_backend.batch_set_v1(
        block_hashes, torch.tensor(kv_host_indices)
    )
    assert all(set_v1_result), f"batch_set_v1 failed: {set_v1_result}"

    # Write Mamba second
    mamba_transfer = PoolTransfer(
        name=PoolName.MAMBA,
        host_indices=torch.tensor(mamba_host_indices),
        keys=block_hashes,
        hit_policy=PoolHitPolicy.TRAILING_PAGES,
    )
    set_v2_result = storage_backend.batch_set_v2([mamba_transfer])
    assert PoolName.MAMBA in set_v2_result
    assert all(set_v2_result[PoolName.MAMBA]), (
        f"batch_set_v2 failed: {set_v2_result[PoolName.MAMBA]}"
    )

    # Verify both exist
    result = storage_backend.batch_exists_v2(block_hashes, [mamba_transfer])
    assert result.kv_hit_pages == num_blocks
    assert result.extra_pool_hit_pages.get(PoolName.MAMBA, 0) == num_blocks

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
        assert torch.mean(tensor_i).item() == bf16_i

    # Verify Mamba data
    orig_temporal = mamba_pool.temporal_buffer[:num_blocks].clone()
    orig_conv = [conv[:num_blocks].clone() for conv in mamba_pool.conv_buffer]

    mamba_pool.temporal_buffer[:num_blocks].zero_()
    for conv_buf in mamba_pool.conv_buffer:
        conv_buf[:num_blocks].zero_()

    get_v2_result = storage_backend.batch_get_v2([mamba_transfer])
    assert all(get_v2_result[PoolName.MAMBA])

    for i in range(num_blocks):
        assert torch.allclose(mamba_pool.temporal_buffer[i], orig_temporal[i])
        for j, conv_buf in enumerate(mamba_pool.conv_buffer):
            assert torch.allclose(conv_buf[i], orig_conv[j][i])

    logger.info("test_kv_then_mamba_cross_order passed!")


def test_full_hybrid_round_trip(kv_pool_host, mamba_pool):
    """Complete write-clear-read-verify cycle for both KV and Mamba."""
    (storage_backend,
     block_hashes, kv_host_indices, mamba_host_indices) = _create_hybrid_backend(kv_pool_host, mamba_pool,
        "linear_full_rt_0"
    )
    num_blocks = len(block_hashes)
    _fill_kv_buffer(kv_pool_host, num_blocks)
    _fill_mamba_buffer(mamba_pool)

    mamba_transfer = PoolTransfer(
        name=PoolName.MAMBA,
        host_indices=torch.tensor(mamba_host_indices),
        keys=block_hashes,
        hit_policy=PoolHitPolicy.TRAILING_PAGES,
    )

    # Write both
    storage_backend.batch_set_v2([mamba_transfer])
    storage_backend.batch_set_v1(block_hashes, torch.tensor(kv_host_indices))

    # Exists check
    result = storage_backend.batch_exists_v2(block_hashes, [mamba_transfer])
    assert result.kv_hit_pages == num_blocks

    # Save originals
    orig_temporal = mamba_pool.temporal_buffer[:num_blocks].clone()
    orig_conv = [conv[:num_blocks].clone() for conv in mamba_pool.conv_buffer]

    # Clear both buffers
    index_shift = 3072
    shifted_kv = [v + index_shift for v in kv_host_indices]
    # Clear mamba
    mamba_pool.temporal_buffer[:num_blocks].zero_()
    for conv_buf in mamba_pool.conv_buffer:
        conv_buf[:num_blocks].zero_()

    # Read back
    get_v1 = storage_backend.batch_get_v1(block_hashes, torch.tensor(shifted_kv))
    assert all(get_v1)
    get_v2 = storage_backend.batch_get_v2([mamba_transfer])
    assert all(get_v2[PoolName.MAMBA])

    # Verify KV
    for i in range(num_blocks * page_size):
        page_id = (i + index_shift) // page_size
        token_id = (i + index_shift) % page_size
        bf16_i = torch.tensor(i, dtype=torch.bfloat16)
        tensor_i = kv_pool_host.kv_buffer[:, page_id, :, token_id]
        assert torch.mean(tensor_i).item() == bf16_i

    # Verify Mamba
    for i in range(num_blocks):
        assert torch.allclose(mamba_pool.temporal_buffer[i], orig_temporal[i])
        for j, conv_buf in enumerate(mamba_pool.conv_buffer):
            assert torch.allclose(conv_buf[i], orig_conv[j][i])

    logger.info("test_full_hybrid_round_trip passed!")


def test_trailing_pages_policy(kv_pool_host, mamba_pool):
    """Test TRAILING_PAGES boundary: only write Mamba for last N blocks."""
    (storage_backend,
     block_hashes, kv_host_indices, mamba_host_indices) = _create_hybrid_backend(kv_pool_host, mamba_pool,
        "linear_trailing_0"
    )
    num_blocks = len(block_hashes)
    _fill_kv_buffer(kv_pool_host, num_blocks)
    _fill_mamba_buffer(mamba_pool)

    # Write all KV blocks
    set_v1 = storage_backend.batch_set_v1(
        block_hashes, torch.tensor(kv_host_indices)
    )
    assert all(set_v1)

    # Write Mamba only for last 3 blocks
    trailing_n = 3
    trailing_keys = block_hashes[-trailing_n:]
    trailing_mamba_indices = mamba_host_indices[-trailing_n:]
    trailing_transfer = PoolTransfer(
        name=PoolName.MAMBA,
        host_indices=torch.tensor(trailing_mamba_indices),
        keys=trailing_keys,
        hit_policy=PoolHitPolicy.TRAILING_PAGES,
    )
    set_v2 = storage_backend.batch_set_v2([trailing_transfer])
    assert PoolName.MAMBA in set_v2
    assert all(set_v2[PoolName.MAMBA])

    # Query with TRAILING_PAGES policy using trailing_n keys
    query_transfer = PoolTransfer(
        name=PoolName.MAMBA,
        host_indices=torch.tensor(trailing_mamba_indices),
        keys=trailing_keys,
        hit_policy=PoolHitPolicy.TRAILING_PAGES,
    )
    result = storage_backend.batch_exists_v2(block_hashes, [query_transfer])

    # KV hits all 10 blocks
    assert result.extra_pool_hit_pages.get(PoolName.KV, 0) == num_blocks

    # TRAILING_PAGES: scanning from kv_hit_pages=10 backwards,
    # last 3 pages [7,8,9] have Linear spec -> boundary should be 10
    mamba_boundary = result.extra_pool_hit_pages.get(PoolName.MAMBA, 0)
    assert mamba_boundary == num_blocks, (
        f"Expected trailing boundary {num_blocks}, got {mamba_boundary}"
    )
    assert result.kv_hit_pages == num_blocks

    # Now query with ALL_PAGES — should truncate at page 7 (first without spec)
    all_pages_transfer = PoolTransfer(
        name=PoolName.MAMBA,
        host_indices=torch.tensor(mamba_host_indices),
        keys=block_hashes,
        hit_policy=PoolHitPolicy.ALL_PAGES,
    )
    result_all = storage_backend.batch_exists_v2(block_hashes, [all_pages_transfer])
    mamba_boundary_all = result_all.extra_pool_hit_pages.get(PoolName.MAMBA, 0)
    # First 7 blocks (indices 0-6) have no Mamba spec -> boundary = 0
    assert mamba_boundary_all == 0, (
        f"Expected ALL_PAGES boundary 0 (first block lacks spec), got {mamba_boundary_all}"
    )

    logger.info("test_trailing_pages_policy passed!")


def test_partial_write_buffer_indexing(kv_pool_host, mamba_pool):
    """Verify correct buffer indexing when save_indices is a proper subset.

    Write Mamba for first N blocks, then write ALL 2N blocks. The second call
    triggers a partial write (only the new blocks). With the old buffer slicing
    bug, data from buffer positions 0..N-1 would be written to storage for
    blocks N..2N-1 instead of the correct buffer positions N..2N-1.
    """
    num_blocks = 6
    first_n = 3

    (storage_backend,
     block_hashes, kv_host_indices, mamba_host_indices) = _create_hybrid_backend(
        kv_pool_host, mamba_pool, "linear_partial_idx_0", num_blocks=num_blocks
    )

    _fill_mamba_buffer(mamba_pool)

    # Step 1: Write Mamba for first 3 blocks — all new, save_indices = [0,1,2]
    first_transfer = PoolTransfer(
        name=PoolName.MAMBA,
        host_indices=torch.tensor(mamba_host_indices[:first_n]),
        keys=block_hashes[:first_n],
        hit_policy=PoolHitPolicy.TRAILING_PAGES,
    )
    set_result_1 = storage_backend.batch_set_v2([first_transfer])
    assert PoolName.MAMBA in set_result_1
    assert all(set_result_1[PoolName.MAMBA]), (
        f"First batch_set_v2 failed: {set_result_1[PoolName.MAMBA]}"
    )

    # Step 2: Write Mamba for ALL 6 blocks.
    # Manager recognises blocks 0-2 as cached -> save_indices = [3,4,5].
    # Old bug: buffer[0:3] saved to storage for blocks 3-5 (wrong positions).
    full_transfer = PoolTransfer(
        name=PoolName.MAMBA,
        host_indices=torch.tensor(mamba_host_indices),
        keys=block_hashes,
        hit_policy=PoolHitPolicy.TRAILING_PAGES,
    )
    set_result_2 = storage_backend.batch_set_v2([full_transfer])
    assert PoolName.MAMBA in set_result_2

    # Step 3: Clear buffers and read back ALL blocks
    orig_temporal = mamba_pool.temporal_buffer[:num_blocks].clone()
    orig_conv = [conv[:num_blocks].clone() for conv in mamba_pool.conv_buffer]

    mamba_pool.temporal_buffer[:num_blocks].zero_()
    for conv_buf in mamba_pool.conv_buffer:
        conv_buf[:num_blocks].zero_()

    get_result = storage_backend.batch_get_v2([full_transfer])
    assert PoolName.MAMBA in get_result
    assert all(get_result[PoolName.MAMBA]), (
        f"batch_get_v2 failed: {get_result[PoolName.MAMBA]}"
    )

    # Step 4: Verify each page has its own correct data (not swapped).
    # With the old bug, pages 3-5 would contain data from pages 0-2.
    for i in range(num_blocks):
        assert torch.allclose(
            mamba_pool.temporal_buffer[i], orig_temporal[i]
        ), (
            f"Temporal mismatch at page {i}: "
            f"got {mamba_pool.temporal_buffer[i].flatten()[0].item()}, "
            f"expected {orig_temporal[i].flatten()[0].item()}"
        )
        for j, conv_buf in enumerate(mamba_pool.conv_buffer):
            assert torch.allclose(
                conv_buf[i], orig_conv[j][i]
            ), f"Conv {j} mismatch at page {i}"

    logger.info("test_partial_write_buffer_indexing passed!")


def test_non_mamba_pool_transfer_returns_false(kv_pool_host, mamba_pool):
    """Non-MAMBA pool transfers in batch_set_v2 / batch_get_v2 return all False."""
    (storage_backend,
     block_hashes, kv_host_indices, mamba_host_indices) = _create_hybrid_backend(kv_pool_host, mamba_pool,
        "linear_non_mamba_0"
    )

    # Use a non-MAMBA pool name
    indexer_transfer = PoolTransfer(
        name=PoolName.INDEXER,
        host_indices=torch.tensor(mamba_host_indices),
        keys=block_hashes,
        hit_policy=PoolHitPolicy.ALL_PAGES,
    )

    # batch_set_v2 with non-MAMBA -> all False
    set_result = storage_backend.batch_set_v2([indexer_transfer])
    assert PoolName.INDEXER in set_result
    assert all(v is False for v in set_result[PoolName.INDEXER]), (
        f"Expected all False for INDEXER set, got {set_result[PoolName.INDEXER]}"
    )

    # batch_get_v2 with non-MAMBA -> all False
    get_result = storage_backend.batch_get_v2([indexer_transfer])
    assert PoolName.INDEXER in get_result
    assert all(v is False for v in get_result[PoolName.INDEXER]), (
        f"Expected all False for INDEXER get, got {get_result[PoolName.INDEXER]}"
    )

    logger.info("test_non_mamba_pool_transfer_returns_false passed!")


if __name__ == "__main__":
    logger.info("Starting hybrid (KV + Mamba/Linear) tests")

    try:
        _start_manager()
        _device_pool, _kv_pool_host, _mamba_pool = _create_pools()

        test_hybrid(_kv_pool_host, _mamba_pool)
        test_write_kv_only_then_exists_v2(_kv_pool_host, _mamba_pool)
        test_write_mamba_only_then_exists_v2(_kv_pool_host, _mamba_pool)
        test_kv_then_mamba_cross_order(_kv_pool_host, _mamba_pool)
        test_full_hybrid_round_trip(_kv_pool_host, _mamba_pool)
        test_trailing_pages_policy(_kv_pool_host, _mamba_pool)
        test_partial_write_buffer_indexing(_kv_pool_host, _mamba_pool)
        test_non_mamba_pool_transfer_returns_false(_kv_pool_host, _mamba_pool)
        logger.info("All hybrid tests passed successfully!")
    except Exception as e:
        logger.error(f"Test failed with error: {e}")
        raise
    finally:
        _stop_manager()
