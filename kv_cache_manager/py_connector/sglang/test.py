import logging
logging.basicConfig(level=logging.DEBUG)

import subprocess
import signal
import time
import os
import atexit
import torch
from sglang.srt.mem_cache.hicache_storage import (
    HiCacheStorageConfig,
    HiCacheStorageExtraInfo,
    get_hash_str,
)
from sglang.srt.mem_cache.memory_pool import MHATokenToKVPool
from sglang.srt.mem_cache.memory_pool_host import MHATokenToKVPoolHost
from sglang.srt.distributed import (
    init_distributed_environment,
    initialize_model_parallel,
)
from kv_cache_manager.py_connector.sglang.connector import HiCacheKVCM

logger = logging.getLogger(__name__)

init_distributed_environment(
    world_size=1,
    rank=0,
    distributed_init_method="tcp://127.0.0.1:23456",
    local_rank=0,
    backend="gloo",
)

initialize_model_parallel(
    tensor_model_parallel_size=1,
    pipeline_model_parallel_size=1,
)

# Configuration
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

manager_uri = "http://127.0.0.1:6382"

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
    # Clean up previous logs
    subprocess.run("rm -rf /root/KVCacheManager/logs/*", shell=True, check=False)

    # Start the manager process
    proc = subprocess.Popen([
        "/home/admin/kv_cache_manager/bin/kv_cache_manager_bin",
        "-c", "/home/admin/kv_cache_manager/etc/default_server_config.conf",
        "-l", "/home/admin/kv_cache_manager/etc/default_logger_config.conf"
    ])

    # Register cleanup handlers
    signal.signal(signal.SIGINT, _stop_manager)
    signal.signal(signal.SIGTERM, _stop_manager)
    atexit.register(_stop_manager)
    logger.info(f"kv_cache_manager {proc=}")
    return proc


def test():
    """Run the main test sequence."""
    # Setup memory pools
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
    mem_pool_host = MHATokenToKVPoolHost(
        device_pool=device_pool,
        host_to_device_ratio=hicache_ratio,
        host_size=hicache_size,
        page_size=page_size,
        layout=hicache_mem_layout,
    )

    # Configure storage backend
    storage_config = HiCacheStorageConfig(
        tp_rank=0,
        tp_size=1,
        is_mla_model=False,
        is_page_first_layout=True,
        model_name=model_name,
        extra_config={
            "manager_uri": manager_uri,
            "instance_group": "default",
            "instance_id": "0",
        },
    )

    # Initialize storage backend
    storage_backend = HiCacheKVCM(storage_config)
    storage_backend.register_mem_pool_host(mem_pool_host)

    # Generate test data
    token_ids = list(range(1024))
    block_hashes = []
    block_hash = None
    host_indices = []

    for i in range(0, 1024, page_size):
        block_hash = get_hash_str(token_ids[i: i + page_size], block_hash)
        block_hashes.append(block_hash)
        host_indices.extend(range(i, i + page_size))

    # Fill KV buffer with test data
    for i in range(mem_pool_host.page_num * mem_pool_host.page_size):
        page_id = i // page_size
        token_id = i % page_size
        bf16_i = torch.tensor(i, dtype=torch.bfloat16)
        # (2, page_num, layer_num, page_size, head_num, head_dim)
        mem_pool_host.kv_buffer[:, page_id, :, token_id] = bf16_i

    # Test 1: Basic set/get operations
    block_hashes_0 = block_hashes[:10]
    host_indices_0 = host_indices[:(10 * page_size)]

    # Verify blocks don't exist initially
    assert storage_backend.batch_exists(block_hashes_0) == 0

    # Set blocks and verify they exist
    set_result = storage_backend.batch_set_v1(block_hashes_0, torch.tensor(host_indices_0))
    assert all(set_result)
    assert storage_backend.batch_exists(block_hashes_0) == len(block_hashes_0)

    # Get blocks and verify data integrity
    get_result = storage_backend.batch_get_v1(block_hashes_0, torch.tensor(host_indices_0))
    assert all(get_result)

    # Verify data in KV buffer
    for i in range(mem_pool_host.page_num * mem_pool_host.page_size):
        page_id = i // page_size
        token_id = i % page_size
        bf16_i = torch.tensor(i, dtype=torch.bfloat16)
        tensor_i = mem_pool_host.kv_buffer[:, page_id, :, token_id]
        assert torch.mean(tensor_i).item() == bf16_i
        assert torch.std(tensor_i).item() == 0

    # Test 2: Prefix-based operations
    prefix_keys_1 = block_hashes_0
    extra_info_1 = HiCacheStorageExtraInfo(prefix_keys=prefix_keys_1)
    block_hashes_1 = block_hashes[10:20]
    host_indices_1 = host_indices[(10 * page_size):(20 * page_size)]

    # Verify blocks don't exist initially
    assert storage_backend.batch_exists(block_hashes_1, extra_info_1) == 0

    # Set blocks with prefix and verify they exist
    set_result = storage_backend.batch_set_v1(block_hashes_1, torch.tensor(host_indices_1), extra_info_1)
    assert all(set_result)
    assert storage_backend.batch_exists(block_hashes_1, extra_info_1) == len(block_hashes_1)

    # Test 3: Get with different prefix
    prefix_keys_2 = block_hashes[:5]
    extra_info_2 = HiCacheStorageExtraInfo(prefix_keys=prefix_keys_2)
    block_hashes_2 = block_hashes[5:15]
    host_indices_2 = host_indices[(5 * page_size):(15 * page_size)]
    index_shift = 1024
    host_indices_2 = [v + index_shift for v in host_indices_2]

    # Get blocks with different prefix
    get_result = storage_backend.batch_get_v1(block_hashes_2, torch.tensor(host_indices_2), extra_info_2)
    assert all(get_result)

    # Verify data in KV buffer
    for i in host_indices_2:
        page_id = i // page_size
        token_id = i % page_size
        bf16_i = torch.tensor(i - index_shift, dtype=torch.bfloat16)
        tensor_i = mem_pool_host.kv_buffer[:, page_id, :, token_id]
        assert torch.mean(tensor_i).item() == bf16_i, f"{torch.mean(tensor_i).item()=} == {bf16_i=}"
        assert torch.std(tensor_i).item() == 0


if __name__ == "__main__":
    logger.info("Starting KV Cache Manager test")

    try:
        _start_manager()
        test()
        logger.info("All tests passed successfully!")
    except Exception as e:
        logger.error(f"Test failed with error: {e}")
        raise
    finally:
        _stop_manager()
