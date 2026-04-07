import logging
logging.basicConfig(level=logging.DEBUG)

import subprocess
import signal
import time
import os
import atexit
import requests
import torch
import torch.multiprocessing as mp
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

manager_uri = os.environ.get("KVCM_MANAGER_URI", "http://127.0.0.1:6382")
manager_http_port = int(manager_uri.rsplit(":", 1)[-1])
debug_uri = os.environ.get("KVCM_DEBUG_URI", f"http://127.0.0.1:{manager_http_port + 3000}")

manager_bin = os.environ.get(
    "KVCM_MANAGER_BIN",
    "/home/admin/kv_cache_manager/bin/kv_cache_manager_bin",
)
manager_server_conf = os.environ.get(
    "KVCM_SERVER_CONF",
    "/home/admin/kv_cache_manager/etc/default_server_config.conf",
)
manager_logger_conf = os.environ.get(
    "KVCM_LOGGER_CONF",
    "/home/admin/kv_cache_manager/etc/default_logger_config.conf",
)
manager_log_dir = os.environ.get("KVCM_LOG_DIR", "/root/KVCacheManager/logs")

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
    subprocess.run(f"rm -rf {manager_log_dir}/*", shell=True, check=False)

    # Start the manager process (with debug service enabled for fault injection)
    proc = subprocess.Popen([
        manager_bin,
        "-c", manager_server_conf,
        "-l", manager_logger_conf,
        "--env", "kvcm.service.enable_debug_service=true",
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
        pp_rank=0,
        pp_size=1,
        is_mla_model=False,
        enable_storage_metrics=False,
        is_page_first_layout=True,
        model_name=model_name,
        extra_config={
            "manager_uri": manager_uri,
            "instance_group": "default",
            "instance_id": "0",
        },
    )

    # Initialize storage backend
    storage_backend = HiCacheKVCM(storage_config, {})
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

    return storage_backend


class DebugServiceClient:
    """Client for the KV Cache Manager debug service (fault injection)."""

    def __init__(self, base_url):
        self.base_url = base_url
        self.session = requests.Session()
        self.headers = {'Accept': 'application/json', 'Content-Type': 'application/json'}

    def _make_request(self, endpoint, data=None):
        url = self.base_url + endpoint
        response = self.session.post(url, json=data, headers=self.headers)
        response_data = response.json()
        assert response.status_code == 200, \
            f"Debug API {endpoint} failed: {response.status_code}"
        assert response_data.get('header', {}).get('status', {}).get('code') == 'OK', \
            f"Debug API {endpoint} error: {response_data}"
        return response_data

    def inject_fault(self, api_name, fault_type="INTERNAL_ERROR",
                     strategy="ALWAYS", trigger_at_call=None):
        data = {
            "api_name": api_name,
            "fault_type": fault_type,
            "fault_trigger_strategy": strategy,
        }
        if trigger_at_call is not None:
            data["trigger_at_call"] = trigger_at_call
        return self._make_request('/api/injectFault', data)

    def remove_fault(self, api_name):
        return self._make_request('/api/removeFault', {"api_name": api_name})

    def clear_faults(self):
        return self._make_request('/api/clearFaults', {})

    def close(self):
        self.session.close()


def test_fault_injection(storage_backend):
    """Test fault injection scenarios to verify error handling.

    These tests verify that when manager APIs fail, the connector
    returns graceful errors instead of crashing or (in multi-rank
    scenarios) causing NCCL/gloo hangs due to inconsistent collective
    operations across ranks.
    """
    debug_client = DebugServiceClient(debug_uri)

    # Generate separate test data (uncached blocks) for fault injection tests.
    fi_token_ids = list(range(5000, 6024))
    fi_block_hashes = []
    fi_block_hash = None
    fi_host_indices = []
    for i in range(0, 1024, page_size):
        fi_block_hash = get_hash_str(fi_token_ids[i:i + page_size], fi_block_hash)
        fi_block_hashes.append(fi_block_hash)
        fi_host_indices.extend(range(i + 4096, i + 4096 + page_size))

    try:
        # ----------------------------------------------------------
        # Test FI-1: StartWriteCache ALWAYS fault
        #   In multi-rank mode, without the fix, rank 0 would throw
        #   before broadcast, leaving other ranks hanging forever.
        # ----------------------------------------------------------
        logger.info("=== FI-1: StartWriteCache ALWAYS fault ===")
        debug_client.inject_fault("StartWriteCache")

        hashes = fi_block_hashes[:5]
        indices = fi_host_indices[:(5 * page_size)]
        set_result = storage_backend.batch_set_v1(
            hashes, torch.tensor(indices))
        assert all(r is False for r in set_result), \
            f"FI-1 FAILED: expected all False, got {set_result}"
        logger.info("FI-1 PASSED: StartWriteCache fault → batch_set_v1 returns all False")

        debug_client.remove_fault("StartWriteCache")

        # ----------------------------------------------------------
        # Test FI-2: GetCacheLocation ALWAYS fault (batch_get_v1)
        # ----------------------------------------------------------
        logger.info("=== FI-2: GetCacheLocation ALWAYS fault (batch_get) ===")
        debug_client.inject_fault("GetCacheLocation")

        get_result = storage_backend.batch_get_v1(
            hashes, torch.tensor(indices))
        assert all(r is False for r in get_result), \
            f"FI-2 FAILED: expected all False, got {get_result}"
        logger.info("FI-2 PASSED: GetCacheLocation fault → batch_get_v1 returns all False")

        # ----------------------------------------------------------
        # Test FI-3: GetCacheLocation ALWAYS fault (batch_exists)
        # ----------------------------------------------------------
        logger.info("=== FI-3: GetCacheLocation ALWAYS fault (batch_exists) ===")
        exists_result = storage_backend.batch_exists(hashes)
        assert exists_result == 0, \
            f"FI-3 FAILED: expected 0, got {exists_result}"
        logger.info("FI-3 PASSED: GetCacheLocation fault → batch_exists returns 0")

        debug_client.remove_fault("GetCacheLocation")

        # ----------------------------------------------------------
        # Test FI-4: FinishWriteCache ALWAYS fault
        #   In multi-rank mode, without the fix, rank 0 would throw
        #   after all_reduce, causing inconsistent return values.
        #   Uses separate block hashes because a failed finish leaves
        #   a pending write session in the manager, which would
        #   interfere with subsequent tests on the same keys.
        # ----------------------------------------------------------
        logger.info("=== FI-4: FinishWriteCache ALWAYS fault ===")
        debug_client.inject_fault("FinishWriteCache")

        fi4_hashes = fi_block_hashes[10:15]
        fi4_indices = fi_host_indices[(10 * page_size):(15 * page_size)]
        set_result = storage_backend.batch_set_v1(
            fi4_hashes, torch.tensor(fi4_indices))
        assert all(r is False for r in set_result), \
            f"FI-4 FAILED: expected all False, got {set_result}"
        logger.info("FI-4 PASSED: FinishWriteCache fault → batch_set_v1 returns all False")

        debug_client.remove_fault("FinishWriteCache")

        # ----------------------------------------------------------
        # Test FI-5: Recovery after clearing all faults
        #   Verify full get-miss → set → get-hit cycle works after
        #   faults are removed.
        # ----------------------------------------------------------
        logger.info("=== FI-5: Recovery after clearing faults ===")
        debug_client.clear_faults()

        # The blocks were never successfully written during fault tests,
        # so batch_exists and batch_get should report miss.
        exists_result = storage_backend.batch_exists(hashes)
        assert exists_result == 0, \
            f"FI-5 FAILED: expected 0 (not cached), got {exists_result}"

        get_result = storage_backend.batch_get_v1(
            hashes, torch.tensor(indices))
        assert all(r is False for r in get_result), \
            f"FI-5 FAILED: expected all False (cache miss), got {get_result}"

        # Now set should succeed.
        set_result = storage_backend.batch_set_v1(
            hashes, torch.tensor(indices))
        assert all(set_result), \
            f"FI-5 FAILED: expected all True after clearing faults, got {set_result}"

        # After successful set, get should hit.
        get_result = storage_backend.batch_get_v1(
            hashes, torch.tensor(indices))
        assert all(get_result), \
            f"FI-5 FAILED: expected all True (cache hit), got {get_result}"

        exists_result = storage_backend.batch_exists(hashes)
        assert exists_result == len(hashes), \
            f"FI-5 FAILED: expected {len(hashes)}, got {exists_result}"
        logger.info("FI-5 PASSED: get miss → set → get hit after clearing faults")

        # ----------------------------------------------------------
        # Test FI-6: StartWriteCache fault with prefix (batch_set_v1)
        #   Verify fault handling works correctly with extra_info.
        # ----------------------------------------------------------
        logger.info("=== FI-6: StartWriteCache fault with prefix ===")
        debug_client.inject_fault("StartWriteCache")

        prefix_keys = fi_block_hashes[:5]
        extra_info = HiCacheStorageExtraInfo(prefix_keys=prefix_keys)
        suffix_hashes = fi_block_hashes[5:10]
        suffix_indices = fi_host_indices[(5 * page_size):(10 * page_size)]
        set_result = storage_backend.batch_set_v1(
            suffix_hashes, torch.tensor(suffix_indices), extra_info)
        assert all(r is False for r in set_result), \
            f"FI-6 FAILED: expected all False, got {set_result}"
        logger.info("FI-6 PASSED: StartWriteCache fault with prefix → returns all False")

        debug_client.remove_fault("StartWriteCache")

        # ----------------------------------------------------------
        # Test FI-7: GetCacheLocation fault with prefix (batch_get_v1)
        # ----------------------------------------------------------
        logger.info("=== FI-7: GetCacheLocation fault with prefix (batch_get) ===")
        debug_client.inject_fault("GetCacheLocation")

        get_result = storage_backend.batch_get_v1(
            suffix_hashes, torch.tensor(suffix_indices), extra_info)
        assert all(r is False for r in get_result), \
            f"FI-7 FAILED: expected all False, got {get_result}"
        logger.info("FI-7 PASSED: GetCacheLocation fault with prefix → returns all False")

        debug_client.remove_fault("GetCacheLocation")

    finally:
        debug_client.clear_faults()
        debug_client.close()



# ---------------------------------------------------------------------------
# Multi-rank test (mp.spawn)
# ---------------------------------------------------------------------------
# Smaller parameters to reduce GPU memory usage (2 workers share one GPU).
mr_layer_num = 4
mr_max_total_num_tokens = 2048
mr_init_port = 23457


def _multi_rank_worker(rank, world_size, init_port):
    """Worker function for multi-rank fault injection tests.

    Each spawned process runs this function with its own rank.
    Verifies that collective operations (broadcast, all_reduce)
    remain consistent even when manager API calls fail.
    """
    # We only need gloo for the connector's collective operations.
    # Bypass sglang's initialize_model_parallel entirely because it
    # creates NCCL communicators which fail when multiple processes
    # share a single GPU.
    torch.distributed.init_process_group(
        backend="gloo",
        init_method=f"tcp://127.0.0.1:{init_port}",
        world_size=world_size,
        rank=rank,
    )

    # The connector reads get_tp_group().cpu_group to obtain the TP
    # gloo group.  Patch the global _TP so that get_tp_group() returns
    # a lightweight object with only the .cpu_group we need.
    import sglang.srt.distributed.parallel_state as _ps

    class _GlooTPGroup:
        """Minimal stand-in for GroupCoordinator (gloo only)."""
        def __init__(self, cpu_group):
            self.cpu_group = cpu_group

    _ps._TP = _GlooTPGroup(
        torch.distributed.new_group(list(range(world_size)), backend="gloo")
    )

    # Create memory pools (smaller than single-rank tests)
    device_pool = MHATokenToKVPool(
        size=mr_max_total_num_tokens,
        page_size=page_size,
        dtype=kv_cache_dtype,
        head_num=head_num,
        head_dim=head_dim,
        layer_num=mr_layer_num,
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

    storage_config = HiCacheStorageConfig(
        tp_rank=rank,
        tp_size=world_size,
        pp_rank=0,
        pp_size=1,
        is_mla_model=False,
        enable_storage_metrics=False,
        is_page_first_layout=True,
        model_name=model_name,
        extra_config={
            "manager_uri": manager_uri,
            "instance_group": "default",
            "instance_id": "mr_test",
        },
    )
    storage_backend = HiCacheKVCM(storage_config, {})
    storage_backend.register_mem_pool_host(mem_pool_host)

    # Fill KV buffer with distinguishable data per rank
    for i in range(mem_pool_host.page_num * mem_pool_host.page_size):
        p = i // page_size
        t = i % page_size
        mem_pool_host.kv_buffer[:, p, :, t] = torch.tensor(
            i + rank * 100000, dtype=torch.bfloat16
        )

    # Generate block hashes (different token_ids from single-rank tests)
    mr_token_ids = list(range(20000, 20000 + mr_max_total_num_tokens))
    mr_hashes = []
    mr_hash = None
    mr_indices = []
    for i in range(0, mr_max_total_num_tokens, page_size):
        mr_hash = get_hash_str(mr_token_ids[i:i + page_size], mr_hash)
        mr_hashes.append(mr_hash)
        mr_indices.extend(range(i, i + page_size))

    debug_client = DebugServiceClient(debug_uri) if rank == 0 else None

    # ------------------------------------------------------------------
    # MR-1: Normal multi-rank set
    # ------------------------------------------------------------------
    h1 = mr_hashes[:5]
    idx1 = mr_indices[:5 * page_size]
    torch.distributed.barrier()
    result = storage_backend.batch_set_v1(h1, torch.tensor(idx1))
    assert all(result), f"MR-1 rank {rank}: expected all True, got {result}"
    torch.distributed.barrier()
    logger.info(f"[Rank {rank}] MR-1 PASSED: normal set")

    # ------------------------------------------------------------------
    # MR-2: Normal multi-rank get
    # ------------------------------------------------------------------
    torch.distributed.barrier()
    result = storage_backend.batch_get_v1(h1, torch.tensor(idx1))
    assert all(result), f"MR-2 rank {rank}: expected all True, got {result}"
    torch.distributed.barrier()
    logger.info(f"[Rank {rank}] MR-2 PASSED: normal get")

    # ------------------------------------------------------------------
    # MR-3: StartWriteCache fault (main bug scenario)
    #   Without the fix, rank 0 throws before broadcast, rank 1 hangs
    #   at broadcast_object_list -> gloo timeout.
    #   With the fix, both ranks return [False] gracefully.
    # ------------------------------------------------------------------
    h3 = mr_hashes[5:10]
    idx3 = mr_indices[5 * page_size:10 * page_size]

    torch.distributed.barrier()
    if rank == 0:
        debug_client.inject_fault("StartWriteCache")
    torch.distributed.barrier()

    result = storage_backend.batch_set_v1(h3, torch.tensor(idx3))
    assert all(r is False for r in result), \
        f"MR-3 rank {rank}: expected all False, got {result}"

    torch.distributed.barrier()
    if rank == 0:
        debug_client.remove_fault("StartWriteCache")
    torch.distributed.barrier()
    logger.info(f"[Rank {rank}] MR-3 PASSED: StartWriteCache fault")

    # ------------------------------------------------------------------
    # MR-4: FinishWriteCache fault
    #   After the fix, rank 0 gets [False] (finish failed),
    #   other ranks may get [True] (data transfer succeeded).
    #   This is a known inconsistency. We verify no hang occurs.
    # ------------------------------------------------------------------
    h4 = mr_hashes[15:20]
    idx4 = mr_indices[15 * page_size:20 * page_size]

    torch.distributed.barrier()
    if rank == 0:
        debug_client.inject_fault("FinishWriteCache")
    torch.distributed.barrier()

    result = storage_backend.batch_set_v1(h4, torch.tensor(idx4))
    if rank == 0:
        assert all(r is False for r in result), \
            f"MR-4 rank 0: expected all False, got {result}"
    logger.info(f"[Rank {rank}] MR-4: result = {result}")

    torch.distributed.barrier()
    if rank == 0:
        debug_client.remove_fault("FinishWriteCache")
    torch.distributed.barrier()
    logger.info(f"[Rank {rank}] MR-4 PASSED: FinishWriteCache fault (no hang)")

    # ------------------------------------------------------------------
    # MR-5: Recovery after faults
    #   Clear faults, verify set -> get works on all ranks.
    # ------------------------------------------------------------------
    h5 = mr_hashes[20:25]
    idx5 = mr_indices[20 * page_size:25 * page_size]

    torch.distributed.barrier()
    if rank == 0:
        debug_client.clear_faults()
    torch.distributed.barrier()

    result = storage_backend.batch_set_v1(h5, torch.tensor(idx5))
    assert all(result), f"MR-5 rank {rank}: set expected all True, got {result}"
    torch.distributed.barrier()

    result = storage_backend.batch_get_v1(h5, torch.tensor(idx5))
    assert all(result), f"MR-5 rank {rank}: get expected all True, got {result}"
    torch.distributed.barrier()
    logger.info(f"[Rank {rank}] MR-5 PASSED: recovery")

    if debug_client:
        debug_client.close()
    logger.info(f"[Rank {rank}] All multi-rank tests passed!")


def test_multi_rank(timeout_seconds=120):
    """Run multi-rank tests with timeout detection for gloo hangs.

    Uses multiprocessing.Process with explicit join(timeout) so that
    hung child processes can be forcefully killed.
    """
    world_size = 2
    ctx = mp.get_context("spawn")
    processes = []
    for rank in range(world_size):
        p = ctx.Process(
            target=_multi_rank_worker,
            args=(rank, world_size, mr_init_port),
        )
        p.start()
        processes.append(p)

    # Wait for all workers to finish within the timeout.
    deadline = time.monotonic() + timeout_seconds
    for p in processes:
        remaining = max(0, deadline - time.monotonic())
        p.join(timeout=remaining)

    # Check results — kill any survivors and report.
    hung = [p for p in processes if p.is_alive()]
    if hung:
        for p in hung:
            p.kill()
        for p in hung:
            p.join()
        raise TimeoutError(
            "Multi-rank test timed out -- likely a gloo hang due to "
            "inconsistent collective operations across ranks"
        )

    failed = [p for p in processes if p.exitcode != 0]
    if failed:
        codes = {p.pid: p.exitcode for p in failed}
        raise RuntimeError(f"Multi-rank worker(s) failed: {codes}")


if __name__ == "__main__":
    logger.info("Starting KV Cache Manager test")

    # Initialize distributed for single-rank tests
    init_distributed_environment(
        world_size=1,
        rank=0,
        distributed_init_method="tcp://127.0.0.1:23460",
        local_rank=0,
        backend="gloo",
    )
    initialize_model_parallel(
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=1,
    )

    try:
        _start_manager()

        storage_backend = test()
        logger.info("All basic tests passed!")

        test_fault_injection(storage_backend)
        logger.info("All fault injection tests passed!")

        # Destroy single-rank process group before spawning multi-rank workers
        torch.distributed.destroy_process_group()

        test_multi_rank()
        logger.info("All multi-rank tests passed!")

        logger.info("All tests passed successfully!")
    except Exception as e:
        logger.error(f"Test failed with error: {e}")
        raise
    finally:
        _stop_manager()
