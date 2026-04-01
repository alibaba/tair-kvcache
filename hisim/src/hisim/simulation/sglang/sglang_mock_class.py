from typing import List, Union, Optional, Any
import tempfile
import time
from pathlib import Path
import numpy as np
import torch
import psutil
import os
import threading
from functools import wraps

from hisim.utils.logger import get_logger
from hisim.simulation.manager import StateManager, ConfigManager, Envs
from hisim.simulation.sglang.version import VersionDispatcher

try:
    from kv_cache_manager.optimizer.pybind import kvcm_py_optimizer
except ImportError:
    kvcm_py_optimizer = None

# from sglang.srt.managers.schedule_batch import Req

logger = get_logger("hisim")
_CURRENT_DIR = Path(__file__).parent.resolve()


def synchronized(func):
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        with self.lock:
            return func(self, *args, **kwargs)

    return wrapper


class MockTokenToKVPoolHost:
    KV_CACHE_BYTES: int = None
    KV_CACHE_BYTES_PER_LAYER: int = None

    MEMORY_READ_BANDWIDTH_BYTES: float = None
    MEMORY_WRITE_BANDWIDTH_BYTES: float = None

    # from sglang.srt.mem_cache.memory_pool_host import MHATokenToKVPoolHost
    def __init__(
        self,
        device_pool,
        host_to_device_ratio: float,
        host_size: int,
        page_size: int,
        layout: str,
        pin_memory: bool,
        device: str,
    ):
        self.device_pool = device_pool
        self.page_size = page_size
        self.layout = layout
        self.pin_memory = False
        self.device = device

        self.dtype = device_pool.store_dtype
        self.size_per_token = self.get_size_per_token()
        if host_size > 0:
            self.size = int(host_size * 1e9 // self.size_per_token)
        else:
            self.size = int(device_pool.size * host_to_device_ratio)
        # Align up the host memory pool size to the page size
        self.page_num = self.size // self.page_size + 1
        self.size = self.page_num * self.page_size
        self.start_layer = device_pool.start_layer
        self.end_layer = device_pool.end_layer

        assert self.size > device_pool.size, (
            "The host memory should be larger than the device memory with the current protocol"
        )

        # Verify there is enough available host memory.
        host_mem = psutil.virtual_memory()
        requested_bytes = self.size * self.size_per_token
        # preserve at least 10GB for other usage
        ten_gb = 10 * (1024**3)
        available_bytes = host_mem.available - ten_gb
        if requested_bytes > available_bytes:
            raise ValueError(
                f"Not enough host memory available. Requesting "
                f"{requested_bytes / 1e9:.2f} GB but only have "
                f"{available_bytes / 1e9:.2f} GB free. Please reduce the "
                f"size of the hierarchical cache."
            )
        else:
            logger.info(
                f"Allocating {requested_bytes / 1e9:.2f} GB host memory for hierarchical KV cache."
            )

        self.kv_buffer = self.init_kv_buffer()

        # A lock for synchronized operations on memory allocation and state transitions.
        self.lock = threading.RLock()
        self.clear()

    def get_size_per_token(self):
        # MHA implementation
        self.head_num = self.device_pool.head_num
        self.head_dim = self.device_pool.head_dim
        self.layer_num = self.device_pool.layer_num

        return self.head_dim * self.head_num * self.layer_num * self.dtype.itemsize * 2

    def init_kv_buffer(self):
        if self.layout == "layer_first":
            dims = (2, self.layer_num, self.size, self.head_num, self.head_dim)
        elif self.layout == "page_first":
            dims = (2, self.size, self.layer_num, self.head_num, self.head_dim)
        elif self.layout == "page_first_direct":
            dims = (
                2,
                self.page_num,
                self.layer_num,
                self.page_size,
                self.head_num,
                self.head_dim,
            )
        elif self.layout == "page_head":
            dims = (
                2,
                self.page_num,
                self.head_num,
                self.page_size,
                self.layer_num,
                self.head_dim,
            )
        else:
            raise ValueError(f"Unsupported layout: {self.layout}")
        self.token_stride_size = self.head_num * self.head_dim * self.dtype.itemsize
        self.layout_dim = self.token_stride_size * self.layer_num
        buffer = torch.empty(
            dims,
            dtype=self.dtype,
            device=self.device,
        )
        if self.pin_memory:
            torch.cuda.cudart().cudaHostRegister(
                buffer.data_ptr(), buffer.numel() * buffer.element_size(), 0
            )
        return buffer

    def est_bandwidth_batch(self, size_bytes_arr: np.ndarray, cat: str):
        if MockTokenToKVPoolHost.MEMORY_READ_BANDWIDTH_BYTES is None:
            MockTokenToKVPoolHost.MEMORY_READ_BANDWIDTH_BYTES = (
                ConfigManager.get_platform_config().memory_read_bandwidth
            )
        if MockTokenToKVPoolHost.MEMORY_WRITE_BANDWIDTH_BYTES is None:
            MockTokenToKVPoolHost.MEMORY_WRITE_BANDWIDTH_BYTES = (
                ConfigManager.get_platform_config().memory_write_bandwidth
            )
        x = size_bytes_arr.astype(np.float64)
        if cat == "H2D":
            eff = 0.85
            t0 = 6.67e-6
            bw = MockTokenToKVPoolHost.MEMORY_READ_BANDWIDTH_BYTES * eff
        else:
            eff = 0.85
            t0 = 4e-6
            bw = MockTokenToKVPoolHost.MEMORY_WRITE_BANDWIDTH_BYTES * eff
        return x * bw / (t0 * bw + x)

    def load_to_device_per_layer(
        self, device_pool, host_indices, device_indices, layer_id, io_backend
    ) -> None:
        # update global clock
        # Merge cache indices
        # https://github.com/sgl-project/sglang/blob/v0.5.8/sgl-kernel/csrc/kvcacheio/transfer.cu#L713
        assert len(host_indices) == len(device_indices)
        num_indices = len(host_indices)

        host = np.asarray(host_indices.cpu(), dtype=np.int64)
        dev = np.asarray(device_indices.cpu(), dtype=np.int64)
        cont = (np.diff(host) == 1) & (np.diff(dev) == 1)
        cut = np.flatnonzero(~cont) + 1
        starts = np.r_[0, cut]
        ends = np.r_[cut, num_indices]
        seg_len = (ends - starts).astype(np.float64)

        if MockTokenToKVPoolHost.KV_CACHE_BYTES_PER_LAYER is None:
            MockTokenToKVPoolHost.KV_CACHE_BYTES_PER_LAYER = (
                ConfigManager.get_kv_cache_bytes_per_layer()
            )

        size_bytes_arr = seg_len * float(MockTokenToKVPoolHost.KV_CACHE_BYTES_PER_LAYER)
        bandwidth_arr = self.est_bandwidth_batch(size_bytes_arr, cat="H2D")
        total_time_cost = float(np.sum(size_bytes_arr / bandwidth_arr))
        # total_time_cost += 3.3e-6 * len(size_bytes_arr)  # CPU Overhead
        StateManager.inc_hicache_l2_load_dur(total_time_cost)

    def backup_from_device_all_layer(
        self, device_pool, host_indices, device_indices, io_backend
    ) -> None:
        """
        Backup KV data from the device memory pool to the host memory pool for all layers.
        """
        # update global clock
        num_indices = len(host_indices)

        host = np.asarray(host_indices.cpu(), dtype=np.int64)
        dev = np.asarray(device_indices.cpu(), dtype=np.int64)
        cont = (np.diff(host) == 1) & (np.diff(dev) == 1)
        cut = np.flatnonzero(~cont) + 1
        starts = np.r_[0, cut]
        ends = np.r_[cut, num_indices]
        seg_len = (ends - starts).astype(np.float64)

        if MockTokenToKVPoolHost.KV_CACHE_BYTES is None:
            MockTokenToKVPoolHost.KV_CACHE_BYTES = ConfigManager.get_kv_cache_bytes()

        size_bytes_arr = seg_len * float(MockTokenToKVPoolHost.KV_CACHE_BYTES)
        bandwidth_arr = self.est_bandwidth_batch(size_bytes_arr, cat="D2H")
        total_time_cost = float(np.sum(size_bytes_arr / bandwidth_arr))
        # total_time_cost += 3.3e-6 * len(size_bytes_arr)  # CPU Overhead

        StateManager.inc_hicache_l2_backup_dur(total_time_cost)

    def get_data_page(self, index, flat: bool = True) -> torch.Tensor:
        """
        Get a flat data page from the host memory pool.
        """
        return torch.ones(size=(1, 1)) * index

    def get_dummy_flat_data_page(self) -> torch.Tensor:
        """
        Get a dummy flat data page from the host memory pool.
        This is used for prefetching or initializing empty pages.
        """
        return torch.zeros(
            (2, self.layer_num, self.page_size, self.head_num, self.head_dim),
            dtype=self.dtype,
            device=self.device,
            pin_memory=self.pin_memory,
        ).flatten()

    def set_from_flat_data_page(self, index: int, data_page: torch.Tensor) -> None:
        """
        Set a flat data page to the host memory pool.
        """
        pass

    def clear(self):
        # Initialize memory states and tracking structures.
        self.mem_state = torch.zeros(
            (self.size,), dtype=torch.uint8, device=self.device
        )
        self.free_slots = torch.arange(self.size, dtype=torch.int64)

    def available_size(self):
        return len(self.free_slots)

    @synchronized
    def alloc(self, need_size: int) -> Optional[torch.Tensor]:
        assert need_size % self.page_size == 0, (
            "The requested size should be a multiple of the page size."
        )
        if need_size > self.available_size():
            return None

        select_index = self.free_slots[:need_size]
        self.free_slots = self.free_slots[need_size:]

        return select_index

    @synchronized
    def free(self, indices: torch.Tensor) -> int:
        self.free_slots = torch.cat([self.free_slots, indices])
        return len(indices)


def _pass_str_to_block_ids(hash_key):
    int_hash = int(hash_key, 16)
    MAX_18_DIGITS = 10**18
    return int_hash % MAX_18_DIGITS


class MockHiCacheStorage:
    def __init__(self, *args, **kwargs):
        if kvcm_py_optimizer is not None:
            logger.info("Using KVCM HiCache storage")
            self.init_kvcm()
        else:
            logger.info("Using common set HiCache storage")
            self.storage: set = set()
            self.storage_file_path: str = "/tmp/hisim/hicache/storage_keys.txt"
            os.makedirs(os.path.dirname(self.storage_file_path), exist_ok=True)

            if os.path.exists(self.storage_file_path):
                with open(self.storage_file_path) as f:
                    line = f.readline()
                    while line:
                        self.storage.add(line.strip())
                        line = f.readline()

        if Envs.reset_hicache_storage():
            logger.info(
                "Cleared KV cache saved in the storage backend because the system environment variable (`HISIM_RESET_HICACHE_STORAGE`) is set."
            )
            with open(self.storage_file_path, "w") as f:
                pass

    def init_kvcm(self):
        # Initialize kvcm based on reference examples
        self.temp_dir = tempfile.mkdtemp()
        logger.info(f"Using temporary directory: {self.temp_dir}")

        project_root = _CURRENT_DIR.parents[4]
        config_path = (
            project_root
            / "kv_cache_manager"
            / "optimizer"
            / "test"
            / "testdata"
            / "optimizer_startup_config_load.json"
        )
        config_path = config_path.resolve()
        self.config_loader = kvcm_py_optimizer.OptimizerConfigLoader()

        if not self.config_loader.load(str(config_path)):
            raise RuntimeError(f"Failed to load optimizer config from {config_path}")
        self.config = self.config_loader.config()
        self.storage_manager = kvcm_py_optimizer.OptimizerManager(self.config)
        self.storage_manager.Init()

        # Multi-instance not supported yet; using a single shared instance_id.
        self.instance_id = "3780643326877293460"

    def tearDown(self):
        if hasattr(self, "temp_dir"):
            import shutil

            logger.info("Cleaning up temporary directory")
            shutil.rmtree(self.temp_dir)

    def register_mem_pool_host(self, mem_pool_host):
        pass

    def set(
        self,
        key: str,
        value: Optional[Any] = None,
        target_location: Optional[Any] = None,
        target_sizes: Optional[Any] = None,
    ) -> bool:
        if self.exists(key):
            return True
        self.storage.add(key)
        with open(self.storage_file_path, "a+") as f:
            f.write(key + "\n")
        return True

    def batch_set(
        self,
        keys: List[str],
        values: Optional[Any] = None,
        extra_info = None,  # HiCacheStorageExtraInfo
        target_locations: Optional[Any] = None,
        target_sizes: Optional[Any] = None,
    ) -> bool:
        if hasattr(self, "storage_manager"):
            if extra_info and extra_info.prefix_keys is not None:
                complete_prefix_hashs = extra_info.prefix_keys + keys
            else:
                complete_prefix_hashs = keys
            int_hash_keys = [
                _pass_str_to_block_ids(key) for key in complete_prefix_hashs
            ]
            # insert to kvcm
            trace_id = "1"
            write_timestamp = int(time.time() * 1000)
            write_token_ids = [1]
            self.storage_manager.WriteCache(
                self.instance_id,
                trace_id,
                write_timestamp,
                int_hash_keys,
                write_token_ids,
            )
            return True
        else:
            for key, value in zip(keys, values):
                if not self.set(key, value):
                    return False
            return True

    def exists(self, key: str) -> bool:
        return key in self.storage

    def batch_exists(self, keys: List[str], extra_info) -> int:
        """
        Check if the keys exist in the storage.
        return the number of consecutive existing keys from the start.
        Can be overridden by subclasses for more efficient implementation.
        """
        if hasattr(self, "storage_manager"):
            int_hash_keys = [_pass_str_to_block_ids(key) for key in keys]
            # Call the kvcm interface to match L3 prefix
            trace_id = "2"
            read_timestamp = int(time.time() * 1000)
            read_token_ids = [1]
            mask_offset = 0
            res = self.storage_manager.GetCacheLocation(
                self.instance_id,
                trace_id,
                read_timestamp,
                int_hash_keys,
                read_token_ids,
                mask_offset,
            )
            logger.debug(f"{res.kvcm_hit_length=}")
            return res.kvcm_hit_length
        else:
            for i in range(len(keys)):
                if not self.exists(keys[i]):
                    return i
            return len(keys)

    def clear(self) -> bool:
        if hasattr(self, "storage_manager"):
            logger.info("Clear all storage cache in kvcm.")
            self.storage_manager.ClearAllCaches()
            return True
        else:
            self.storage.clear()
            with open(self.storage_file_path, "w"):
                pass
            return True
