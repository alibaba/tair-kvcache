from typing import List, Optional, Any
import tempfile
import time
from pathlib import Path
import os

from hisim.utils.logger import get_logger
from hisim.simulation.manager import Envs
from hisim.hook import BaseHook

try:
    from kv_cache_manager.optimizer.pybind import kvcm_py_optimizer
except ImportError:
    kvcm_py_optimizer = None

# from sglang.srt.managers.schedule_batch import Req

logger = get_logger("hisim")
_CURRENT_DIR = Path(__file__).parent.resolve()


def _pass_str_to_block_ids(hash_key):
    int_hash = int(hash_key, 16)
    MAX_18_DIGITS = 10**18
    return int_hash % MAX_18_DIGITS


class C_StorageBackendFactory(BaseHook):
    HOOK_CLASS_NAME = "StorageBackendFactory"
    HOOK_MODULE_NAME = "sglang.srt.mem_cache.storage.backend_factory"

    @classmethod
    def hook(cls, target):
        def override_create_backend(cls, *args, **kwargs):
            logger.info("Creating hijacked cache storage backend.")
            return MockHiCacheStorage()

        target.create_backend = override_create_backend


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
        extra_info=None,  # HiCacheStorageExtraInfo
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
