import tempfile
from typing import Optional, Union, List
from kunlun_commons.utils import get_logger

try:
    from kv_cache_manager.optimizer.pybind import kvcm_py_optimizer
except ImportError:
    kvcm_py_optimizer = None

logger = get_logger("kvcache_utils")


class KVCachePool:
    """
    对kvcache内存池的简单模拟，维护一个全局kv容量,对应sglang的TokenToKVPoolAllocator
    不区分host侧或dev侧
    """

    def __init__(
        self,
        size: int,
        page_size: int,
    ):
        self.size = size
        self.available_size = size
        self.page_size = page_size
        self.evictable_size = 0

    def get_available_size(self):
        return self.available_size

    def get_evictable_size(self):
        return self.evictable_size

    def alloc(self, need_size: int):
        if need_size > self.available_size:
            return None
        self.available_size -= need_size
        self.evictable_size += need_size
        return need_size

    def free(self, free_size: int):
        if free_size + self.available_size > self.size:
            return None
        self.available_size += free_size
        self.evictable_size -= free_size
        return free_size

    def clear(self):
        self.available_size = self.size
        self.evictable_size = 0


class ReqToTokenPoolHost:
    """模拟req_to_token_pool，记录每个req已记录的长度，不记录具体的indices."""

    def __init__(
        self,
        size: int,
        max_context_len: int,
    ):
        self.size = size  # 最大req数量
        self.max_context_len = max_context_len  # 每个req最大长度
        self.req_length = [0] * size  # 存储req已缓存的长度

        self.free_slots = list(range(size))

    # 写入逻辑替换为长度更新，接口名与原框架保持一致
    def write(self, indices, new_length: int):
        self.req_length[indices] = new_length

    def available_size(self):
        return len(self.free_slots)

    def alloc(self, need_size: int) -> List[int]:
        if need_size > len(self.free_slots):
            return None

        select_index = self.free_slots[:need_size]
        self.free_slots = self.free_slots[need_size:]

        return select_index

    def free(self, free_index: Union[int, List[int]]):
        if isinstance(free_index, (int,)):
            self.free_slots.append(free_index)
        else:
            self.free_slots.extend(free_index)

    def clear(self):
        self.free_slots = list(range(self.size))


# 用于替换hicache中的HiCacheController
# mem_pool_device对应token_to_kv_pool_allocator，负责记录device上的内存分配与释放
# mem_pool_host对应mem_pool_host，负责记录host侧的内存分配与释放
class SimHiCacheController:
    def __init__(
        self,
        mem_pool_device: KVCachePool,
        mem_pool_host: KVCachePool,
        page_size: int,
        hicache_write_policy: str = "write_through",
        storage_backend: Optional[str] = "kvcm",
    ):
        self.page_size = page_size
        self.mem_pool_device = mem_pool_device
        self.mem_pool_host = mem_pool_host

        self.prefetch_tokens_occupied = 0
        self.prefetch_capacity_limit = mem_pool_device.get_available_size()
        self.write_policy = hicache_write_policy

        self.write_queue = []
        self.load_queue = []

        # only support kvcm
        if storage_backend == "kvcm" and kvcm_py_optimizer is not None:
            # kvcm相关部分初始化
            self.init_kvcm()
        self.timestamp = 0

    def init_kvcm(self):
        # 参考用例对kvcm进行初始化
        self.temp_dir = tempfile.mkdtemp()
        print(f"Using temporary directory: {self.temp_dir}")
        # 直接初始化 OptimizerManager
        optimizer_config = kvcm_py_optimizer.OptimizerConfig()
        tier = kvcm_py_optimizer.TierConfig()
        tier.set_unique_name("tier1")
        tier.set_storage_type(kvcm_py_optimizer.DataStorageType.DATA_STORAGE_TYPE_HF3FS)
        tier.set_priority(0)
        tier.set_eviction_policy_type(kvcm_py_optimizer.EvictionPolicyType.LRU)
        tier.set_capacity(1000000)
        tier.set_band_width_mbps(1000)
        optimizer_config.set_tiers([tier])
        optimizer_config.set_trace_type(kvcm_py_optimizer.TraceType.TRACE_PUBLISHER_LOG)
        optimizer_config.set_block_size(1)
        # load trace时会自动根据 trace type 定义是否开启读写分离
        # 这里强制开启读写分离以测试读写逻辑
        optimizer_config.set_rw_separation(True)
        self.storage_manager = kvcm_py_optimizer.OptimizerManager(optimizer_config)

    def tearDown(self):
        # 清理临时目录
        if hasattr(self, "temp_dir"):
            import shutil

            logger.info("清理临时目录")
            shutil.rmtree(self.temp_dir)

    def reset(self):
        self.mem_pool_device.clear()
        self.mem_pool_host.clear()

    def print_mem_stats(self):
        print(
            f"device: available_size: {self.mem_pool_device.get_available_size()}, evictable_size: {self.mem_pool_device.get_evictable_size()}"
        )
        print(
            f"host: available_size: {self.mem_pool_host.get_available_size()}, evictable_size: {self.mem_pool_host.get_evictable_size()}"
        )

    def write(self, indices_len: int):
        """
        将KV缓存从device备份到host,仅模拟内存池大小变化
        后续应维护一个write_queue以模拟时间顺序
        """
        host_indices_len = self.mem_pool_host.alloc(indices_len)
        return host_indices_len

    def evict_device(self, device_indices_len: int) -> int:
        ret = self.mem_pool_device.free(device_indices_len)
        return ret

    def evict_host(self, host_indices_len) -> int:
        ret = self.mem_pool_host.free(host_indices_len)
        return ret

    def load(self, host_indices_len: int) -> int:
        ret = self.mem_pool_device.alloc(host_indices_len)
        return ret

    def get_prefetch_length(self, ids: List[int], timestamp=0):
        # 未初始化kvcm则返回0
        if not hasattr(self, "storage_manager"):
            return 0
        # 调用kvcm接口进行L3前缀查询
        trace_id = "1"
        read_timestamp = int(timestamp * 1000)
        read_block_ids = ids
        read_token_ids = [1]
        mask_offset = 0
        res = self.storage_manager.GetCacheLocation(
            trace_id, read_timestamp, read_block_ids, read_token_ids, mask_offset
        )
        return res.kvcm_hit_length

    def write_storage(self, ids: List[int], timestamp=0):
        if not hasattr(self, "storage_manager"):
            return
        trace_id = "111"
        write_timestamp = int(timestamp * 1000)
        write_block_ids = ids
        write_token_ids = [1]
        self.storage_manager.WriteCache(
            trace_id, write_timestamp, write_block_ids, write_token_ids
        )

    # 预取速率限制，功能需要进一步完善
    def prefetch_rate_limited(self) -> bool:
        if self.prefetch_tokens_occupied >= self.prefetch_capacity_limit:
            return True
        return False

    def __del__(self):
        self.tearDown()
