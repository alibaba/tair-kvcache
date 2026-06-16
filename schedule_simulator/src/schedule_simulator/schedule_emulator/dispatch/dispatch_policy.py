import threading
import logging
import random
import abc
from typing import List, Optional

from schedule_simulator.schedule_emulator.types import (
    RouterConfig,
    FakeRequest,
    RoutingPolicy,
)
from schedule_simulator.schedule_emulator.sglang_scheduler import (
    SGLangScheduleEmulator,
)
from schedule_simulator.schedule_emulator.dispatch.tree import (
    MultiTenantRadixTree,
)

logger = logging.getLogger(__name__)


class Worker:
    """简化的 Worker, 只有 id 和负载信息"""

    def __init__(self, id: str):
        self.id = id
        self.total_req = 0
        self._load = 0
        self._lock = threading.Lock()
        self.healthy = True

    def is_healthy(self) -> bool:
        return self.healthy

    def get_load(self) -> int:
        """获取当前负载"""
        with self._lock:
            return self._load

    def increment_load(self) -> None:
        """请求开始时增加负载"""
        with self._lock:
            self._load += 1
            self.total_req += 1

    def decrement_load(self) -> None:
        """请求结束时减少负载"""
        with self._lock:
            self._load = max(0, self._load - 1)

    def update_load(self, completed_requests_len) -> None:
        """请求结束时减少负载"""
        with self._lock:
            self._load = max(0, self.total_req - completed_requests_len)


class BasePolicy(abc.ABC):
    """策略基类"""

    def __init__(self, num_schedulers: int, config: RouterConfig):
        self.num_schedulers = num_schedulers
        self.config = config
        self.name = "base"
        self.workers = self.init_workers(num_schedulers)

    @abc.abstractmethod
    async def select_worker(self, req: FakeRequest) -> int:
        """选择 worker 的核心逻辑"""
        pass

    async def update_load(
        self, schedulers: List[SGLangScheduleEmulator], current_time: float
    ):
        pass

    def init_workers(self, worker_count: int) -> List[Worker]:
        """初始化 worker 列表"""
        workers = []
        for i in range(worker_count):
            worker = Worker(id=f"worker_{i}")
            workers.append(worker)

        logger.debug(f"Initialized {worker_count} workers")
        return workers

    def complete_request(self, worker_idx: int) -> None:
        """
        请求完成时调用，减少 worker 负载

        Args:
            workers: worker 列表
            worker_idx: 完成请求的 worker 索引
        """
        if 0 <= worker_idx < len(self.workers):
            self.workers[worker_idx].decrement_load()

    async def update_workload(
        self, schedulers: List[SGLangScheduleEmulator], current_time: float
    ):
        if not schedulers:
            return
        is_prefill = schedulers[0].scheduler_config.scenario == "disagg_prefill"
        for i in range(len(schedulers)):
            completed_count = len(schedulers[i].completed_requests)
            last_batch = schedulers[i].last_batch
            # current_time在上一个batch时间内,且有请求结束,那么该请求是负载,不算已完成请求
            if last_batch is not None and last_batch.reqs:
                if current_time <= schedulers[i].last_batch_run_timestamp:
                    for req in last_batch.reqs:
                        output_token_length = (
                            1 if is_prefill else req.output_token_length
                        )
                        if len(req.gen_token_latencies) >= output_token_length:
                            completed_count -= 1

            self.workers[i].update_load(completed_count)

    async def get_workload(self):
        return [i.get_load() for i in self.workers]


class RandomPolicy(BasePolicy):
    def __init__(self, num_schedulers: int, config: RouterConfig):
        super().__init__(num_schedulers, config)
        self.name = RoutingPolicy.RANDOM

    async def select_worker(self, req):
        selected_idx = random.randint(0, self.num_schedulers - 1)
        self.workers[selected_idx].increment_load()
        return selected_idx


class RoundRobinPolicy(BasePolicy):
    def __init__(self, num_schedulers: int, config: RouterConfig):
        super().__init__(num_schedulers, config)
        self.name = RoutingPolicy.ROUND_ROBIN
        self.current_idx = 0

    async def select_worker(self, req: FakeRequest) -> Optional[int]:
        selected = self.current_idx
        self.current_idx = (selected + 1) % self.num_schedulers
        self.workers[selected].increment_load()
        return selected


class PowerOfTwoPolicy(BasePolicy):
    def __init__(self, num_schedulers: int, config: RouterConfig):
        super().__init__(num_schedulers, config)
        self.name = RoutingPolicy.POWER_OF_TWO
        self.loads: List[int] = [0] * num_schedulers
        self.next_update_time: float = 0
        self.check_interval = config.worker_startup_check_interval
        # 刚开始'token负载'未传回来的时 用 '请求负载'(同cache_aware)
        self.token_load_flag = False

    async def update_load(
        self, schedulers: List[SGLangScheduleEmulator], current_time: float
    ):
        if not schedulers:
            return
        # 异步更新负载
        if current_time > self.next_update_time:
            self.token_load_flag = True
            # 模拟异步获取负载
            self.loads = [await s.get_load(current_time) for s in schedulers]
            self.next_update_time = (
                current_time / self.check_interval * self.check_interval
                + self.check_interval
            )

        if not self.token_load_flag:
            await self.update_workload(schedulers, current_time)
            self.loads = await self.get_workload()

    async def select_worker(self, req: FakeRequest) -> Optional[int]:
        logger.debug(f"power_of_two loads {self.loads}")
        if self.num_schedulers < 2:
            return 0
        idx1, idx2 = random.sample(range(self.num_schedulers), 2)
        select_index = idx1 if self.loads[idx1] <= self.loads[idx2] else idx2
        if not self.token_load_flag:
            self.workers[select_index].increment_load()
        return select_index


class CacheAwarePolicyOld(BasePolicy):
    """[DEPRECATED] 旧的缓存感知负载均衡策略（Python 实现的 MultiTenantRadixTree）。

    已被新的 CacheAwarePolicy（基于 Optimizer C++ 前缀树）替代。
    使用 RoutingPolicy.CACHE_AWARE_OLD 访问此策略。
    """

    def __init__(self, num_schedulers: int, config: RouterConfig):
        super().__init__(num_schedulers, config)

        self.name = RoutingPolicy.CACHE_AWARE
        self.next_evict_time = config.eviction_interval_secs

        self.tree = MultiTenantRadixTree()
        for worker in self.workers:
            self.tree.insert("", worker.id, 0)

    def evict_cache(self, max_size: int) -> None:
        """执行缓存驱逐"""
        self.tree.evict_tenant_by_size(max_size)
        logger.debug(f"Cache eviction completed, max_size: {max_size}")

    async def select_worker(self, req: FakeRequest) -> Optional[int]:
        logger.debug(
            f"cache_aware loads {await self.get_workload()} tree:{list(self.tree.get_tenant_token_count().values())}"
        )
        if req.origin_input_ids:
            request_text = req.origin_input_ids
        else:
            token_len = req.input_token_length + len(req.gen_token_latencies)
            request_text = [random.randint(1, 10000) for _ in range(token_len)]
        timestamp = req.last_event_time

        """选择 worker"""
        healthy_indices = [i for i, w in enumerate(self.workers) if w.is_healthy()]

        if not healthy_indices:
            self.workers[0].increment_load()
            return 0

        loads = [self.workers[i].get_load() for i in healthy_indices]

        max_load = max(loads) if loads else 0
        min_load = min(loads) if loads else 0

        is_imbalanced = (max_load - min_load > self.config.balance_abs_threshold)

        if is_imbalanced:
            logger.debug(
                f"Load balancing | max: {max_load} | min: {min_load} | workers: {loads}"
            )

            min_load_idx = min(
                healthy_indices, key=lambda idx: self.workers[idx].get_load()
            )

            if request_text:
                self.tree.insert(request_text, self.workers[min_load_idx].id, timestamp)

            # 增加选中 worker 的负载
            self.workers[min_load_idx].increment_load()
            return min_load_idx

        text = request_text or ""
        matched_text, matched_tenant = self.tree.prefix_match(text, timestamp)

        match_rate = 0.0
        if text:
            match_rate = len(matched_text) / len(text)

        if match_rate > self.config.cache_threshold:
            logger.debug(f"Cache hit: {match_rate:.2f}")
            selected_id = matched_tenant
        else:
            logger.debug(f"Cache miss: {match_rate:.2f}")
            selected_id = self.tree.get_smallest_tenant()

        selected_idx = None
        for i, w in enumerate(self.workers):
            if w.id == selected_id and w.is_healthy():
                selected_idx = i
                break

        logger.debug(
            f"match_rate:{match_rate} matched_tenant:{matched_tenant} selected_idx:{selected_idx}"
        )
        if selected_idx is not None:
            self.tree.insert(text, self.workers[selected_idx].id, timestamp)
            # 增加选中 worker 的负载
            self.workers[selected_idx].increment_load()
            return selected_idx
        else:
            if selected_id != "empty":
                self.tree.remove_tenant(selected_id)
                logger.debug(f"Removed stale worker {selected_id} from cache tree")

        self.tree.insert(text, self.workers[healthy_indices[0]].id, timestamp)
        self.workers[healthy_indices[0]].increment_load()
        return healthy_indices[0]

    async def update_load(
        self, schedulers: List[SGLangScheduleEmulator], current_time: float
    ):
        if not schedulers:
            return
        if current_time > self.next_evict_time:  # Timed eviction
            self.evict_cache(self.config.max_tree_size)
            self.next_evict_time += self.config.eviction_interval_secs

        if not schedulers[0].scheduler_config.request_level_scheduling:
            await self.update_workload(schedulers, current_time)



class CacheAwarePolicy(BasePolicy):
    """缓存感知的负载均衡策略（基于 Optimizer C++ 前缀树）。

    在路由层维护一棵独立的 Optimizer HierarchicalReplayManager 作为近似前缀树，
    替代旧的 Python MultiTenantRadixTree 实现。

    核心优势：使用 ChooseBestEngine 单次 C++ 调用查询所有 Pod 的前缀匹配，
    替代旧的逐 Pod Python 前缀匹配，性能提升显著。

    工作流程：
    1. 请求到达时，调用 ChooseBestEngine 找到前缀匹配最长的 Pod（单次 C++ 调用）
    2. 如果匹配率 > cache_threshold，路由到该 Pod
    3. 否则路由到负载最小的 Pod
    4. 路由决策后，WriteCache 到近似前缀树以反映分配
    5. 定时驱逐（eviction）保持树大小可控
    """

    def __init__(self, num_schedulers: int, config: RouterConfig,
                 scheduler_config=None, platform_config=None):
        super().__init__(num_schedulers, config)

        self.name = RoutingPolicy.CACHE_AWARE
        self.next_evict_time = config.eviction_interval_secs

        # SchedulerConfig and PlatformConfig for lazy init
        self._scheduler_config = scheduler_config
        self._platform_config = platform_config

        # Optimizer tree (lazy init in update_load)
        self._optimizer_manager = None
        self._engine_ids: list = []
        self._engine_id_to_idx: dict = {}
        self._initialized = False

    def _init_optimizer(self, schedulers):
        """Lazy init: create a dedicated Optimizer instance for routing."""
        try:
            import sys
            kvcm_so_dir = "/sgl-workspace/claude_workspace/tair-kvcache/bazel-bin/kv_cache_manager/optimizer/pybind"
            if kvcm_so_dir not in sys.path:
                sys.path.insert(0, kvcm_so_dir)
            import kvcm_py_optimizer as kvcm
        except ImportError:
            logger.warning("kvcm_py_optimizer not available, falling back to load-balance only")
            self._initialized = True
            return

        from schedule_simulator.schedule_emulator.hierarchical_config_builder import build_hierarchical_config

        # Create independent Optimizer instance for routing layer
        self._engine_ids = [f"P{i}" for i in range(self.num_schedulers)]
        for i, eid in enumerate(self._engine_ids):
            self._engine_id_to_idx[eid] = i

        sc = self._scheduler_config
        pc = self._platform_config

        if sc is None:
            from schedule_simulator.schedule_emulator.types import SchedulerConfig
            sc = SchedulerConfig("Qwen2.5-3B", hicache_storage_backend="hf3fs")
        if pc is None:
            from schedule_simulator.schedule_emulator.types import PlatformConfig
            pc = PlatformConfig(device="H20")

        import tempfile
        output_dir = tempfile.mkdtemp(prefix="cache_aware_router_")
        config_path = build_hierarchical_config(
            scheduler_config=sc,
            platform_config=pc,
            p_instance_ids=self._engine_ids,
            output_dir=output_dir,
            storage_pool_capacity_gb=0.0001,  # Router doesn't need storage pool
            enable_p2p=True,  # Must match real tree's P2P setting
        )

        loader = kvcm.HierarchicalReplayConfigLoader()
        if not loader.load(config_path):
            raise ValueError(f"Failed to load router optimizer config from {config_path}")

        self._optimizer_manager = kvcm.HierarchicalReplayManager(loader.config())
        if not self._optimizer_manager.Init():
            raise RuntimeError("Router Optimizer Init failed")

        self._initialized = True
        logger.info(f"CacheAwarePolicy: Initialized router Optimizer with {self.num_schedulers} engines")

    def evict_cache(self, max_size: int) -> None:
        """Optimizer handles LRU eviction internally via capacity limits."""
        logger.debug(f"Cache eviction: Optimizer handles LRU internally (max_tree_size={max_size})")

    async def select_worker(self, req: FakeRequest) -> Optional[int]:
        try:
            # Lazy init: ensure optimizer is ready before using it
            if not self._initialized:
                # This should have been called from update_load first
                logger.warning("CacheAwarePolicy.select_worker called before initialization")
                # Fall back to round-robin
                healthy_indices = [i for i, w in enumerate(self.workers) if w.is_healthy()]
                if not healthy_indices:
                    return 0
                min_load_idx = min(healthy_indices, key=lambda idx: self.workers[idx].get_load())
                self.workers[min_load_idx].increment_load()
                return min_load_idx

            healthy_indices = [i for i, w in enumerate(self.workers) if w.is_healthy()]
            if not healthy_indices:
                self.workers[0].increment_load()
                return 0

            loads = [self.workers[i].get_load() for i in healthy_indices]
            max_load = max(loads) if loads else 0
            min_load = min(loads) if loads else 0

            is_imbalanced = (max_load - min_load > self.config.balance_abs_threshold)

            if is_imbalanced:
                logger.debug(
                    f"Load balancing | max: {max_load} | min: {min_load} | workers: {loads}"
                )
                min_load_idx = min(
                    healthy_indices, key=lambda idx: self.workers[idx].get_load()
                )
                engine_id = self._engine_ids[min_load_idx] if self._engine_ids and min_load_idx < len(self._engine_ids) else self.workers[min_load_idx].id
                self._write_to_approx_tree(req, engine_id)
                self.workers[min_load_idx].increment_load()
                return min_load_idx

            # Use Optimizer C++ ChooseBestEngine (single C++ call across all engines)
            best_idx = None
            best_hit = 0
            if self._optimizer_manager and req.origin_input_ids:
                try:
                    block_ids = list(req.origin_input_ids)
                    timestamp_ns = int(req.last_event_time * 1e9) if req.last_event_time else 0
                    res = self._optimizer_manager.ChooseBestEngine(block_ids, timestamp_ns)
                    if res.hit_count > 0 and res.engine_instance_id in self._engine_id_to_idx:
                        best_idx = self._engine_id_to_idx[res.engine_instance_id]
                        best_hit = res.hit_count
                except Exception as e:
                    logger.error(f"CacheAwarePolicy: ChooseBestEngine failed: {e}")
                    # Fall back to load balancing on error
                    min_load_idx = min(
                        healthy_indices, key=lambda idx: self.workers[idx].get_load()
                    )
                    self.workers[min_load_idx].increment_load()
                    return min_load_idx

            # Check match rate
            if best_idx is not None and req.input_token_length > 0:
                page_size = 1
                if self._scheduler_config and self._scheduler_config.page_size:
                    page_size = self._scheduler_config.page_size
                matched_tokens = best_hit * page_size
                match_rate = matched_tokens / req.input_token_length
                if match_rate > self.config.cache_threshold:
                    logger.debug(f"Cache hit: match_rate={match_rate:.2f}, hit_blocks={best_hit}")
                    engine_id = self._engine_ids[best_idx] if self._engine_ids and best_idx < len(self._engine_ids) else self.workers[best_idx].id
                    self._write_to_approx_tree(req, engine_id)
                    self.workers[best_idx].increment_load()
                    return best_idx

            logger.debug(f"Cache miss: best_hit={best_hit}, routing to min load")
            min_load_idx = min(
                healthy_indices, key=lambda idx: self.workers[idx].get_load()
            )
            engine_id = self._engine_ids[min_load_idx] if self._engine_ids and min_load_idx < len(self._engine_ids) else self.workers[min_load_idx].id
            self._write_to_approx_tree(req, engine_id)
            self.workers[min_load_idx].increment_load()
            return min_load_idx
        except Exception as e:
            logger.error(f"CacheAwarePolicy.select_worker unexpected error: {e}", exc_info=True)
            # Ultimate fallback: return first healthy worker
            healthy_indices = [i for i, w in enumerate(self.workers) if w.is_healthy()]
            if healthy_indices:
                idx = healthy_indices[0]
                self.workers[idx].increment_load()
                return idx
            return 0

    def _write_to_approx_tree(self, req: FakeRequest, engine_id: str):
        """Write request block_ids to the approximate prefix tree."""
        try:
            if not self._optimizer_manager or not req.origin_input_ids:
                return
            block_ids = list(req.origin_input_ids)
            timestamp_ns = int(req.last_event_time * 1e9) if req.last_event_time else 0
            page_size = 1
            if self._scheduler_config and self._scheduler_config.page_size:
                page_size = self._scheduler_config.page_size
            max_full_blocks = req.input_token_length // page_size
            if max_full_blocks == 0 and block_ids:
                max_full_blocks = 1
            full_block_ids = block_ids[:max_full_blocks]
            self._optimizer_manager.WriteCache(engine_id, f"r{req.id}", timestamp_ns, full_block_ids)
        except Exception as e:
            logger.error(f"CacheAwarePolicy._write_to_approx_tree failed: {e}")
            # Don't re-raise, just log and continue

    async def update_load(
        self, schedulers: List[SGLangScheduleEmulator], current_time: float
    ):
        if not schedulers:
            return

        # Lazy init Optimizer
        if not self._initialized:
            self._init_optimizer(schedulers)

        if current_time > self.next_evict_time:
            self.evict_cache(self.config.max_tree_size)
            self.next_evict_time += self.config.eviction_interval_secs

        await self.update_workload(schedulers, current_time)


class DirectCacheAwarePolicy(BasePolicy):
    """Routes requests to the scheduler with the longest real prefix cache match.

    Unlike CacheAwarePolicy which maintains an approximate router-side radix tree,
    this policy directly queries each scheduler's actual tree_cache. This eliminates
    staleness at negligible cost in simulation.

    For HierarchicalCacheAdapter: uses ChooseBestEngine (single C++ call across all engines).
    For SimHiRadixCache: queries each scheduler's radix tree with read-only prefix match.
    """

    def __init__(self, num_schedulers: int, config: RouterConfig):
        super().__init__(num_schedulers, config)
        self.name = RoutingPolicy.DIRECT_CACHE_AWARE
        self.schedulers: List[SGLangScheduleEmulator] = []
        self._engine_id_to_idx: dict = {}
        self._use_hierarchical: bool = False

    def _query_prefix_hit(self, scheduler: SGLangScheduleEmulator, req: FakeRequest) -> int:
        tree_cache = scheduler.tree_cache
        if not hasattr(tree_cache, "query_prefix_length"):
            return 0
        if not req.origin_input_ids:
            return 0
        from schedule_simulator.schedule_emulator.kvcache_simulation.kvcache_base_classes import (
            RadixKey,
        )
        return tree_cache.query_prefix_length(
            RadixKey(token_ids=list(req.origin_input_ids), extra_key=None),
            req.input_token_length
        )

    def _choose_best_engine_fast(self, req: FakeRequest) -> Optional[tuple]:
        """Use single ChooseBestEngine C++ call. Returns (scheduler_idx, hit_count) or None."""
        if not req.origin_input_ids:
            return None
        adapter = self.schedulers[0].tree_cache
        block_ids = list(req.origin_input_ids)
        timestamp_ns = int(adapter.global_values.clock * 1e9)
        res = adapter.choose_best_engine(block_ids, timestamp_ns, req.input_token_length)
        if res.hit_count > 0 and res.engine_instance_id in self._engine_id_to_idx:
            return (self._engine_id_to_idx[res.engine_instance_id], res.hit_count)
        return None

    async def select_worker(self, req: FakeRequest) -> Optional[int]:
        healthy_indices = [i for i, w in enumerate(self.workers) if w.is_healthy()]
        if not healthy_indices:
            self.workers[0].increment_load()
            return 0

        loads = [self.workers[i].get_load() for i in healthy_indices]
        max_load = max(loads) if loads else 0
        min_load = min(loads) if loads else 0
        is_imbalanced = (max_load - min_load > self.config.balance_abs_threshold)
        if is_imbalanced:
            min_load_idx = min(
                healthy_indices, key=lambda idx: self.workers[idx].get_load()
            )
            self.workers[min_load_idx].increment_load()
            return min_load_idx

        best_idx = None
        best_hit = 0

        if self.schedulers and self._use_hierarchical:
            result = self._choose_best_engine_fast(req)
            if result is not None:
                best_idx, best_hit = result
        elif self.schedulers:
            for idx in healthy_indices:
                if idx < len(self.schedulers):
                    hit = self._query_prefix_hit(self.schedulers[idx], req)
                    if hit > best_hit:
                        best_hit = hit
                        best_idx = idx

        if best_idx is not None and req.input_token_length > 0:
            page_size = 1
            if self.schedulers and self.schedulers[0].scheduler_config.page_size:
                page_size = self.schedulers[0].scheduler_config.page_size
            matched_tokens = best_hit * page_size
            match_rate = matched_tokens / req.input_token_length
            if match_rate > self.config.cache_threshold:
                self.workers[best_idx].increment_load()
                return best_idx

        min_load_idx = min(
            healthy_indices, key=lambda idx: self.workers[idx].get_load()
        )
        self.workers[min_load_idx].increment_load()
        return min_load_idx

    async def update_load(
        self, schedulers: List[SGLangScheduleEmulator], current_time: float
    ):
        if not schedulers:
            return
        self.schedulers = schedulers
        if not self._engine_id_to_idx:
            from schedule_simulator.schedule_emulator.hierarchical_cache_adapter import (
                HierarchicalCacheAdapter,
            )
            for i, sched in enumerate(schedulers):
                if isinstance(sched.tree_cache, HierarchicalCacheAdapter):
                    self._use_hierarchical = True
                    self._engine_id_to_idx[sched.tree_cache.engine_id] = i
        await self.update_workload(schedulers, current_time)
