import asyncio
from typing import Optional
from copy import deepcopy
import os
import time
import csv

from schedule_simulator.schedule_emulator.types import (
    BenchmarkConfig,
    SchedulerConfig,
    PlatformConfig,
    RouterConfig,
    RoutingPolicy,
    IterationStats,
    RequestCacheFetchStats,
    IterationCacheFetchStats,
    FakeRequest,
)
from schedule_simulator.schedule_emulator.benchmark import BenchmarkEmulator
from schedule_simulator.schedule_emulator.sglang_scheduler import (
    SGLangScheduleEmulator,
)
from schedule_simulator.infer_time_predictor import (
    InferTimePredictor,
)
from schedule_simulator.schedule_emulator.dispatch.dispatch_policy import (
    BasePolicy,
    RandomPolicy,
    RoundRobinPolicy,
    PowerOfTwoPolicy,
    CacheAwarePolicy,
    CacheAwarePolicyOld,
    DirectCacheAwarePolicy,
)
from schedule_simulator.schedule_emulator.schedule_policy import SchedulePolicy
from schedule_simulator.schedule_emulator.hierarchical_config_builder import build_hierarchical_config

from kunlun_commons.utils.logger import get_logger


logger = get_logger("schedule_simulator")


class BenchmarkRunner:
    def __init__(
        self,
        benchmark_config: BenchmarkConfig,  # Don’t remove it just to support the old API.
        scheduler_config: SchedulerConfig,
        platform_config: PlatformConfig,
        infer_time_predictor: Optional[InferTimePredictor] = None,
        use_real_token: bool = False,
        kvcm_block_size: int = None,
    ):
        request_queue = asyncio.Queue()
        response_queue = asyncio.Queue()

        self.benchmark_emulator: BenchmarkEmulator = None
        self.benchmark_config = benchmark_config
        self.use_real_token = use_real_token
        self.kvcm_block_size = kvcm_block_size

        self.benchmark_emulator = BenchmarkEmulator(
            benchmark_config,
            request_queue=request_queue,
            response_queue=response_queue,
            use_real_token_ids=use_real_token,
        )

        self.scheduler_emulator = SGLangScheduleEmulator(
            scheduler_config=scheduler_config,
            platform_config=platform_config,
            request_queue=request_queue,
            response_queue=response_queue,
            time_predictor=infer_time_predictor,
            use_real_token=use_real_token,
            kvcm_block_size=kvcm_block_size,
        )

    async def async_run_benchmark_emulation(self):
        benchmark_task = asyncio.create_task(self.benchmark_emulator.benchmark())
        scheduler_task = asyncio.create_task(self.scheduler_emulator.event_loop())

        try:
            # While it is only necessary to wait for the benchmark task,
            # waiting for both the scheduled and benchmark tasks is required to enable a quick exit in case of any errors.
            done, pending = await asyncio.wait(
                [benchmark_task, scheduler_task], return_when=asyncio.FIRST_COMPLETED
            )

            if scheduler_task in done and scheduler_task.exception() is not None:
                raise scheduler_task.exception()

            # Wait for the benchmark task to finish if the scheduled task has completed normally,
            # and the benchmark task is still in progress.
            if benchmark_task in pending:
                await benchmark_task

            scheduler_task.cancel()

            if benchmark_task.exception() is not None:
                raise benchmark_task.exception()

            metrics = benchmark_task.result()
            return metrics

        except Exception as e:
            logger.error(f"An error occurred during benchmark emulation: {e}")
            raise e

    def run_benchmark_emulation(
        self,
        benchmark_config: BenchmarkConfig | None = None,
        reset_scheduler: bool = True,
    ):
        if benchmark_config is None:
            benchmark_config = self.benchmark_config

        # Update synchronization queue
        new_request_queue = asyncio.Queue()
        new_response_queue = asyncio.Queue()
        self.scheduler_emulator.request_queue = new_request_queue
        self.scheduler_emulator.response_queue = new_response_queue
        # Keep benchmark endpoint, which has all response.

        self.benchmark_emulator = BenchmarkEmulator(
            benchmark_config,
            request_queue=new_request_queue,
            response_queue=new_response_queue,
            use_real_token_ids=self.use_real_token,
            kvcm_block_size=self.kvcm_block_size,
        )
        if reset_scheduler:
            self.scheduler_emulator.reset()
        self.scheduler_emulator.set_num_requests(
            (benchmark_config.num_prompts + benchmark_config.num_instances - 1)
            // benchmark_config.num_instances
        )

        start = time.time()
        metrics = asyncio.run(self.async_run_benchmark_emulation())
        end = time.time()
        metrics["time_cost"] = end - start
        return metrics

    def get_iteration_stats(self) -> list[IterationStats]:
        stats = self.scheduler_emulator.get_iteration_stats()
        if len(stats) == 0:
            logger.warning(
                "No iteration stats available. Was the benchmark already executed?"
                if self.scheduler_emulator.scheduler_config.enable_stats
                else "Stats information collection is disabled."
            )
        return stats

    def get_request_cache_fetch_stats(self) -> list[RequestCacheFetchStats]:
        stats = list(
            self.scheduler_emulator.tree_cache.get_request_fetch_stats().values()
        )
        if len(stats) == 0:
            logger.warning(
                "No iteration stats available. Was the benchmark already executed?"
                if self.scheduler_emulator.scheduler_config.enable_stats
                else "Stats information collection is disabled."
            )
        return stats

    def get_iteration_cache_fetch_stats(self) -> list[IterationCacheFetchStats]:
        return list(
            self.scheduler_emulator.tree_cache.get_iteration_fetch_stats().values()
        )

    def get_hierarchical_metrics(self) -> dict:
        if self.hierarchical_manager is None:
            return {}
        from schedule_simulator.schedule_emulator.hierarchical_cache_adapter import HierarchicalCacheAdapter
        total = {"total_blocks_queried": 0, "total_blocks_hit": 0,
                 "num_reads": 0, "num_writes": 0, "read_records": [], "write_records": []}
        for sched in self.p_schedulers + self.d_schedulers:
            if isinstance(sched.tree_cache, HierarchicalCacheAdapter):
                m = sched.tree_cache.get_hierarchical_metrics()
                total["total_blocks_queried"] += m.get("total_blocks_queried", 0)
                total["total_blocks_hit"] += m.get("total_blocks_hit", 0)
                total["num_reads"] += m["num_reads"]
                total["num_writes"] += m["num_writes"]
                total["read_records"].extend(sched.tree_cache.read_records)
                total["write_records"].extend(sched.tree_cache.write_records)
        total_engine_hit_blocks = sum(r.engine_hit for r in total["read_records"])
        total_peer_hit_blocks = sum(r.peer_hit for r in total["read_records"])
        total_pool_hit_blocks = sum(r.pool_hit for r in total["read_records"])
        tb = max(total["total_blocks_queried"], 1)
        total["total_engine_hit_blocks"] = total_engine_hit_blocks
        total["total_peer_hit_blocks"] = total_peer_hit_blocks
        total["total_pool_hit_blocks"] = total_pool_hit_blocks
        total["engine_hit_block_ratio"] = total_engine_hit_blocks / tb
        total["peer_hit_block_ratio"] = total_peer_hit_blocks / tb
        total["pool_hit_block_ratio"] = total_pool_hit_blocks / tb
        total["block_hit_ratio"] = total["total_blocks_hit"] / tb
        return total

    def analyze_hierarchical_results(self):
        if self.hierarchical_manager is not None:
            self.hierarchical_manager.AnalyzeResults()


    def export_results(self, output_dir: str, metrics: dict = None):
        """Export simulation results to files: summary JSON, per-request CSV, per-iteration CSV."""
        import json as _json
        os.makedirs(output_dir, exist_ok=True)
        results = self.get_response_results()

        if metrics is None:
            from schedule_simulator.schedule_emulator.utils import calc_metrics
            metrics = calc_metrics(results)

        # 1. Summary JSON (metrics + hierarchical)
        summary = dict(metrics)
        hier = self.get_hierarchical_metrics()
        if hier:
            for k, v in hier.items():
                if k not in ("read_records", "write_records"):
                    summary["hierarchical_" + k] = v
        with open(os.path.join(output_dir, "simulation_summary.json"), "w") as f:
            _json.dump(summary, f, indent=2)

        # 2. Per-request CSV
        hit_map = {}
        if hier and "read_records" in hier:
            for r in hier["read_records"]:
                hit_map[r.req_id] = (r.engine_hit, r.peer_hit, r.pool_hit, r.num_blocks)

        with open(os.path.join(output_dir, "per_request.csv"), "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["req_id", "input_length", "output_length",
                        "ttft_ms", "e2e_latency_ms", "queue_wait_ms",
                        "cache_reused_tokens",
                        "engine_hit", "peer_hit", "pool_hit", "num_blocks"])
            for req in sorted(results, key=lambda r: r.id):
                ttft = req.gen_token_latencies[0] * 1000 if req.gen_token_latencies else 0
                e2e = sum(req.gen_token_latencies) * 1000
                qw = (req.queue_time_end - req.queue_time_start) * 1000 if req.queue_time_start >= 0 else 0
                eh, ph, sh, nb = hit_map.get(req.id, (0, 0, 0, 0))
                w.writerow([req.id, req.input_token_length, req.output_token_length,
                            round(ttft, 3), round(e2e, 3), round(qw, 3),
                            req.final_reused_tokens, eh, ph, sh, nb])

        # 3. Per-iteration CSV
        with open(os.path.join(output_dir, "per_iteration.csv"), "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["pod", "iteration", "timestamp", "iter_latency_ms",
                        "num_ctx_requests", "num_ctx_tokens", "num_gen_requests",
                        "num_waiting"])
            for i, sched in enumerate(self.p_schedulers + self.d_schedulers):
                pod_name = "P%d" % i if i < len(self.p_schedulers) else "D%d" % (i - len(self.p_schedulers))
                for s in sched.get_iteration_stats():
                    w.writerow([pod_name, s.iter, round(s.timestamp, 6),
                                round(s.iter_latency_ms, 3),
                                s.num_context_requests, s.num_ctx_tokens,
                                s.num_gen_requests, s.num_waiting_requests])


        # 4. Per-pod stats CSV
        from schedule_simulator.schedule_emulator.hierarchical_cache_adapter import HierarchicalCacheAdapter
        all_scheds = self.p_schedulers + self.d_schedulers if hasattr(self, 'p_schedulers') else [self.scheduler_emulator]
        with open(os.path.join(output_dir, "per_pod_stats.csv"), "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["pod", "total_requests", "total_input_tokens", "total_output_tokens",
                        "total_blocks", "total_engine_hit_blocks", "total_peer_hit_blocks",
                        "total_pool_hit_blocks"])
            num_p = len(self.p_schedulers) if hasattr(self, 'p_schedulers') else len(all_scheds)
            for i, sched in enumerate(all_scheds):
                pod_name = "P%d" % i if i < num_p else "D%d" % (i - num_p)
                completed = sched.completed_requests
                total_reqs = len(completed)
                total_input = sum(r.input_token_length for r in completed)
                total_output = sum(r.output_token_length for r in completed)
                page_size = sched.scheduler_config.page_size or 1
                total_blocks = sum(max(r.input_token_length // page_size, 1) for r in completed)
                engine_hit = 0
                peer_hit = 0
                pool_hit = 0
                if isinstance(sched.tree_cache, HierarchicalCacheAdapter):
                    engine_hit = sched.tree_cache.total_engine_hit_blocks
                    peer_hit = sched.tree_cache.total_peer_hit_blocks
                    pool_hit = sched.tree_cache.total_pool_hit_blocks
                w.writerow([pod_name, total_reqs, total_input, total_output,
                            total_blocks, engine_hit, peer_hit, pool_hit])
    def get_response_results(self) -> list[FakeRequest]:
        return self.benchmark_emulator.get_response_results()


class DisaggBenchmarkRunner:
    def __init__(
        self,
        benchmark_config: BenchmarkConfig,
        p_scheduler_config: SchedulerConfig,
        d_scheduler_config: SchedulerConfig,
        p_platform_config: PlatformConfig,
        d_platform_config: PlatformConfig,
        router_config: RouterConfig,
        num_p_instance: int = 1,
        num_d_instance: int = 1,
        infer_time_predictor: Optional[InferTimePredictor] = None,
        use_real_token: bool = False,
        hierarchical_config_path: Optional[str] = None,
        enable_hierarchical: bool = False,
        hierarchical_output_dir: Optional[str] = None,
        storage_pool_capacity_gb: float = 1.0,
        enable_p2p: bool = True,
    ):
        assert p_scheduler_config.scenario == "disagg_prefill"
        assert d_scheduler_config.scenario == "disagg_decode"
        assert not p_scheduler_config.enable_real_time_request, (
            "The number of requests is unknown during PD disaggregation."
        )
        assert not d_scheduler_config.enable_real_time_request, (
            "The number of requests is unknown during PD disaggregation."
        )
        self.use_real_token = use_real_token

        gateway_request_queue = asyncio.Queue()
        gateway_response_queue = asyncio.Queue()
        self.router_queue = asyncio.Queue()

        self.benchmark_emulator = BenchmarkEmulator(
            benchmark_config,
            request_queue=gateway_request_queue,
            response_queue=gateway_response_queue,
            use_real_token_ids=use_real_token,
        )

        self.p_schedulers = [
            SGLangScheduleEmulator(
                scheduler_config=deepcopy(
                    p_scheduler_config
                ),  # The configuration might be modified during emulator startup.
                platform_config=p_platform_config,
                request_queue=asyncio.Queue(),
                response_queue=self.router_queue,
                time_predictor=infer_time_predictor,
                name=f"Prefill{i}",
                use_real_token=use_real_token,
            )
            for i in range(num_p_instance)
        ]

        self.d_schedulers = [
            SGLangScheduleEmulator(
                scheduler_config=deepcopy(d_scheduler_config),
                platform_config=d_platform_config,
                request_queue=asyncio.Queue(),
                response_queue=gateway_response_queue,
                time_predictor=infer_time_predictor,
                name=f"Decode{i}",
                use_real_token=use_real_token,
            )
            for i in range(num_d_instance)
        ]

        self.request_output_lengths: dict[int, int] = dict()
        self.request_output_ids: dict[int, list] = dict()

        # 策略状态缓存
        self.p_staging_queue: list[FakeRequest] = []
        self.d_staging_queue: list[FakeRequest] = []
        self.p_policy = self._create_policy(
            num_p_instance, router_config.p_policy, router_config,
            scheduler_config=p_scheduler_config, platform_config=p_platform_config,
        )
        self.d_policy = self._create_policy(
            num_d_instance, router_config.d_policy, router_config
        )

        # Hierarchical cache integration
        self.hierarchical_manager = None
        if hierarchical_config_path is not None:
            self._setup_hierarchical_cache(
                hierarchical_config_path, p_scheduler_config, p_platform_config
            )
        elif enable_hierarchical:
            p_ids = [f"P{i}" for i in range(num_p_instance)]
            d_ids = [f"D{i}" for i in range(num_d_instance)] if num_d_instance > 0 else None
            auto_config_path = build_hierarchical_config(
                scheduler_config=p_scheduler_config,
                platform_config=p_platform_config,
                p_instance_ids=p_ids,
                d_instance_ids=d_ids,
                output_dir=hierarchical_output_dir,
                storage_pool_capacity_gb=storage_pool_capacity_gb,
                enable_p2p=enable_p2p,
            )
            self._setup_hierarchical_cache(
                auto_config_path, p_scheduler_config, p_platform_config
            )

    def _setup_hierarchical_cache(self, config_path, scheduler_config, platform_config):
        try:
            import sys
            kvcm_so_dir = "/sgl-workspace/claude_workspace/tair-kvcache/bazel-bin/kv_cache_manager/optimizer/pybind"
            if kvcm_so_dir not in sys.path:
                sys.path.insert(0, kvcm_so_dir)
            import kvcm_py_optimizer as kvcm
        except ImportError:
            logger.warning("kvcm_py_optimizer not available, skipping hierarchical cache setup")
            return

        from schedule_simulator.schedule_emulator.hierarchical_cache_adapter import HierarchicalCacheAdapter

        loader = kvcm.HierarchicalReplayConfigLoader()
        if not loader.load(config_path):
            raise ValueError(f"Failed to load hierarchical config from {config_path}")

        self.hierarchical_manager = kvcm.HierarchicalReplayManager(loader.config())
        if not self.hierarchical_manager.Init():
            raise RuntimeError("HierarchicalReplayManager Init failed")

        config = loader.config()
        
        # Print Optimizer initialization parameters from config file
        print("=" * 60)
        print("Optimizer Initialization Parameters")
        print("=" * 60)
        try:
            import json
            with open(config_path) as f:
                cfg = json.load(f)
            
            for cluster in cfg.get("infer_clusters", []):
                model = cluster.get("model", {})
                print(f"  block_size: {model.get('block_size')}")
                print(f"  bytes_per_token: {model.get('bytes_per_token')}")
                bytes_per_block = model.get('block_size', 1) * model.get('bytes_per_token', 1)
                print(f"  bytes_per_block: {bytes_per_block}")
                print(f"  num_instances: {len(cluster.get('infer_ids', []))}")
                for i, tier in enumerate(cluster.get("tiers", [])):
                    print(f"  tier[{i}]: {tier.get('name')} = {tier.get('capacity')} GB")
                break  # Only print first cluster (all clusters share same model config)
        except Exception as e:
            print(f"  Failed to read config: {e}")
        print("=" * 60)
        
        infer_ids = []
        try:
            for cluster in config.infer_clusters():
                infer_ids.extend(cluster.infer_ids())
        except Exception:
            infer_ids = [f"P{i}" for i in range(len(self.p_schedulers))]
            infer_ids += [f"D{i}" for i in range(len(self.d_schedulers))]

        for i, sched in enumerate(self.p_schedulers):
            engine_id = infer_ids[i] if i < len(infer_ids) else f"P{i}"
            adapter = HierarchicalCacheAdapter(
                manager=self.hierarchical_manager,
                engine_instance_id=engine_id,
                platform_config=platform_config,
                kv_cache_space_per_token=sched.kv_cache_space_per_token,
                page_size=sched.scheduler_config.page_size,
                global_values=sched.global_values,
                prefetch_stop_policy=scheduler_config.hicache_storage_prefetch_policy,
                read_query_type=scheduler_config.hicache_read_query_type,
                enable_stats=scheduler_config.enable_stats,
            )
            sched.tree_cache = adapter
            sched.policy = SchedulePolicy(
                scheduler_config.schedule_policy, adapter, sched.time_predictor
            )

    def _create_policy(
        self, num_schedulers: int, policy_name: RoutingPolicy, config: RouterConfig,
        scheduler_config=None, platform_config=None,
    ) -> BasePolicy:
        policy_name = config.p_policy
        if policy_name == RoutingPolicy.RANDOM:
            return RandomPolicy(num_schedulers, config)
        elif policy_name == RoutingPolicy.ROUND_ROBIN:
            return RoundRobinPolicy(num_schedulers, config)
        elif policy_name == RoutingPolicy.POWER_OF_TWO:
            return PowerOfTwoPolicy(num_schedulers, config)
        elif policy_name == RoutingPolicy.CACHE_AWARE:
            return CacheAwarePolicy(num_schedulers, config, scheduler_config=scheduler_config, platform_config=platform_config)
        elif policy_name == RoutingPolicy.CACHE_AWARE_OLD:
            return CacheAwarePolicyOld(num_schedulers, config)
        elif policy_name == RoutingPolicy.DIRECT_CACHE_AWARE:
            return DirectCacheAwarePolicy(num_schedulers, config)
        else:
            raise ValueError(f"Unknown policy: {policy_name}")

    async def _dispatch_request(self, req: FakeRequest, is_decode: bool) -> int:
        """根据策略选择 worker 索引 (逻辑保持不变，仅做微调)"""
        if is_decode:
            schedulers, policy = self.d_schedulers, self.d_policy
        else:
            schedulers, policy = self.p_schedulers, self.p_policy

        num_schedulers = len(schedulers)
        if num_schedulers == 1:
            select_id = 0
        else:
            select_id = await policy.select_worker(req)

        logger.debug(
            f"policy:{policy.name} reqid:{req.id}  reqlen:{req.input_token_length} is_decode:{is_decode} select_id: {select_id}"
        )
        schedulers[select_id].request_queue.put_nowait(req)
        return select_id

    async def _enable_scheduler(self, sch: SGLangScheduleEmulator):
        # 执行一步：取消暂停 -> 运行一轮 (由 event_loop 内部逻辑消费) -> 重新暂停
        await sch.cancel_pause()
        await asyncio.sleep(0)
        await sch.set_pause()

    def _safe_get_request(self, queue: asyncio.Queue) -> FakeRequest:
        """辅助方法：安全地从队列获取请求，队列为空时返回 None"""
        try:
            return queue.get_nowait()
        except asyncio.QueueEmpty:
            return None

    async def time_routing_loop(self):
        all_schedulers = self.p_schedulers + self.d_schedulers
        await asyncio.sleep(0)  # 等待请求到来

        while True:
            """
            1. 请求入队：将新请求添加至缓冲区等待处理。
            2. 调度决策：对比 '最早请求到达时间'与 '最早忙Scheduler时间' 时间”。
            - 若队列为空或所有 Scheduler 均闲置：进入等待状态。
            - 若请求到达时间更早：优先分配请求；若无忙Scheduler，则立即启动首个请求。
            - 若 忙Scheduler 时钟更早：执行一步调度任务。
            3. 状态同步：将所有 Scheduler 的时间戳更新为当前最小时间基准。
            """
            req = self._safe_get_request(self.benchmark_emulator.request_queue)
            if req:
                self.request_output_lengths[req.id] = req.output_token_length
                self.request_output_ids[req.id] = req.output_ids
                req.output_token_length = 1
                if req.output_ids:
                    req.output_ids = req.output_ids[:1]
                self.p_staging_queue.append(req)
                continue

            req = self._safe_get_request(self.router_queue)
            if req:
                req.output_token_length = self.request_output_lengths.pop(req.id, 1)
                req.output_ids = self.request_output_ids.pop(req.id, 1)
                self.d_staging_queue.append(req)
                continue

            min_clock = 0
            min_scheduler = None

            # 找非空闲的第一个节点
            sorted_scheds = sorted(all_schedulers, key=lambda s: s.global_values.clock)
            for s in sorted_scheds:
                if not await s.is_idle():
                    min_scheduler = s
                    break

            # 找时间最早的请求
            next_req_time = float("inf")
            target_req_type = None
            if self.p_staging_queue:
                next_req_time = min(
                    next_req_time, self.p_staging_queue[0].create_time()
                )
                target_req_type = "p"
            if self.d_staging_queue:
                d_time = self.d_staging_queue[0].last_event_time
                if d_time < next_req_time:
                    next_req_time = d_time
                    target_req_type = "d"

            # 没有 忙scheduler 或 请求, 则 , 等待新请求或结束信号
            if not min_scheduler and not target_req_type:
                await asyncio.sleep(0)
                continue

            # 如果没有 scheduler 或者 请求才是最小的时间 , 先走请求 因为要更新负载或分配
            if (
                min_scheduler is None
                or next_req_time <= min_scheduler.global_values.clock
            ):
                min_clock = next_req_time
                # power_of_two 定时更新负载, cahce_aware 请求分配前更新负载
                await self.p_policy.update_load(self.p_schedulers, min_clock)
                await self.d_policy.update_load(self.d_schedulers, min_clock)
                if target_req_type == "p":
                    req = self.p_staging_queue.pop(0)
                    selected_idx = await self._dispatch_request(
                        req=req, is_decode=False
                    )
                    select_scheduler = self.p_schedulers[selected_idx]
                else:
                    req = self.d_staging_queue.pop(0)
                    if self.d_schedulers and req.output_token_length != 1:
                        selected_idx = await self._dispatch_request(
                            req=req, is_decode=True
                        )
                        select_scheduler = self.d_schedulers[selected_idx]
                    else:
                        # 如果没有d节点，直接输出
                        self.benchmark_emulator.response_queue.put_nowait(req)
                        await asyncio.sleep(0)
                        continue

                if await select_scheduler.is_idle(
                    min_clock
                ):  # 如果选择的scheduler闲，第一个请求要立即跑
                    select_scheduler.global_values.last_batch_run_time = (
                        select_scheduler.last_batch_run_time
                    ) = 0
                    await self._enable_scheduler(select_scheduler)
            else:
                # 没有请求 或 最小时间是scheduler
                min_clock = min_scheduler.global_values.clock
                # power_of_two 定时更新负载
                if self.p_policy.name == RoutingPolicy.POWER_OF_TWO:
                    await self.p_policy.update_load(self.p_schedulers, min_clock)
                if self.d_policy.name == RoutingPolicy.POWER_OF_TWO:
                    await self.d_policy.update_load(self.d_schedulers, min_clock)
                await self._enable_scheduler(min_scheduler)

            # 更新 时间
            for s in sorted_scheds:
                if s.global_values.clock < min_clock:
                    s.global_values.clock = min_clock
                else:
                    break

            await asyncio.sleep(0)

    async def async_run_benchmark_emulation(self):
        tasks: list[asyncio.Task] = []
        benchmark_task = asyncio.create_task(self.benchmark_emulator.benchmark())
        tasks.append(benchmark_task)

        for sched in self.p_schedulers + self.d_schedulers:
            tasks.append(asyncio.create_task(sched.event_loop()))

        routing_task = asyncio.create_task(self.time_routing_loop())
        tasks.append(routing_task)

        try:
            done, pending = await asyncio.wait(
                tasks, return_when=asyncio.FIRST_COMPLETED
            )
            if benchmark_task not in done:
                raise RuntimeError(
                    "The task terminated unexpectedly during the disaggregation benchmark. Only the benchmark task will proceed."
                )
            if benchmark_task.exception() is not None:
                raise benchmark_task.exception()

            metrics = benchmark_task.result()
            return metrics
        except Exception as e:
            logger.error(f"An error occurred during benchmark emulation: {e}")
            raise e
        finally:
            for task in tasks:
                task.cancel()

    def run_benchmark_emulation(
        self,
    ):
        start = time.time()
        metrics = asyncio.run(self.async_run_benchmark_emulation())
        end = time.time()
        metrics["time_cost"] = end - start
        return metrics

    # def get_iteration_stats(self) -> list[IterationStats]:
    #     # TODO: Is it necessary to get iteration statistics from multiple instances?
    #     pass

    def get_request_cache_fetch_stats(self) -> list[list[RequestCacheFetchStats]]:
        stats = []
        for sched in self.d_schedulers:
            # Get prefix cache stats from decoding scheduler only.
            stats.append(list(sched.tree_cache.get_request_fetch_stats().values()))
        return stats

    def get_iteration_cache_fetch_stats(self) -> list[list[IterationCacheFetchStats]]:
        stats = []
        for sched in self.d_schedulers:
            stats.append(list(sched.tree_cache.get_iteration_fetch_stats().values()))
        return stats

    def get_hierarchical_metrics(self) -> dict:
        if self.hierarchical_manager is None:
            return {}
        from schedule_simulator.schedule_emulator.hierarchical_cache_adapter import HierarchicalCacheAdapter
        total = {"total_blocks_queried": 0, "total_blocks_hit": 0,
                 "num_reads": 0, "num_writes": 0, "read_records": [], "write_records": []}
        for sched in self.p_schedulers + self.d_schedulers:
            if isinstance(sched.tree_cache, HierarchicalCacheAdapter):
                m = sched.tree_cache.get_hierarchical_metrics()
                total["total_blocks_queried"] += m.get("total_blocks_queried", 0)
                total["total_blocks_hit"] += m.get("total_blocks_hit", 0)
                total["num_reads"] += m["num_reads"]
                total["num_writes"] += m["num_writes"]
                total["read_records"].extend(sched.tree_cache.read_records)
                total["write_records"].extend(sched.tree_cache.write_records)
        total_engine_hit_blocks = sum(r.engine_hit for r in total["read_records"])
        total_peer_hit_blocks = sum(r.peer_hit for r in total["read_records"])
        total_pool_hit_blocks = sum(r.pool_hit for r in total["read_records"])
        tb = max(total["total_blocks_queried"], 1)
        total["total_engine_hit_blocks"] = total_engine_hit_blocks
        total["total_peer_hit_blocks"] = total_peer_hit_blocks
        total["total_pool_hit_blocks"] = total_pool_hit_blocks
        total["engine_hit_block_ratio"] = total_engine_hit_blocks / tb
        total["peer_hit_block_ratio"] = total_peer_hit_blocks / tb
        total["pool_hit_block_ratio"] = total_pool_hit_blocks / tb
        total["block_hit_ratio"] = total["total_blocks_hit"] / tb
        return total

    def analyze_hierarchical_results(self):
        if self.hierarchical_manager is not None:
            self.hierarchical_manager.AnalyzeResults()


    def export_results(self, output_dir: str, metrics: dict = None):
        """Export simulation results to files: summary JSON, per-request CSV, per-iteration CSV."""
        import json as _json
        os.makedirs(output_dir, exist_ok=True)
        results = self.get_response_results()

        if metrics is None:
            from schedule_simulator.schedule_emulator.utils import calc_metrics
            metrics = calc_metrics(results)

        # 1. Summary JSON (metrics + hierarchical)
        summary = dict(metrics)
        hier = self.get_hierarchical_metrics()
        if hier:
            for k, v in hier.items():
                if k not in ("read_records", "write_records"):
                    summary["hierarchical_" + k] = v
        with open(os.path.join(output_dir, "simulation_summary.json"), "w") as f:
            _json.dump(summary, f, indent=2)

        # 2. Per-request CSV
        hit_map = {}
        if hier and "read_records" in hier:
            for r in hier["read_records"]:
                hit_map[r.req_id] = (r.engine_hit, r.peer_hit, r.pool_hit, r.num_blocks)

        with open(os.path.join(output_dir, "per_request.csv"), "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["req_id", "input_length", "output_length",
                        "ttft_ms", "e2e_latency_ms", "queue_wait_ms",
                        "cache_reused_tokens",
                        "engine_hit", "peer_hit", "pool_hit", "num_blocks"])
            for req in sorted(results, key=lambda r: r.id):
                ttft = req.gen_token_latencies[0] * 1000 if req.gen_token_latencies else 0
                e2e = sum(req.gen_token_latencies) * 1000
                qw = (req.queue_time_end - req.queue_time_start) * 1000 if req.queue_time_start >= 0 else 0
                eh, ph, sh, nb = hit_map.get(req.id, (0, 0, 0, 0))
                w.writerow([req.id, req.input_token_length, req.output_token_length,
                            round(ttft, 3), round(e2e, 3), round(qw, 3),
                            req.final_reused_tokens, eh, ph, sh, nb])

        # 3. Per-iteration CSV
        with open(os.path.join(output_dir, "per_iteration.csv"), "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["pod", "iteration", "timestamp", "iter_latency_ms",
                        "num_ctx_requests", "num_ctx_tokens", "num_gen_requests",
                        "num_waiting"])
            for i, sched in enumerate(self.p_schedulers + self.d_schedulers):
                pod_name = "P%d" % i if i < len(self.p_schedulers) else "D%d" % (i - len(self.p_schedulers))
                for s in sched.get_iteration_stats():
                    w.writerow([pod_name, s.iter, round(s.timestamp, 6),
                                round(s.iter_latency_ms, 3),
                                s.num_context_requests, s.num_ctx_tokens,
                                s.num_gen_requests, s.num_waiting_requests])


        # 4. Per-pod stats CSV
        from schedule_simulator.schedule_emulator.hierarchical_cache_adapter import HierarchicalCacheAdapter
        with open(os.path.join(output_dir, "per_pod_stats.csv"), "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["pod", "total_requests", "total_input_tokens", "total_output_tokens",
                        "total_blocks", "total_engine_hit_blocks", "total_peer_hit_blocks",
                        "total_pool_hit_blocks"])
            for i, sched in enumerate(self.p_schedulers + self.d_schedulers):
                pod_name = "P%d" % i if i < len(self.p_schedulers) else "D%d" % (i - len(self.p_schedulers))
                completed = sched.completed_requests
                total_reqs = len(completed)
                total_input = sum(r.input_token_length for r in completed)
                total_output = sum(r.output_token_length for r in completed)
                page_size = sched.scheduler_config.page_size or 1
                total_blocks = sum(max(r.input_token_length // page_size, 1) for r in completed)
                engine_hit = 0
                peer_hit = 0
                pool_hit = 0
                if isinstance(sched.tree_cache, HierarchicalCacheAdapter):
                    engine_hit = sched.tree_cache.total_engine_hit_blocks
                    peer_hit = sched.tree_cache.total_peer_hit_blocks
                    pool_hit = sched.tree_cache.total_pool_hit_blocks
                w.writerow([pod_name, total_reqs, total_input, total_output,
                            total_blocks, engine_hit, peer_hit, pool_hit])
    def get_response_results(self) -> list[FakeRequest]:
        return self.benchmark_emulator.get_response_results()


def run_benchmark_emulation(
    benchmark_config: BenchmarkConfig,
    scheduler_config: SchedulerConfig,
    platform_config: PlatformConfig,
    infer_time_predictor: Optional[InferTimePredictor] = None,
):
    return BenchmarkRunner(
        benchmark_config,
        scheduler_config,
        platform_config,
        infer_time_predictor,
    ).run_benchmark_emulation()
