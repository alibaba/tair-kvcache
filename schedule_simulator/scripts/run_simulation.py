#!/usr/bin/env python3
"""
One-click simulation script. See --help for full options.

Examples:
  # Random requests, request-level, 5 P instances
  python scripts/run_simulation.py --num-prompts 200 --request-level --ms-per-token 0.1 --num-p-instances 5

  # Enriched data with hierarchical cache
  python scripts/run_simulation.py --dataset input.jsonl --request-level --predictor-pkl model.pkl \
    --num-p-instances 10 --enable-hierarchical --output-dir results/

  # Iteration-level (default model, needs kunlun_commons)
  python scripts/run_simulation.py --num-prompts 100 --model Qwen2.5-3B --num-p-instances 5
"""
import argparse, json, os, sys, time, random
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

def main():
    p = argparse.ArgumentParser(description="LLM Multi-Instance Prefill Simulator")
    p.add_argument("--dataset", type=str, default=None)
    p.add_argument("--num-prompts", type=int, default=100)
    p.add_argument("--min-input", type=int, default=500)
    p.add_argument("--max-input", type=int, default=2000)
    p.add_argument("--min-output", type=int, default=1)
    p.add_argument("--max-output", type=int, default=2)
    p.add_argument("--request-rate", type=float, default=float("inf"))
    p.add_argument("--model", type=str, default="Qwen2.5-3B")
    p.add_argument("--device", type=str, default="H20")
    p.add_argument("--disk-bw", type=float, default=2.0)
    p.add_argument("--mem-bw", type=float, default=16.0)
    p.add_argument("--peer-bw", type=float, default=10.0)
    p.add_argument("--mem-capacity", type=float, default=None)
    p.add_argument("--hbm-capacity", type=float, default=None, help="HBM capacity in GB (overrides hardware default)")
    p.add_argument("--num-p-instances", type=int, default=1)
    p.add_argument("--routing", type=str, default="round_robin",
                   choices=["random","round_robin","power_of_two","cache_aware","cache_aware_old","direct_cache_aware"])
    p.add_argument("--request-level", action="store_true")
    p.add_argument("--chunk-size", type=int, default=8192)
    p.add_argument("--kv-bytes-per-token", type=int, default=None,
                   help="KV cache bytes per token (skip ModelInfo derivation)")
    p.add_argument("--max-num-tokens", type=int, default=None,
                   help="L1 device KV cache capacity in tokens (skip HBM derivation)")
    p.add_argument("--l2-cache-tokens", type=int, default=None,
                   help="L2 host cache capacity in tokens (default: 2x L1)")
    p.add_argument("--predictor-pkl", type=str, default=None)
    p.add_argument("--predictor-qwen-dir", type=str, default=None,
                   help="Path to qwen latency predictor models directory")
    p.add_argument("--ms-per-token", type=float, default=None)
    p.add_argument("--enable-hierarchical", action="store_true")
    p.add_argument("--page-size", type=int, default=None, help="Page/block size for cache (default: 1)")
    p.add_argument("--data-block-size", type=int, default=None,
                   help="Block size in dataset block_ids (e.g., 256). If different from --page-size, block_ids are converted.")
    p.add_argument("--enable-p2p", action="store_true", default=True)
    p.add_argument("--no-p2p", action="store_true")
    p.add_argument("--write-policy", type=str, default="write_through")
    p.add_argument("--pool-capacity", type=float, default=2.0)
    p.add_argument("--hierarchical-config", type=str, default=None)
    p.add_argument("--output-dir", type=str, default="./sim_results")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--enable-stats", action="store_true")
    # Probabilistic routing (topk) parameters
    p.add_argument("--topk-routing", action="store_true", help="Enable probabilistic topK routing")
    p.add_argument("--lmax", type=int, default=40, help="Max load baseline for normalization (Lmax)")
    p.add_argument("--weight-prefix", type=float, default=30.0, help="Prefix hit weight (wp) in scoring formula")
    p.add_argument("--weight-load", type=float, default=10.0, help="Load balance weight (wl) in scoring formula")
    p.add_argument("--quiet", action="store_true")
    args = p.parse_args()

    random.seed(args.seed); np.random.seed(args.seed)
    from schedule_simulator.schedule_emulator.types import BenchmarkConfig, SchedulerConfig, PlatformConfig, RouterConfig, RoutingPolicy
    from schedule_simulator.schedule_emulator.run import DisaggBenchmarkRunner, BenchmarkRunner

    bc = BenchmarkConfig(
        dataset_path=args.dataset, num_prompts=args.num_prompts,
        min_input_length=args.min_input if not args.dataset else None,
        max_input_length=args.max_input if not args.dataset else None,
        min_output_length=args.min_output if not args.dataset else None,
        max_output_length=args.max_output if not args.dataset else None,
        request_rate=args.request_rate, disable_tqdm=args.quiet,
        data_block_size=args.data_block_size,
    )
    sc = SchedulerConfig(
        model=args.model,
        scenario="disagg_prefill" if args.num_p_instances > 1 else "normal",
        request_level_scheduling=args.request_level,
        chunked_prefill_size=args.chunk_size,
        hicache_storage_backend="hf3fs" if args.enable_hierarchical else None,
        hicache_write_policy=args.write_policy,
        enable_stats=args.enable_stats,
        kv_cache_space_per_token=args.kv_bytes_per_token,
        max_num_tokens=args.max_num_tokens,
        l2_cache_num_tokens=args.l2_cache_tokens,
        page_size=args.page_size,
    )
    pc = PlatformConfig(
        device=args.device, disk_read_bandwidth_gb=args.disk_bw,
        memory_read_bandwidth_gb=args.mem_bw, peer_read_bandwidth_gb=args.peer_bw,
        memory_capacity_gb=args.mem_capacity, hbm_capacity_gb=args.hbm_capacity,
    )

    predictor = None
    if args.request_level:
        if args.predictor_qwen_dir:
            sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
            from qwen_latency_predictor import RequestLevelTimePredictor as QwenPredictor
            predictor = QwenPredictor(args.predictor_qwen_dir)
            if not args.quiet:
                print(f"[INFO] Loaded Qwen latency predictor from {args.predictor_qwen_dir}")
        else:
            from schedule_simulator.infer_time_predictor import RequestLevelTimePredictor
            if args.predictor_pkl:
                predictor = RequestLevelTimePredictor(lookup_table_path=args.predictor_pkl)
            elif args.ms_per_token:
                predictor = RequestLevelTimePredictor(constant_ms_per_token=args.ms_per_token)
            else:
                predictor = RequestLevelTimePredictor(constant_ms_per_token=0.1)
                print("[WARN] No predictor specified, using default 0.1 ms/token")

    policy_map = {"random": RoutingPolicy.RANDOM, "round_robin": RoutingPolicy.ROUND_ROBIN,
                  "power_of_two": RoutingPolicy.POWER_OF_TWO, "cache_aware": RoutingPolicy.CACHE_AWARE, "cache_aware_old": RoutingPolicy.CACHE_AWARE_OLD, "direct_cache_aware": RoutingPolicy.DIRECT_CACHE_AWARE}

    if not args.quiet:
        print("=" * 60)
        print("Config: %d requests, %d P nodes, %s, %s" % (
            args.num_prompts, args.num_p_instances, args.routing,
            "request-level" if args.request_level else "iteration-level"))
        print("=" * 60)

    t0 = time.time()
    if args.num_p_instances > 1:
        runner = DisaggBenchmarkRunner(
            benchmark_config=bc, p_scheduler_config=sc,
            d_scheduler_config=SchedulerConfig(args.model, scenario="disagg_decode"),
            p_platform_config=pc, d_platform_config=PlatformConfig(device=args.device),
            router_config=RouterConfig(p_policy=policy_map[args.routing],
                d_policy=RoutingPolicy.ROUND_ROBIN, worker_startup_check_interval=0.01,
                topk_routing=args.topk_routing, lmax=args.lmax,
                weight_prefix=args.weight_prefix, weight_load=args.weight_load),
            num_p_instance=args.num_p_instances, num_d_instance=0,
            infer_time_predictor=predictor,
            enable_hierarchical=args.enable_hierarchical,
            enable_p2p=args.enable_p2p and not args.no_p2p,
            hierarchical_config_path=args.hierarchical_config,
            hierarchical_output_dir=os.path.join(args.output_dir, "optimizer") if args.enable_hierarchical else None,
            storage_pool_capacity_gb=max(args.pool_capacity, 0.0001),
        )
    else:
        sc.scenario = "normal"
        runner = BenchmarkRunner(benchmark_config=bc, scheduler_config=sc,
                                  platform_config=pc, infer_time_predictor=predictor)

    m = runner.run_benchmark_emulation()
    os.makedirs(args.output_dir, exist_ok=True)
    if hasattr(runner, "export_results"):
        runner.export_results(args.output_dir, m)
    else:
        with open(os.path.join(args.output_dir, "simulation_summary.json"), "w") as f:
            json.dump(m, f, indent=2)

    if not args.quiet:
        print("\nCompleted: %d  TTFT: mean=%.0fms p99=%.0fms  Throughput: %.1f req/s  (%.1fs)" % (
            m["completed"], m["mean_ttft_ms"], m["p99_ttft_ms"], m["request_throughput"], time.time()-t0))
        if hasattr(runner, "get_hierarchical_metrics"):
            h = runner.get_hierarchical_metrics()
            if h and h.get("total_blocks_queried", 0) > 0:
                print("Cache: engine=%d peer=%d pool=%d hit=%.1f%%" % (
                    h["total_engine_hit_blocks"], h["total_peer_hit_blocks"],
                    h["total_pool_hit_blocks"], h.get("block_hit_ratio", 0)*100))
        print("Output: %s" % args.output_dir)

if __name__ == "__main__":
    main()
