#!/usr/bin/env python3
"""
Optimizer Standalone Simulation - bypass the scheduler/tree_cache layer.

Directly calls the C++ optimizer (HierarchicalReplayManager) with the same dataset
to get theoretical hit rates. Useful for verifying whether the simulator introduces
errors vs the optimizer's ground truth.

Usage:
  python scripts/run_optimizer_standalone.py \
    --dataset /path/to/h21_32_256k_full.jsonl \
    --num-prompts 1000 \
    --page-size 256 \
    --num-instances 5 \
    --hbm-capacity 80 \
    --kv-bytes-per-token 46080 \
    --routing round_robin \
    --output-dir results/standalone/
"""
import argparse, json, os, sys, time
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


def main():
    p = argparse.ArgumentParser(description="Optimizer Standalone Hit Rate Simulation")
    p.add_argument("--dataset", type=str, required=True, help="Input JSONL dataset path")
    p.add_argument("--num-prompts", type=int, default=None, help="Max requests to process (default: all)")
    p.add_argument("--page-size", type=int, default=256, help="Engine block/page size in tokens")
    p.add_argument("--data-block-size", type=int, default=None,
                   help="Block size in dataset (e.g., 256). If different from --page-size, converts block_ids.")
    p.add_argument("--num-instances", type=int, default=5, help="Number of engine instances")
    p.add_argument("--model", type=str, default="Qwen2.5-3B")
    p.add_argument("--device", type=str, default="H20")
    p.add_argument("--hbm-capacity", type=float, default=None, help="HBM capacity in GB")
    p.add_argument("--mem-capacity", type=float, default=None, help="DRAM capacity in GB")
    p.add_argument("--kv-bytes-per-token", type=int, default=None, help="KV cache bytes per token")
    p.add_argument("--max-num-tokens", type=int, default=None, help="L1 capacity in tokens")
    p.add_argument("--routing", type=str, default="round_robin",
                   choices=["round_robin", "random", "sequential"])
    p.add_argument("--enable-p2p", action="store_true", default=True)
    p.add_argument("--no-p2p", action="store_true")
    p.add_argument("--write-policy", type=str, default="write_through")
    p.add_argument("--pool-capacity", type=float, default=2.0, help="Storage pool capacity in GB")
    p.add_argument("--output-dir", type=str, default="./standalone_results")
    p.add_argument("--quiet", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    # --- Import optimizer bindings ---
    kvcm_so_dir = "/sgl-workspace/claude_workspace/tair-kvcache/bazel-bin/kv_cache_manager/optimizer/pybind"
    if kvcm_so_dir not in sys.path:
        sys.path.insert(0, kvcm_so_dir)
    try:
        import kvcm_py_optimizer as kvcm
    except ImportError:
        print("[ERROR] Cannot import kvcm_py_optimizer. Build the C++ optimizer first.")
        sys.exit(1)

    from schedule_simulator.schedule_emulator.types import SchedulerConfig, PlatformConfig
    from schedule_simulator.schedule_emulator.hierarchical_config_builder import build_hierarchical_config

    # --- Build optimizer config (same as full simulator) ---
    sc = SchedulerConfig(
        model=args.model,
        scenario="disagg_prefill",
        page_size=args.page_size,
        kv_cache_space_per_token=args.kv_bytes_per_token,
        max_num_tokens=args.max_num_tokens,
        hicache_write_policy=args.write_policy,
    )
    pc = PlatformConfig(
        device=args.device,
        hbm_capacity_gb=args.hbm_capacity,
        memory_capacity_gb=args.mem_capacity,
    )

    instance_ids = [f"P{i}" for i in range(args.num_instances)]
    os.makedirs(args.output_dir, exist_ok=True)

    config_path = build_hierarchical_config(
        scheduler_config=sc,
        platform_config=pc,
        p_instance_ids=instance_ids,
        output_dir=args.output_dir,
        storage_pool_capacity_gb=args.pool_capacity,
        enable_p2p=args.enable_p2p and not args.no_p2p,
    )

    if not args.quiet:
        print(f"[INFO] Optimizer config: {config_path}")
        with open(config_path) as f:
            cfg = json.load(f)
        for cluster in cfg.get("infer_clusters", []):
            model_cfg = cluster.get("model", {})
            print(f"  block_size={model_cfg.get('block_size')}, bytes_per_token={model_cfg.get('bytes_per_token')}")
            print(f"  instances={len(cluster.get('infer_ids', []))}")
            for t in cluster.get("tiers", []):
                print(f"  tier[{t['name']}] = {t['capacity']} GB")

    # --- Init optimizer ---
    loader = kvcm.HierarchicalReplayConfigLoader()
    if not loader.load(config_path):
        print(f"[ERROR] Failed to load config: {config_path}")
        sys.exit(1)
    mgr = kvcm.HierarchicalReplayManager(loader.config())
    if not mgr.Init():
        print("[ERROR] HierarchicalReplayManager Init failed")
        sys.exit(1)

    # --- Optional block_id conversion ---
    convert_fn = None
    if args.data_block_size and args.data_block_size != args.page_size:
        from schedule_simulator.schedule_emulator.block_id_converter import convert_block_ids
        convert_fn = lambda bids: convert_block_ids(bids, args.data_block_size, args.page_size)
        if not args.quiet:
            print(f"[INFO] Block ID conversion: {args.data_block_size} -> {args.page_size} (ratio={args.page_size // args.data_block_size})")

    # --- Load dataset ---
    if not args.quiet:
        print(f"[INFO] Loading dataset: {args.dataset}")

    requests = []
    with open(args.dataset) as f:
        for line in f:
            req = json.loads(line)
            requests.append(req)
            if args.num_prompts and len(requests) >= args.num_prompts:
                break

    # Sort by timestamp (same order as real simulation)
    requests.sort(key=lambda r: r.get("timestamp", 0))

    if not args.quiet:
        print(f"[INFO] Loaded {len(requests)} requests, routing={args.routing}")
        print("=" * 60)

    # --- Routing assignment ---
    np.random.seed(args.seed)
    if args.routing == "round_robin":
        assignments = [instance_ids[i % args.num_instances] for i in range(len(requests))]
    elif args.routing == "random":
        assignments = [instance_ids[np.random.randint(0, args.num_instances)] for _ in requests]
    elif args.routing == "sequential":
        # All to first instance (for debugging single-instance behavior)
        assignments = [instance_ids[0]] * len(requests)

    # --- Run simulation: GetCacheLocation -> WriteCache for each request ---
    t0 = time.time()
    total_engine_hit = 0
    total_peer_hit = 0
    total_pool_hit = 0
    total_blocks = 0
    per_req_results = []

    for idx, (req, pod) in enumerate(zip(requests, assignments)):
        block_ids = req.get("block_ids", [])
        input_len = req.get("input_length", len(block_ids) * args.page_size)
        ts_ns = int(req.get("timestamp", idx) * 1e6)  # ms -> ns

        # Convert block_ids if needed
        if convert_fn and block_ids:
            block_ids = convert_fn(block_ids)

        # Truncate to full blocks (same as HierarchicalCacheAdapter)
        max_full_blocks = input_len // args.page_size
        if max_full_blocks == 0 and block_ids:
            max_full_blocks = 1
        full_block_ids = block_ids[:max_full_blocks]

        # Read (GetCacheLocation)
        res = mgr.GetCacheLocation(pod, f"r{idx}", ts_ns, full_block_ids, input_len)
        eh = res.engine_hit_length
        ph = res.peer_hit_length
        sh = res.storage_pool_hit_length

        total_engine_hit += eh
        total_peer_hit += ph
        total_pool_hit += sh
        total_blocks += len(full_block_ids)

        per_req_results.append({
            "req_id": idx,
            "pod": pod,
            "input_length": input_len,
            "num_blocks": len(full_block_ids),
            "engine_hit": eh,
            "peer_hit": ph,
            "pool_hit": sh,
            "total_hit": eh + ph + sh,
            "hit_ratio": (eh + ph + sh) / max(len(full_block_ids), 1),
        })

        # Write (WriteCache)
        mgr.WriteCache(pod, f"w{idx}", ts_ns + 1, full_block_ids)

        if not args.quiet and (idx + 1) % 10000 == 0:
            elapsed = time.time() - t0
            current_hit = (total_engine_hit + total_peer_hit + total_pool_hit)
            print(f"  [{idx+1}/{len(requests)}] hit_ratio={current_hit/max(total_blocks,1)*100:.1f}% ({elapsed:.1f}s)")

    elapsed = time.time() - t0
    total_hit = total_engine_hit + total_peer_hit + total_pool_hit
    hit_ratio = total_hit / max(total_blocks, 1)

    # --- Summary ---
    summary = {
        "mode": "optimizer_standalone",
        "dataset": args.dataset,
        "num_requests": len(requests),
        "num_instances": args.num_instances,
        "page_size": args.page_size,
        "data_block_size": args.data_block_size,
        "routing": args.routing,
        "enable_p2p": args.enable_p2p and not args.no_p2p,
        "total_blocks": total_blocks,
        "total_engine_hit": total_engine_hit,
        "total_peer_hit": total_peer_hit,
        "total_pool_hit": total_pool_hit,
        "total_hit": total_hit,
        "block_hit_ratio": hit_ratio,
        "engine_hit_ratio": total_engine_hit / max(total_blocks, 1),
        "peer_hit_ratio": total_peer_hit / max(total_blocks, 1),
        "pool_hit_ratio": total_pool_hit / max(total_blocks, 1),
        "elapsed_seconds": elapsed,
    }

    # --- Per-pod stats ---
    pod_stats = {}
    for r in per_req_results:
        pod = r["pod"]
        if pod not in pod_stats:
            pod_stats[pod] = {"requests": 0, "blocks": 0, "engine_hit": 0, "peer_hit": 0, "pool_hit": 0}
        pod_stats[pod]["requests"] += 1
        pod_stats[pod]["blocks"] += r["num_blocks"]
        pod_stats[pod]["engine_hit"] += r["engine_hit"]
        pod_stats[pod]["peer_hit"] += r["peer_hit"]
        pod_stats[pod]["pool_hit"] += r["pool_hit"]

    summary["per_pod"] = pod_stats

    # --- Output ---
    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, "standalone_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    # Per-request CSV
    import csv
    with open(os.path.join(args.output_dir, "standalone_per_request.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["req_id", "pod", "input_length", "num_blocks", "engine_hit", "peer_hit", "pool_hit", "total_hit", "hit_ratio"])
        for r in per_req_results:
            w.writerow([r["req_id"], r["pod"], r["input_length"], r["num_blocks"],
                       r["engine_hit"], r["peer_hit"], r["pool_hit"], r["total_hit"],
                       round(r["hit_ratio"], 4)])

    print("\n" + "=" * 60)
    print("Optimizer Standalone Results")
    print("=" * 60)
    print(f"  Requests:     {len(requests)}")
    print(f"  Instances:    {args.num_instances} ({args.routing})")
    print(f"  Total blocks: {total_blocks}")
    print(f"  Engine hit:   {total_engine_hit} ({total_engine_hit/max(total_blocks,1)*100:.2f}%)")
    print(f"  Peer hit:     {total_peer_hit} ({total_peer_hit/max(total_blocks,1)*100:.2f}%)")
    print(f"  Pool hit:     {total_pool_hit} ({total_pool_hit/max(total_blocks,1)*100:.2f}%)")
    print(f"  Total hit:    {total_hit} ({hit_ratio*100:.2f}%)")
    print(f"  Time:         {elapsed:.1f}s")
    print(f"  Output:       {args.output_dir}")
    print("=" * 60)

    # Per-pod breakdown
    print("\nPer-Pod Breakdown:")
    print(f"  {'Pod':<6} {'Reqs':>6} {'Blocks':>8} {'EngHit':>8} {'PeerHit':>8} {'PoolHit':>8} {'HitRate':>8}")
    for pod in sorted(pod_stats.keys()):
        s = pod_stats[pod]
        hr = (s["engine_hit"] + s["peer_hit"] + s["pool_hit"]) / max(s["blocks"], 1) * 100
        print(f"  {pod:<6} {s['requests']:>6} {s['blocks']:>8} {s['engine_hit']:>8} {s['peer_hit']:>8} {s['pool_hit']:>8} {hr:>7.1f}%")


if __name__ == "__main__":
    main()
