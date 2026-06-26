#!/usr/bin/env python3
"""统计仿真输入文件中各 Pod 的实际负载分布。

用法:
  python3 scripts/analyze_pod_load.py --input /path/to/sim.jsonl [--top 20]

输出:
  - 各 pod 的请求数、block 总数、平均 block 数
  - 全局汇总统计
  - 按请求数排序的 Top-N 列表
"""
import json, argparse, time, statistics, csv
from collections import defaultdict


def main():
    p = argparse.ArgumentParser(description="统计各 Pod 的请求数和 block 数")
    p.add_argument("--input", type=str, required=True, help="仿真输入 JSONL 文件路径")
    p.add_argument("--top", type=int, default=20, help="显示 Top-N pods (默认 20)")
    p.add_argument("--output", type=str, default=None, help="输出 CSV 路径 (可选)")
    args = p.parse_args()

    t0 = time.time()

    pod_stats = defaultdict(lambda: {"requests": 0, "blocks": 0})
    total_requests = 0
    no_pod_count = 0

    with open(args.input, "r") as f:
        for line in f:
            d = json.loads(line)
            total_requests += 1

            instance_id = d.get("instance_id")
            if not instance_id:
                pods = d.get("pods", [])
                instance_id = pods[0] if pods else None

            if not instance_id:
                no_pod_count += 1
                continue

            block_ids = d.get("block_ids") or d.get("input_block_hash_ids") or []
            num_blocks = len(block_ids)

            pod_stats[instance_id]["requests"] += 1
            pod_stats[instance_id]["blocks"] += num_blocks

    elapsed = time.time() - t0

    num_pods = len(pod_stats)
    all_reqs = [v["requests"] for v in pod_stats.values()]
    all_blocks = [v["blocks"] for v in pod_stats.values()]

    print("=" * 70)
    print(f"文件: {args.input}")
    print(f"总请求数: {total_requests:,}")
    print(f"有效 Pod 数: {num_pods}")
    if no_pod_count:
        print(f"无 Pod 信息的请求: {no_pod_count}")
    print(f"解析耗时: {elapsed:.1f}s")
    print("=" * 70)

    if not all_reqs:
        print("无数据")
        return

    print(f"\n{'指标':<20} {'请求数':>12} {'Block数':>12}")
    print("-" * 50)
    print(f"{'总计':<20} {sum(all_reqs):>12,} {sum(all_blocks):>12,}")
    print(f"{'每Pod平均':<20} {statistics.mean(all_reqs):>12.1f} {statistics.mean(all_blocks):>12.1f}")
    print(f"{'每Pod中位数':<18} {statistics.median(all_reqs):>12.1f} {statistics.median(all_blocks):>12.1f}")
    print(f"{'每Pod最小':<20} {min(all_reqs):>12,} {min(all_blocks):>12,}")
    print(f"{'每Pod最大':<20} {max(all_reqs):>12,} {max(all_blocks):>12,}")
    print(f"{'标准差':<20} {statistics.stdev(all_reqs):>12.1f} {statistics.stdev(all_blocks):>12.1f}")

    avg_blocks_per_req = sum(all_blocks) / sum(all_reqs) if sum(all_reqs) > 0 else 0
    print(f"\n每请求平均 block 数: {avg_blocks_per_req:.1f}")

    sorted_pods = sorted(pod_stats.items(), key=lambda x: x[1]["requests"], reverse=True)
    print(f"\n--- Top {args.top} Pods (按请求数) ---")
    print(f"{'#':<4} {'Pod':<40} {'请求数':>8} {'Block数':>10} {'平均Block':>10}")
    print("-" * 75)
    for i, (pod, stats) in enumerate(sorted_pods[:args.top]):
        avg_b = stats["blocks"] / stats["requests"] if stats["requests"] > 0 else 0
        print(f"{i+1:<4} {pod:<40} {stats['requests']:>8,} {stats['blocks']:>10,} {avg_b:>10.1f}")

    if num_pods > args.top:
        print(f"\n--- Bottom 5 Pods (按请求数) ---")
        print(f"{'#':<4} {'Pod':<40} {'请求数':>8} {'Block数':>10} {'平均Block':>10}")
        print("-" * 75)
        for i, (pod, stats) in enumerate(sorted_pods[-5:]):
            avg_b = stats["blocks"] / stats["requests"] if stats["requests"] > 0 else 0
            print(f"{num_pods-4+i:<4} {pod:<40} {stats['requests']:>8,} {stats['blocks']:>10,} {avg_b:>10.1f}")

    if args.output:
        with open(args.output, "w", newline="") as csvf:
            writer = csv.writer(csvf)
            writer.writerow(["pod", "requests", "blocks", "avg_blocks_per_req"])
            for pod, stats in sorted_pods:
                avg_b = stats["blocks"] / stats["requests"] if stats["requests"] > 0 else 0
                writer.writerow([pod, stats["requests"], stats["blocks"], f"{avg_b:.1f}"])
        print(f"\nCSV 已保存: {args.output}")


if __name__ == "__main__":
    main()
