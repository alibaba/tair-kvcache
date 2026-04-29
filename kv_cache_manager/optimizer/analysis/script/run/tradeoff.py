#!/usr/bin/env python3
"""
Pareto 曲线分析（统一入口）

给定 1 个策略 → 单策略 Pareto 曲线（每 instance 一条线）
给定 N 个策略 → 多策略对比子图（每 instance 一个子图，每策略一条线）

Applicability:
  This analysis is designed for NON-TIERED mode. In tiered mode
  (hierarchical_eviction_enabled=true), the capacity sweep only modifies
  quota_capacity, which does not affect per-tier eviction decisions. Each
  tier's eviction is governed by its independent storages[i].capacity.

用法:
  # 单策略（默认使用配置文件中的策略）
  python run/tradeoff.py -c config.json
  python run/tradeoff.py -c config.json --skip-run --csv-output-dir /path/to/csvs

  # 多策略对比
  python run/tradeoff.py -c config.json --eviction-policies lru leaf_aware_lru random_lru
  python run/tradeoff.py -c config.json --eviction-policies lru leaf_aware_lru --skip-run
"""

import argparse
import os
import sys
from collections import defaultdict

from kv_cache_manager.optimizer.pybind import kvcm_py_optimizer

from utils.optimizer_runner import init_kvcm_logger, warmup_pass, run_experiments_parallel, extract_bytes_per_block_map, extract_config_quota_gb_map
from utils.csv_loader import generate_capacity_list, load_results_from_csv_dir
from utils.plot_utils import plot_single_policy_curves, plot_multi_policy_subplots
from plot.hit_rate_plot import plot_multi_instance_analysis


# ============================================================================
# 结果整理
# ============================================================================

def _group_by_policy(raw_results):
    """将 run_experiments_parallel 原始结果按策略分组"""
    by_policy = defaultdict(list)
    for r in raw_results:
        if not r["success"]:
            continue
        pol = r.get("policy") or "default_policy"
        by_policy[pol].append({
            "capacity": r["capacity"],
            "instances": r["instances"],
        })
    for pol in by_policy:
        by_policy[pol].sort(key=lambda x: x["capacity"])
    return dict(by_policy)


# ============================================================================
# 表格打印
# ============================================================================

def _print_single_policy_table(results, bytes_per_block_map):
    """单策略命中率表格"""
    if not results:
        return
    rep_bpb = next(iter(bytes_per_block_map.values()))
    iids = sorted(results[0]["instances"].keys())
    print("{:>12} | {:<20} | {:>10} | {:>10} | {:>10}".format(
        "Cap (GB)", "Instance", "Total", "Local", "Remote"))
    print("-" * 70)
    for r in results:
        for iid in iids:
            m = r["instances"].get(iid)
            if m:
                print("{:12.3f} | {:<20} | {:10.6f} | {:10.6f} | {:10.6f}".format(
                    r["capacity"] * rep_bpb / (1024 ** 3), iid, m["total"], m["local"], m["remote"]))


def _print_multi_policy_table(results_by_policy, policies, bytes_per_block_map):
    """多策略对比表格"""
    rep_bpb = next(iter(bytes_per_block_map.values()))
    all_caps = sorted({r["capacity"] for rs in results_by_policy.values() for r in rs})
    all_iids = sorted({iid for rs in results_by_policy.values() for r in rs for iid in r["instances"]})

    cap_lookup = defaultdict(dict)
    for pol, rs in results_by_policy.items():
        for r in rs:
            cap_lookup[(r["capacity"], pol)] = r["instances"]

    for iid in all_iids:
        print("\n" + "=" * 80)
        print("Instance: {}".format(iid))
        print("=" * 80)
        header = "{:>12} |".format("Cap (GB)")
        for pol in policies:
            header += " {:<22} |".format(pol)
        print(header)
        print("-" * 80)

        for cap in all_caps:
            row = "{:12.3f} |".format(cap * rep_bpb / (1024 ** 3))
            for pol in policies:
                insts = cap_lookup.get((cap, pol), {})
                m = insts.get(iid)
                if m:
                    row += " {:6.4f}/{:6.4f}/{:6.4f} |".format(m["total"], m["local"], m["remote"])
                else:
                    row += " {:^22} |".format("N/A")
            print(row)
        print("\nLegend: Total / Local / Remote")


# ============================================================================
# 时序图
# ============================================================================

def _plot_timeseries(csv_save_dir, results_by_policy, output_dir, target_caps=None, bytes_per_block_map=None):
    """为指定容量点生成命中率时序图，图表保存至 output_dir/timeseries/"""
    print("\n" + "=" * 60)
    print("Generating Timeseries Plots")
    print("=" * 60)
    count = 0
    for pol, results in results_by_policy.items():
        caps = target_caps or [r["capacity"] for r in results]
        for cap in caps:
            cap_dir = os.path.join(csv_save_dir, "cap_{}_{}" .format(cap, pol))
            if os.path.exists(cap_dir):
                print("Plotting {} capacity={}...".format(pol, cap))
                try:
                    plot_multi_instance_analysis(
                        cap_dir, output_dir,
                        show_template=False,
                        bytes_per_block_map=bytes_per_block_map,
                    )
                    count += 1
                except Exception as e:
                    print("  Failed: {}".format(e))
    print("\nGenerated {} timeseries plots.".format(count))


# ============================================================================
# 主流程
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Pareto curve analysis (single or multi-policy)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 单策略
  python run/tradeoff.py -c config.json --num-points 30

  # 多策略对比
  python run/tradeoff.py -c config.json --eviction-policies lru leaf_aware_lru random_lru

  # 从已有 CSV 加载（跳过实验）
  python run/tradeoff.py -c config.json --skip-run
  python run/tradeoff.py -c config.json --eviction-policies lru leaf_aware_lru --skip-run
        """,
    )
    parser.add_argument("-c", "--config", required=True, help="Config file path")
    parser.add_argument("--eviction-policies", nargs="+", default=None,
                        help="驱逐策略列表（不指定则使用配置中的默认策略）")
    parser.add_argument("--warmup-capacity", type=float, default=10000.0,
                        help="Warmup 容量上限，单位 GB（应足够大以容纳全量 trace 数据，默认 10000 GB）")
    parser.add_argument("--num-points", type=int, default=40)
    parser.add_argument("--hit-rate-type", default="total",
                        choices=["total", "local", "remote", "all"])
    parser.add_argument("--max-workers", type=int, default=4)
    parser.add_argument("--save-csv", action="store_true",
                        help="保留每次运行的 CSV 文件")
    parser.add_argument("--csv-output-dir", default=None,
                        help="CSV 保存目录（默认: <output_result_path>/csv_results）")
    parser.add_argument("--skip-run", action="store_true",
                        help="跳过实验，从已有 CSV 目录加载数据")
    parser.add_argument("--plot-timeseries", action="store_true",
                        help="为容量点生成时序图（需要 --save-csv 或 --skip-run）")
    parser.add_argument("--plot-capacity", type=int, nargs="+", default=None,
                        help="只为指定容量点生成时序图")
    parser.add_argument("--x-min", type=float, default=None)
    parser.add_argument("--x-max", type=float, default=None)
    parser.add_argument("--y-min", type=float, default=None)
    parser.add_argument("--y-max", type=float, default=None)
    args = parser.parse_args()

    init_kvcm_logger()

    # ------------------------------------------------------------------
    # Check for tiered mode and warn if detected
    # ------------------------------------------------------------------
    import json as _json
    with open(args.config, "r") as _f:
        _raw_config = _json.load(_f)
    for group in _raw_config.get("instance_groups", []):
        if group.get("hierarchical_eviction_enabled", False):
            print("\n" + "="*70)
            print("WARNING: Tradeoff analysis is designed for non-tiered mode.")
            print("In tiered mode (hierarchical_eviction_enabled=true), each tier's")
            print("capacity is fixed in the config. The capacity sweep only modifies")
            print("quota_capacity, which does not affect per-tier eviction decisions.")
            print("Results may not reflect meaningful capacity-performance tradeoffs.")
            print("="*70 + "\n")
            break

    bytes_per_block_map = extract_bytes_per_block_map(args.config)
    config_quota_gb_map = extract_config_quota_gb_map(args.config)

    config_loader = kvcm_py_optimizer.OptimizerConfigLoader()
    if not config_loader.load(args.config):
        print("Failed to load config")
        sys.exit(1)
    config = config_loader.config()

    policies = args.eviction_policies or [None]
    multi_policy = len(policies) > 1 or (policies[0] is not None)
    mode_name = "Multi-Policy Comparison" if multi_policy else "Single Policy Pareto"

    print("=" * 60)
    print(mode_name)
    print("=" * 60)
    print("Config:   {}".format(args.config))
    if multi_policy:
        print("Policies: {}".format(", ".join(p for p in policies if p)))
    print("Output:   {}".format(config.output_result_path()))
    print()

    csv_save_dir = args.csv_output_dir or os.path.join(config.output_result_path(), "csv_results")

    # ----------------------------------------------------------------
    # 数据获取：运行实验 or 加载已有 CSV
    # ----------------------------------------------------------------
    if args.skip_run:
        results_by_policy = load_results_from_csv_dir(csv_save_dir, bytes_per_block_map)
        if not results_by_policy:
            print("Error: No data loaded. Check --csv-output-dir path.")
            sys.exit(1)
        if multi_policy and policies[0] is not None:
            results_by_policy = {p: results_by_policy[p] for p in policies if p in results_by_policy}
    else:
        # Use first instance's bytes_per_block as a representative value to convert
        # warmup_capacity (GB) to blocks. This assumes all instances share the same
        # bytes_per_block (i.e., same model / same block_size and bytes_per_token).
        # If instances have heterogeneous bytes_per_block, the conversion is approximate
        # but acceptable since warmup only needs a sufficiently large upper bound.
        rep_bpb = next(iter(bytes_per_block_map.values()))
        warmup_blocks = int(args.warmup_capacity * (1024 ** 3) / rep_bpb)
        max_blocks = warmup_pass(args.config, warmup_blocks, bytes_per_block_map)
        capacities = generate_capacity_list(max_blocks, args.num_points)
        if not capacities:
            print("Error: No valid capacity points generated (max_blocks={})".format(max_blocks))
            sys.exit(1)
        print("\nGenerated {} capacity points: {:.2f} GB to {:.2f} GB".format(
            len(capacities),
            capacities[0] * rep_bpb / (1024 ** 3),
            capacities[-1] * rep_bpb / (1024 ** 3)))

        if args.save_csv:
            print("CSV files will be saved to: {}\n".format(csv_save_dir))

        # NOTE: The capacity sweep modifies quota_capacity only. In tiered mode,
        # eviction is driven by each tier's independent storages[i].capacity,
        # so these experiments do not produce meaningful tradeoff data.
        experiments = [(cap, pol) for pol in policies for cap in capacities]
        csv_dir_arg = csv_save_dir if args.save_csv else None
        raw = run_experiments_parallel(
            args.config, experiments, bytes_per_block_map, args.max_workers,
            csv_dir_arg,
        )
        results_by_policy = _group_by_policy(raw)

    # ----------------------------------------------------------------
    # 打印命中率表格
    # ----------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Hit Rate Results")
    print("=" * 60)
    actual_policies = sorted(results_by_policy.keys())

    if len(actual_policies) == 1:
        _print_single_policy_table(results_by_policy[actual_policies[0]], bytes_per_block_map)
    else:
        _print_multi_policy_table(results_by_policy, actual_policies, bytes_per_block_map)
        print("\n" + "=" * 60)
        print("Execution Summary")
        print("=" * 60)
        for pol in actual_policies:
            print("{}: {} capacity points".format(pol, len(results_by_policy[pol])))

    # ----------------------------------------------------------------
    # 绘图
    # ----------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Plotting Results")
    print("=" * 60)
    output_dir = config.output_result_path()
    axis_limits = {"x_min": args.x_min, "x_max": args.x_max, "y_min": args.y_min, "y_max": args.y_max}
    hit_types = ["total", "local", "remote"] if args.hit_rate_type == "all" else [args.hit_rate_type]

    if len(actual_policies) == 1:
        for ht in hit_types:
            plot_single_policy_curves(
                results_by_policy[actual_policies[0]], output_dir, bytes_per_block_map, ht,
                axis_limits=axis_limits, config_quota_gb_map=config_quota_gb_map)
    else:
        for ht in hit_types:
            plot_multi_policy_subplots(
                results_by_policy, output_dir, bytes_per_block_map, ht,
                config_quota_gb_map=config_quota_gb_map)

    # ----------------------------------------------------------------
    # 可选：时序图
    # ----------------------------------------------------------------
    has_csv = args.save_csv or args.skip_run

    if args.plot_timeseries and has_csv:
        _plot_timeseries(csv_save_dir, results_by_policy, output_dir, args.plot_capacity, bytes_per_block_map)
    elif args.plot_timeseries and not has_csv:
        print("\nWarning: --plot-timeseries requires --save-csv or --skip-run")

    print("\nAnalysis complete!")


if __name__ == "__main__":
    main()
