#!/usr/bin/env python3
"""
多策略对比分析工具
"""

import argparse
import sys
import os
import numpy as np
from collections import defaultdict
from kv_cache_manager.optimizer.pybind import kvcm_py_optimizer
import optimizer_analysis_utils as utils


def main():
    parser = argparse.ArgumentParser(
        description='Multi-policy comparison analysis'
    )
    parser.add_argument('-c', '--config', required=True, help='Config file path')
    parser.add_argument('--warmup-capacity', type=int, default=30000000)
    parser.add_argument('--eviction-policies', nargs='+', 
                        default=['lru', 'random_lru', 'leaf_aware_lru'])
    parser.add_argument('--num-points', type=int, default=40)
    parser.add_argument('--hit-rate-type', default='total',
                        choices=['total', 'internal', 'external', 'all'])
    parser.add_argument('--max-workers', type=int, default=4)
    parser.add_argument('--save-csv', action='store_true',
                        help='保留每次运行的 CSV 文件（用于后续画时序图）')
    parser.add_argument('--csv-output-dir', default=None,
                        help='CSV 保存目录（默认为 <output_result_path>/csv_results）')
    parser.add_argument('--plot-timeseries', action='store_true',
                        help='自动为每个容量点生成时序图（需要 --save-csv）')
    parser.add_argument('--plot-capacity', type=int, nargs='+', default=None,
                        help='只为指定的容量点生成时序图（例如：--plot-capacity 5000 10000）')
    
    args = parser.parse_args()
    
    # 初始化
    utils.init_kvcm_logger()
    
    config_loader = kvcm_py_optimizer.OptimizerConfigLoader()
    if not config_loader.load(args.config):
        print("Failed to load config")
        sys.exit(1)
    config = config_loader.config()
    
    print("=" * 60)
    print("Multi-Policy Comparison Analysis")
    print("=" * 60)
    print(f"Config: {args.config}")
    print(f"Policies: {', '.join(args.eviction_policies)}")
    print(f"Output: {config.output_result_path()}")
    print()
    
    # Warmup
    max_blocks = utils.warmup_pass(args.config, args.warmup_capacity)
    
    # 生成容量列表
    capacities = utils.generate_capacity_list(max_blocks, args.num_points)
    print(f"\nGenerated {len(capacities)} capacity points")
    print(f"Range: {capacities[0]} to {capacities[-1]} blocks\n")
    
    # 确定 CSV 保存目录
    csv_save_dir = None
    if args.save_csv:
        if args.csv_output_dir:
            csv_save_dir = args.csv_output_dir
        else:
            csv_save_dir = os.path.join(config.output_result_path(), 'csv_results')
        print(f"CSV files will be saved to: {csv_save_dir}\n")
    
    # 生成所有实验组合
    experiments = [
        (capacity, policy) 
        for policy in args.eviction_policies 
        for capacity in capacities
    ]
    
    # 并行运行所有实验
    results = utils.run_experiments_parallel(args.config, experiments, args.max_workers, csv_save_dir)
    
    # 按策略组织结果
    results_by_policy = defaultdict(list)
    for r in results:
        if r["success"]:
            results_by_policy[r["policy"]].append({
                "capacity": r["capacity"],
                "instances": r["instances"]
            })
    
    # 排序
    for policy in results_by_policy:
        results_by_policy[policy].sort(key=lambda x: x["capacity"])
    
    # 打印统计信息
    print("\n" + "=" * 60)
    print("Statistics Summary")
    print("=" * 60)
    
    # 按容量组织结果，方便横向对比策略
    print("\nHit Rate Results:")
    
    # 获取所有容量点
    all_capacities = sorted(set(
        r["capacity"] 
        for policy in results_by_policy.values() 
        for r in policy
    ))
    
    # 获取所有 instance
    all_instances = sorted(set(
        instance_id
        for policy in results_by_policy.values()
        for r in policy
        for instance_id in r["instances"].keys()
    ))
    
    # 构建 {capacity: {policy: {instance: metrics}}} 的结构
    capacity_data = {}
    for capacity in all_capacities:
        capacity_data[capacity] = {}
        for policy in args.eviction_policies:
            if policy in results_by_policy:
                for r in results_by_policy[policy]:
                    if r["capacity"] == capacity:
                        capacity_data[capacity][policy] = r["instances"]
                        break
    
    # 为每个 instance 打印一个表格
    for instance_id in all_instances:
        print("\n" + "=" * 80)
        print(f"Instance: {instance_id}")
        print("=" * 80)
        print(f"{'Capacity':>12} |", end="")
        for policy in args.eviction_policies:
            print(f" {policy:<20} |", end="")
        print()
        print("-" * 80)
        
        for capacity in all_capacities:
            print(f"{capacity:12,} |", end="")
            for policy in args.eviction_policies:
                if policy in capacity_data[capacity] and instance_id in capacity_data[capacity][policy]:
                    metrics = capacity_data[capacity][policy][instance_id]
                    print(f" {metrics['total']:10.6f} / {metrics['internal']:8.6f} / {metrics['external']:8.6f} |", end="")
                else:
                    print(f" {'N/A':^44} |", end="")
            print()
        
        print("\nLegend: Total / Internal / External hit rate")
    
    # 打印汇总摘要
    print("\n" + "=" * 80)
    print("Execution Summary")
    print("=" * 60)
    for policy in args.eviction_policies:
        count = len(results_by_policy[policy]) if policy in results_by_policy else 0
        print(f"{policy}: {count}/{len(capacities)} experiments succeeded")
    
    # 绘图
    print("\n" + "=" * 60)
    print("Plotting Results")
    print("=" * 60)
    
    output_dir = config.output_result_path()
    
    if args.hit_rate_type == 'all':
        for hit_type in ['total', 'internal', 'external']:
            print(f"\nPlotting {hit_type} hit rate...")
            utils.plot_multi_policy_subplots(dict(results_by_policy), output_dir, hit_type)
    else:
        utils.plot_multi_policy_subplots(dict(results_by_policy), output_dir, args.hit_rate_type)
    
    # 打印摘要
    print("\n" + "=" * 60)
    print("Execution Summary")
    print("=" * 60)
    for policy in args.eviction_policies:
        count = len(results_by_policy[policy]) if policy in results_by_policy else 0
        print(f"{policy}: {count}/{len(capacities)} experiments succeeded")
    
    # 如果启用了时序图生成
    if args.plot_timeseries and args.save_csv:
        print("\n" + "=" * 60)
        print("Generating Timeseries Plots")
        print("=" * 60)
        
        from plot_hit_rate_with_storage import plot_multi_instance_analysis
        
        # 确定要画图的容量点
        if args.plot_capacity:
            target_capacities = args.plot_capacity
            print(f"Plotting specified capacities: {target_capacities}")
        else:
            # 默认：为所有容量点生成图
            target_capacities = capacities
            print(f"Plotting all {len(target_capacities)} capacity points...")
        
        # 为每个策略和容量点生成时序图
        total_plots = 0
        for policy in args.eviction_policies:
            for capacity in target_capacities:
                csv_dir = os.path.join(csv_save_dir, f"cap_{capacity}_{policy}")
                
                if os.path.exists(csv_dir):
                    print(f"Plotting {policy} capacity={capacity}...")
                    try:
                        plot_multi_instance_analysis(csv_dir)
                        total_plots += 1
                    except Exception as e:
                        print(f"  ✗ Failed: {e}")
        
        print(f"\nGenerated {total_plots} timeseries plots!")
    elif args.plot_timeseries and not args.save_csv:
        print("\n⚠️  Warning: --plot-timeseries requires --save-csv")
    
    print("\nAnalysis complete!")


if __name__ == "__main__":
    main()