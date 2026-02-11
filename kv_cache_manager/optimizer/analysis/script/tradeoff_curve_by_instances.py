#!/usr/bin/env python3
"""
单策略 Pareto 曲线分析工具
"""

import argparse
import sys
import os
import numpy as np
from kv_cache_manager.optimizer.pybind import kvcm_py_optimizer
import optimizer_analysis_utils as utils


def main():
    parser = argparse.ArgumentParser(
        description='Generate Pareto curve for single policy'
    )
    parser.add_argument('-c', '--config', required=True, help='Config file path')
    parser.add_argument('--warmup-capacity', type=int, default=30000000)
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
    
    args = parser.parse_args()
    
    # 初始化
    utils.init_kvcm_logger()
    
    config_loader = kvcm_py_optimizer.OptimizerConfigLoader()
    if not config_loader.load(args.config):
        print("Failed to load config")
        sys.exit(1)
    config = config_loader.config()
    
    print("=" * 60)
    print("Single Policy Pareto Analysis")
    print("=" * 60)
    print(f"Config: {args.config}")
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
    
    # 运行实验（使用配置文件中的策略）
    experiments = [(cap, None) for cap in capacities]
    results = utils.run_experiments_parallel(args.config, experiments, args.max_workers, csv_save_dir)
    
    # 整理结果
    successful_results = [
        {"capacity": r["capacity"], "instances": r["instances"]}
        for r in results if r["success"]
    ]
    successful_results.sort(key=lambda x: x["capacity"])
    
    # 打印统计信息
    print("\n" + "=" * 60)
    print("Statistics Summary")
    print("=" * 60)
    
    if successful_results:
        # 收集每个 instance 的所有命中率数据
        instance_stats = {}
        instance_ids = list(successful_results[0]["instances"].keys())
        
        for instance_id in instance_ids:
            hit_rates = []
            for result in successful_results:
                if instance_id in result["instances"]:
                    hit_rates.append(result["instances"][instance_id]["total"])
            
            if hit_rates:
                instance_stats[instance_id] = {
                    "total": hit_rates,
                    "internal": [r["instances"][instance_id]["internal"] 
                                for r in successful_results if instance_id in r["instances"]],
                    "external": [r["instances"][instance_id]["external"] 
                                for r in successful_results if instance_id in r["instances"]]
                }
        
        # 打印每个实验的命中率
        print("\nHit Rate Results:")
        print("-" * 60)
        print(f"{'Capacity':>12} | {'Instance':<20} | {'Total':>10} | {'Internal':>10} | {'External':>10}")
        print("-" * 60)
        for result in successful_results:
            capacity = result["capacity"]
            for instance_id in instance_ids:
                if instance_id in result["instances"]:
                    metrics = result["instances"][instance_id]
                    print(f"{capacity:12,} | {instance_id:<20} | {metrics['total']:10.6f} | {metrics['internal']:10.6f} | {metrics['external']:10.6f}")
    
    # 绘图
    print("\n" + "=" * 60)
    print("Plotting Results")
    print("=" * 60)
    
    output_dir = config.output_result_path()
    
    # 绘制 Pareto 曲线
    if args.hit_rate_type == 'all':
        for hit_type in ['total', 'internal', 'external']:
            utils.plot_single_policy_curves(successful_results, output_dir, hit_type)
    else:
        utils.plot_single_policy_curves(successful_results, output_dir, args.hit_rate_type)
    
    # 如果启用了时序图生成
    if args.plot_timeseries and args.save_csv:
        print("\n" + "=" * 60)
        print("Generating Timeseries Plots")
        print("=" * 60)
        
        from plot_hit_rate_with_storage import plot_multi_instance_analysis
        
        # 为每个容量点生成时序图
        for result in successful_results:
            capacity = result["capacity"]
            csv_dir = os.path.join(csv_save_dir, f"cap_{capacity}_default_policy")
            
            if os.path.exists(csv_dir):
                print(f"Plotting capacity={capacity}...")
                try:
                    plot_multi_instance_analysis(csv_dir)
                except Exception as e:
                    print(f"  ✗ Failed: {e}")
        
        print("\nTimeseries plots complete!")
    elif args.plot_timeseries and not args.save_csv:
        print("\n⚠️  Warning: --plot-timeseries requires --save-csv")
    
    print("\nAnalysis complete!")


if __name__ == "__main__":
    main()