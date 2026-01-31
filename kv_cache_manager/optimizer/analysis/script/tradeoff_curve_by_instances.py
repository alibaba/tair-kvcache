#!/usr/bin/env python3
"""
单策略 Pareto 曲线分析工具
"""

import argparse
import sys
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
    
    # 运行实验（使用配置文件中的策略）
    experiments = [(cap, None) for cap in capacities]
    results = utils.run_experiments_parallel(args.config, experiments, args.max_workers)
    
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
    
    if args.hit_rate_type == 'all':
        for hit_type in ['total', 'internal', 'external']:
            utils.plot_single_policy_curves(successful_results, output_dir, hit_type)
    else:
        utils.plot_single_policy_curves(successful_results, output_dir, args.hit_rate_type)
    
    print("\nAnalysis complete!")


if __name__ == "__main__":
    main()