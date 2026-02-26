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
    parser.add_argument('--skip-run', action='store_true',
                        help='跳过实验运行，直接从已有 CSV 加载数据并绘图')
    parser.add_argument('--x-min', type=float, default=None,
                        help='X轴（容量）最小值（单位：blocks）')
    parser.add_argument('--x-max', type=float, default=None,
                        help='X轴（容量）最大值（单位：blocks）')
    parser.add_argument('--y-min', type=float, default=None,
                        help='Y轴（命中率）最小值（0-1）')
    parser.add_argument('--y-max', type=float, default=None,
                        help='Y轴（命中率）最大值（0-1）')
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
    
    # 确定 CSV 保存目录
    if args.csv_output_dir:
        csv_save_dir = args.csv_output_dir
    else:
        csv_save_dir = os.path.join(config.output_result_path(), 'csv_results')
    
    if args.skip_run:
        # 从 CSV 加载数据
        print(f"Loading data from CSV directory: {csv_save_dir}")
        
        if not os.path.exists(csv_save_dir):
            print(f"Error: CSV directory not found: {csv_save_dir}")
            sys.exit(1)
        
        # 扫描所有 CSV 目录
        csv_dirs = []
        for item in os.listdir(csv_save_dir):
            item_path = os.path.join(csv_save_dir, item)
            if os.path.isdir(item_path) and item.startswith('cap_'):
                csv_dirs.append(item_path)
        
        if not csv_dirs:
            print(f"Error: No CSV directories found in {csv_save_dir}")
            sys.exit(1)
        
        csv_dirs.sort()
        print(f"Found {len(csv_dirs)} CSV directories\n")
        
        # 从每个目录加载数据
        successful_results = []
        for csv_dir in csv_dirs:
            dirname = os.path.basename(csv_dir)
            # 解析目录名: cap_XXX_default_policy
            try:
                capacity = int(dirname.split('_')[1])
            except:
                print(f"Warning: Cannot parse capacity from {dirname}, skipping")
                continue
            
            print(f"Loading capacity={capacity} from {dirname}...")
            
            # 读取该目录下所有 CSV 文件并计算命中率
            instances = {}
            for csv_file in os.listdir(csv_dir):
                if csv_file.endswith('.csv'):
                    # 解析实例名: Qwen3-Coder-Plus_hit_rates.csv -> Qwen3-Coder-Plus
                    instance_id = csv_file.replace('_hit_rates.csv', '')
                    csv_path = os.path.join(csv_dir, csv_file)
                    
                    try:
                        import pandas as pd
                        df = pd.read_csv(csv_path)
                        
                        # 使用最后一行的累积命中率
                        if len(df) > 0:
                            last_row = df.iloc[-1]
                            
                            # 尝试使用累积命中率列
                            if 'AccHitRate' in df.columns:
                                total_hit_rate = last_row['AccHitRate']
                            elif 'HitRate' in df.columns:
                                total_hit_rate = last_row['HitRate']
                            else:
                                total_hit_rate = 0.0
                            
                            # Internal 命中率
                            if 'AccInternalHitRate' in df.columns:
                                internal_hit_rate = last_row['AccInternalHitRate']
                            elif 'InternalHitRate' in df.columns:
                                internal_hit_rate = last_row['InternalHitRate']
                            else:
                                internal_hit_rate = 0.0
                            
                            # External 命中率
                            if 'AccExternalHitRate' in df.columns:
                                external_hit_rate = last_row['AccExternalHitRate']
                            elif 'ExternalHitRate' in df.columns:
                                external_hit_rate = last_row['ExternalHitRate']
                            else:
                                external_hit_rate = 0.0
                            
                            instances[instance_id] = {
                                'total': total_hit_rate,
                                'internal': internal_hit_rate,
                                'external': external_hit_rate
                            }
                    except Exception as e:
                        print(f"  Warning: Failed to load {csv_file}: {e}")
            
            if instances:
                successful_results.append({
                    "capacity": capacity,
                    "instances": instances
                })
                print(f"  ✓ Loaded {len(instances)} instances\n")
            else:
                print(f"  ✗ No valid data found\n")
        
        if not successful_results:
            print("Error: No valid data loaded from CSV files")
            sys.exit(1)
        
        successful_results.sort(key=lambda x: x["capacity"])
        print(f"Successfully loaded data for {len(successful_results)} capacity points\n")
    
    else:
        # 原有的运行实验逻辑
        # Warmup
        max_blocks = utils.warmup_pass(args.config, args.warmup_capacity)
        
        # 生成容量列表
        capacities = utils.generate_capacity_list(max_blocks, args.num_points)
        print(f"\nGenerated {len(capacities)} capacity points")
        print(f"Range: {capacities[0]} to {capacities[-1]} blocks\n")
        
        csv_save_dir_arg = csv_save_dir if args.save_csv else None
        if args.save_csv:
            print(f"CSV files will be saved to: {csv_save_dir}\n")
        
        # 运行实验（使用配置文件中的策略）
        experiments = [(cap, None) for cap in capacities]
        results = utils.run_experiments_parallel(args.config, experiments, args.max_workers, csv_save_dir_arg)
        
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
    axis_limits = {
        'x_min': args.x_min,
        'x_max': args.x_max,
        'y_min': args.y_min,
        'y_max': args.y_max
    }
    
    if args.hit_rate_type == 'all':
        for hit_type in ['total', 'internal', 'external']:
            utils.plot_single_policy_curves(successful_results, output_dir, hit_type, 
                                           title=None, axis_limits=axis_limits)
    else:
        utils.plot_single_policy_curves(successful_results, output_dir, args.hit_rate_type,
                                       title=None, axis_limits=axis_limits)
    
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