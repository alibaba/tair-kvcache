#!/usr/bin/env python3
"""
KVCache Manager Analysis Utilities

公共工具函数库，包含：
- CSV 文件收集和解析
- Optimizer 运行封装
- 并行执行框架
- 容量列表生成
- 绘图工具
"""

import os
import shutil
import tempfile
import threading
from typing import Dict, List, Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
import json

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from kv_cache_manager.optimizer.pybind import kvcm_py_optimizer


# ============================================================================
# 配置和常量
# ============================================================================

COLORS = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 
          'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan']
MARKERS = ['o', 's', '^', 'D', 'v', '<', '>', 'p', 'h', '*']


# ============================================================================
# CSV 文件处理
# ============================================================================

def collect_instance_csvs(output_dir: str) -> Dict[str, str]:
    """
    收集所有 instance 的 CSV 文件
    
    Args:
        output_dir: 输出目录路径
    
    Returns:
        {instance_id: csv_file_path}
    """
    csv_map = {}
    
    if not os.path.exists(output_dir):
        return csv_map
    
    for filename in os.listdir(output_dir):
        if filename.endswith('_hit_rates.csv'):
            instance_id = filename.replace('_hit_rates.csv', '')
            csv_path = os.path.join(output_dir, filename)
            csv_map[instance_id] = csv_path
    
    return csv_map


def parse_instance_metrics(csv_file: str) -> dict:
    """
    从单个 instance 的 CSV 解析指标（取最后一行）
    
    Returns:
        {
            "acc_total_hit_rate": float,
            "acc_internal_hit_rate": float,
            "acc_external_hit_rate": float,
            "cached_blocks_all": int
        }
    """
    df = pd.read_csv(csv_file)
    if df.empty:
        return None
    
    last_row = df.iloc[-1]
    
    return {
        "acc_total_hit_rate": float(last_row['AccHitRate']),
        "acc_internal_hit_rate": float(last_row['AccInternalHitRate']),
        "acc_external_hit_rate": float(last_row['AccExternalHitRate']),
        "cached_blocks_all": int(last_row['CachedBlocksAllInstance'])
    }


# ============================================================================
# Optimizer 运行封装
# ============================================================================

def run_optimizer_with_config(config_path: str, capacity: int, policy: str = None) -> str:
    """
    运行 optimizer 并返回临时输出目录
    
    Args:
        config_path: 配置文件路径
        capacity: quota_capacity (blocks)
        policy: 驱逐策略名称（可选，如果为 None 则使用配置文件中的策略）
    
    Returns:
        临时输出目录路径
    """
    # 创建临时输出目录
    temp_dir = tempfile.mkdtemp(prefix="kvcm_analysis_")
    
    # 读取原始配置文件
    with open(config_path, 'r') as f:
        config_json = json.load(f)
    
    # 修改配置
    for group in config_json.get('instance_groups', []):
        # 设置容量
        group['quota_capacity'] = capacity
        
        # 设置策略（如果指定）
        if policy is not None:
            for instance in group.get('instances', []):
                instance['eviction_policy_type'] = policy
    
    # 修改输出路径
    config_json['output_result_path'] = temp_dir
    
    # 创建临时配置文件
    temp_config_path = os.path.join(temp_dir, 'temp_config.json')
    with open(temp_config_path, 'w') as f:
        json.dump(config_json, f, indent=2)
    
    # 加载配置并运行
    config_loader = kvcm_py_optimizer.OptimizerConfigLoader()
    if not config_loader.load(temp_config_path):
        raise RuntimeError(f"Failed to load config from {temp_config_path}")
    config = config_loader.config()
    
    manager = kvcm_py_optimizer.OptimizerManager(config)
    manager.Init()
    manager.DirectRun()
    manager.AnalyzeResults()
    
    return temp_dir


def warmup_pass(config_path: str, warmup_capacity: int, policy: str = None) -> int:
    """
    Warmup 阶段: 用大容量运行,获取整个 group 的最大 block 数
    
    Args:
        config_path: 配置文件路径
        warmup_capacity: 大容量值
        policy: 驱逐策略（可选）
    
    Returns:
        max_blocks (int)
    """
    print(f"Running warmup with capacity={warmup_capacity}...")
    
    temp_dir = run_optimizer_with_config(config_path, warmup_capacity, policy)
    
    try:
        csv_map = collect_instance_csvs(temp_dir)
        if not csv_map:
            raise RuntimeError("No CSV files found after warmup")
        
        first_csv = next(iter(csv_map.values()))
        df = pd.read_csv(first_csv)
        
        max_blocks = int(df['CachedBlocksAllInstance'].max())
        print(f"Warmup complete. Max blocks in group: {max_blocks}")
        return max_blocks
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


# ============================================================================
# 容量列表生成
# ============================================================================

def generate_capacity_list(max_blocks: int, num_points: int, min_capacity: int = 2000) -> List[int]:
    """
    生成容量列表,使用指数分布
    
    Args:
        max_blocks: 最大容量
        num_points: 采样点数量
        min_capacity: 最小容量阈值
    
    Returns:
        容量列表（从小到大排序）
    """
    x = np.linspace(-4, 4, num_points)
    ratios = np.exp(x) / np.exp(4)
    capacities = sorted(set(
        int(max_blocks * r) 
        for r in ratios 
        if int(max_blocks * r) > min_capacity
    ))
    return capacities


# ============================================================================
# 并行执行框架
# ============================================================================

def run_single_experiment(
    config_path: str, 
    capacity: int, 
    policy: str,
    exp_id: int, 
    total_exps: int
) -> dict:
    """
    运行单个实验
    
    Returns:
        {
            "policy": str,
            "capacity": int,
            "instances": {...},
            "success": bool,
            "error": str (if failed)
        }
    """
    result = {
        "policy": policy,
        "capacity": capacity,
        "instances": {},
        "success": False,
        "error": None
    }
    
    temp_dir = None
    try:
        thread_id = threading.current_thread().name
        print(f"[{thread_id}] [{exp_id}/{total_exps}] Running {policy} capacity={capacity}...")
        
        temp_dir = run_optimizer_with_config(config_path, capacity, policy)
        csv_map = collect_instance_csvs(temp_dir)
        
        if not csv_map:
            result["error"] = "No CSV files found"
            return result
        
        # 收集每个 instance 的指标
        instance_metrics = {}
        for instance_id, csv_file in csv_map.items():
            metrics = parse_instance_metrics(csv_file)
            if metrics is None:
                continue
            
            instance_metrics[instance_id] = {
                "total": metrics["acc_total_hit_rate"],
                "internal": metrics["acc_internal_hit_rate"],
                "external": metrics["acc_external_hit_rate"],
                "cached_blocks_all": metrics["cached_blocks_all"]
            }
        
        result["instances"] = instance_metrics
        result["success"] = True
        
        print(f"[{thread_id}] [{exp_id}/{total_exps}] ✓ {policy} capacity={capacity} completed")
        
    except Exception as e:
        result["error"] = str(e)
        print(f"[{thread_id}] [{exp_id}/{total_exps}] ✗ {policy} capacity={capacity} failed: {e}")
    
    finally:
        if temp_dir and os.path.exists(temp_dir):
            shutil.rmtree(temp_dir, ignore_errors=True)
    
    return result


def run_experiments_parallel(
    config_path: str,
    experiments: List[tuple],  # [(capacity, policy), ...]
    max_workers: int = 4
) -> List[dict]:
    """
    并行运行多个实验
    
    Args:
        config_path: 配置文件路径
        experiments: 实验列表 [(capacity, policy), ...]
        max_workers: 最大并行线程数
    
    Returns:
        实验结果列表
    """
    print(f"\n{'='*60}")
    print(f"Running Parallel Experiments")
    print(f"{'='*60}")
    print(f"Total experiments: {len(experiments)}")
    print(f"Max parallel workers: {max_workers}")
    print(f"{'='*60}\n")
    
    # 准备任务
    tasks = []
    for i, (capacity, policy) in enumerate(experiments):
        tasks.append((config_path, capacity, policy, i+1, len(experiments)))
    
    # 并行执行
    results = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_task = {
            executor.submit(run_single_experiment, *task): task 
            for task in tasks
        }
        
        for future in as_completed(future_to_task):
            result = future.result()
            results.append(result)
    
    # 统计成功率
    success_count = sum(1 for r in results if r["success"])
    print(f"\n{'='*60}")
    print(f"Execution Summary: {success_count}/{len(experiments)} succeeded")
    print(f"{'='*60}\n")
    
    return results


# ============================================================================
# 绘图工具
# ============================================================================

def plot_single_policy_curves(
    results: List[dict],
    output_dir: str,
    hit_rate_type: str = 'total',
    title: str = None
):
    """
    绘制单策略的 Trade-off 曲线（每个 instance 一条曲线）
    
    Args:
        results: 分析结果 [{"capacity": int, "instances": {...}}, ...]
        output_dir: 输出目录
        hit_rate_type: 'total', 'internal', 'external'
        title: 图表标题
    """
    if not results:
        print("No data to plot!")
        return
    
    instance_ids = list(results[0]["instances"].keys())
    
    plt.figure(figsize=(12, 8))
    
    for idx, instance_id in enumerate(instance_ids):
        capacities = []
        hit_rates = []
        
        for result in results:
            if instance_id in result["instances"]:
                capacities.append(result["capacity"])
                hit_rates.append(result["instances"][instance_id][hit_rate_type])
        
        if not capacities:
            continue
        
        color = COLORS[idx % len(COLORS)]
        marker = MARKERS[idx % len(MARKERS)]
        
        plt.scatter(capacities, hit_rates, color=color, marker=marker,
                   s=1, label=instance_id, alpha=0.8)
    
    plt.xlabel('Cache Capacity (blocks)', fontsize=12)
    plt.ylabel(f'{hit_rate_type.capitalize()} Hit Rate', fontsize=12)
    
    if title is None:
        title = f'KVCache Trade-off Curve - {hit_rate_type.capitalize()} Hit Rate'
    plt.title(title, fontsize=14)
    
    plt.legend(loc='lower right', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.ylim(0, 1)
    plt.tight_layout()
    
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f'pareto_curve_{hit_rate_type}.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved to {output_file}")
    plt.close()


def plot_multi_policy_subplots(
    results_by_policy: Dict[str, List[dict]],
    output_dir: str,
    hit_rate_type: str = 'total'
):
    """
    绘制多策略对比子图（每个 instance 一个子图，每个子图包含多条策略曲线）
    
    Args:
        results_by_policy: {"policy": [{"capacity": int, "instances": {...}}, ...]}
        output_dir: 输出目录
        hit_rate_type: 'total', 'internal', 'external'
    """
    if not results_by_policy:
        print("No data to plot!")
        return
    
    # 重组数据
    from collections import defaultdict
    plot_data = defaultdict(lambda: defaultdict(lambda: {"storage": [], "hit_rates": []}))
    
    for policy, results in results_by_policy.items():
        for result in results:
            for instance_id, metrics in result["instances"].items():
                plot_data[instance_id][policy]["storage"].append(metrics["cached_blocks_all"])
                plot_data[instance_id][policy]["hit_rates"].append(metrics[hit_rate_type])
    
    instance_ids = sorted(plot_data.keys())
    n_instances = len(instance_ids)
    
    # 计算子图布局
    if n_instances == 1:
        nrows, ncols = 1, 1
    elif n_instances <= 4:
        nrows, ncols = 2, 2
    elif n_instances <= 6:
        nrows, ncols = 2, 3
    elif n_instances <= 9:
        nrows, ncols = 3, 3
    else:
        ncols = 3
        nrows = (n_instances + ncols - 1) // ncols
    
    fig, axes = plt.subplots(nrows, ncols, figsize=(6*ncols, 5*nrows))
    if n_instances == 1:
        axes = [axes]
    else:
        axes = axes.flatten()
    
    policies = sorted(list(next(iter(plot_data.values())).keys()))
    
    for idx, instance_id in enumerate(instance_ids):
        ax = axes[idx]
        
        for policy_idx, policy_name in enumerate(policies):
            data = plot_data[instance_id][policy_name]
            color = COLORS[policy_idx % len(COLORS)]
            marker = MARKERS[policy_idx % len(MARKERS)]
            
            ax.scatter(data['storage'], data['hit_rates'],
                      label=policy_name, color=color, marker=marker, 
                      s=1, alpha=0.8)
        
        ax.set_xlabel('Group Total Storage (blocks)', fontsize=11)
        ax.set_ylabel(f'{hit_rate_type.capitalize()} Hit Rate', fontsize=11)
        ax.set_ylim(0, 1.05)
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.set_title(f'{instance_id}', fontsize=13, fontweight='bold')
        ax.legend(loc='lower right', fontsize=9, framealpha=0.9)
    
    # 隐藏多余的子图
    for idx in range(n_instances, len(axes)):
        axes[idx].set_visible(False)
    
    plt.suptitle(f'Multi-Policy Comparison - {hit_rate_type.capitalize()} Hit Rate', 
                 fontsize=16, fontweight='bold', y=0.998)
    plt.tight_layout()
    
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f'multi_policy_{hit_rate_type}.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\nSaved plot to: {output_file}")
    plt.close()


# ============================================================================
# 初始化工具
# ============================================================================

def init_kvcm_logger(log_level: int = 4):
    """
    初始化 KVCM 日志系统
    
    Args:
        log_level: 日志级别（默认 4）
    """
    kvcm_py_optimizer.LoggerBroker.InitLogger("")
    kvcm_py_optimizer.LoggerBroker.SetLogLevel(log_level)