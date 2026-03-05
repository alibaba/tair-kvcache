#!/usr/bin/env python3
"""
Optimizer 运行封装层

职责：
- 单次 optimizer 运行（临时目录管理）
- Warmup pass（获取 max_blocks）
- 并行实验框架（ThreadPoolExecutor）
- KVCM Logger 初始化
"""

import gc
import json
import os
import shutil
import tempfile
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Optional, Tuple

from kv_cache_manager.optimizer.pybind import kvcm_py_optimizer

from .csv_loader import collect_instance_csvs, parse_instance_metrics


# ============================================================================
# Logger
# ============================================================================

def init_kvcm_logger(log_level: int = 4):
    """初始化 KVCM 日志系统"""
    kvcm_py_optimizer.LoggerBroker.InitLogger("", False)
    kvcm_py_optimizer.LoggerBroker.SetLogLevel(log_level)


# ============================================================================
# 单次运行
# ============================================================================

def run_optimizer_with_config(
    config_path: str,
    capacity: int,
    policy: str = None,
    save_csv_to: str = None,
    enable_lifecycle_tracking: bool = False,
) -> str:
    """
    运行 optimizer，返回临时输出目录路径。

    调用方负责在使用完毕后删除临时目录。
    """
    temp_dir, _ = run_optimizer_with_config_explicit(
        config_path, capacity, policy, save_csv_to,
        enable_lifecycle_tracking,
    )
    return temp_dir


def run_optimizer_with_config_explicit(
    config_path: str,
    capacity: int,
    policy: str = None,
    save_csv_to: str = None,
    enable_lifecycle_tracking: bool = False,
) -> Tuple[str, object]:
    """
    运行 optimizer，返回 (临时目录路径, OptimizerManager对象)。

    显式返回 manager，供调用方在确定不再使用后手动 del + gc.collect()。
    """
    temp_dir = tempfile.mkdtemp(prefix="kvcm_analysis_")

    with open(config_path, "r") as f:
        config_json = json.load(f)

    # 设置容量 / 策略
    for group in config_json.get("instance_groups", []):
        group["quota_capacity"] = capacity
        if policy is not None:
            for instance in group.get("instances", []):
                instance["eviction_policy_type"] = policy

    config_json["output_result_path"] = temp_dir

    temp_config_path = os.path.join(temp_dir, "temp_config.json")
    with open(temp_config_path, "w") as f:
        json.dump(config_json, f, indent=2)

    config_loader = kvcm_py_optimizer.OptimizerConfigLoader()
    if not config_loader.load(temp_config_path):
        raise RuntimeError(f"Failed to load config: {temp_config_path}")
    config = config_loader.config()

    manager = kvcm_py_optimizer.OptimizerManager(config, enable_lifecycle_tracking)
    manager.Init()
    manager.DirectRun()
    manager.AnalyzeResults()

    if save_csv_to:
        import glob
        os.makedirs(save_csv_to, exist_ok=True)
        for csv_file in glob.glob(os.path.join(temp_dir, "*_hit_rates.csv")):
            shutil.copy(csv_file, save_csv_to)
        print(f"  → CSV saved to: {save_csv_to}")

    return temp_dir, manager


# ============================================================================
# Warmup Pass
# ============================================================================

def warmup_pass(
    config_path: str,
    warmup_capacity: int,
    policy: str = None,
    enable_lifecycle_tracking: bool = False,
) -> int:
    """
    用大容量跑一遍，获取 group 内最大 block 数。

    Returns:
        max_blocks (int)
    """
    import pandas as pd

    print(f"Running warmup with capacity={warmup_capacity}...")
    temp_dir, manager = run_optimizer_with_config_explicit(
        config_path, warmup_capacity, policy,
        enable_lifecycle_tracking=enable_lifecycle_tracking,
    )

    try:
        csv_map = collect_instance_csvs(temp_dir)
        if not csv_map:
            raise RuntimeError("No CSV files found after warmup")

        first_csv = next(iter(csv_map.values()))
        df = pd.read_csv(first_csv)
        max_blocks = int(df["CachedBlocksAllInstance"].max())

        acc_read = int(df["AccReadBlocks"].iloc[-1]) if "AccReadBlocks" in df.columns else 0
        acc_write = int(df["AccWriteBlocks"].iloc[-1]) if "AccWriteBlocks" in df.columns else 0
        print(f"Warmup done. Max cached: {max_blocks}, AccReadBlocks: {acc_read}, AccWriteBlocks: {acc_write}")

        return max_blocks
    finally:
        if manager is not None:
            manager.ClearAllCachesAndResetStats()
            del manager
        gc.collect()
        shutil.rmtree(temp_dir, ignore_errors=True)


# ============================================================================
# 并行实验框架
# ============================================================================

def run_single_experiment(
    config_path: str,
    capacity: int,
    policy: str,
    exp_id: int,
    total_exps: int,
    save_csv_to: str = None,
    enable_lifecycle_tracking: bool = False,
) -> dict:
    """
    运行单个实验，返回结果字典。

    Returns:
        {
            "policy": str,
            "capacity": int,
            "instances": {instance_id: {"total": float, "internal": float,
                                        "external": float, "cached_blocks_all": int}},
            "success": bool,
            "error": str | None,
        }
    """
    result = {
        "policy": policy,
        "capacity": capacity,
        "instances": {},
        "success": False,
        "error": None,
    }
    temp_dir = None
    manager = None
    try:
        tid = threading.current_thread().name
        print(f"[{tid}] [{exp_id}/{total_exps}] {policy} capacity={capacity}...")

        temp_dir, manager = run_optimizer_with_config_explicit(
            config_path, capacity, policy, save_csv_to,
            enable_lifecycle_tracking,
        )
        csv_map = collect_instance_csvs(temp_dir)
        if not csv_map:
            result["error"] = "No CSV files found"
            return result

        instance_metrics = {}
        for iid, csv_file in csv_map.items():
            metrics = parse_instance_metrics(csv_file)
            if metrics is None:
                continue
            instance_metrics[iid] = {
                "total": metrics["acc_total_hit_rate"],
                "internal": metrics["acc_internal_hit_rate"],
                "external": metrics["acc_external_hit_rate"],
                "cached_blocks_all": metrics["cached_blocks_all"],
            }

        result["instances"] = instance_metrics
        result["success"] = True
        print(f"[{tid}] [{exp_id}/{total_exps}] ✓ {policy} capacity={capacity}")

    except Exception as e:
        result["error"] = str(e)
        print(f"[{threading.current_thread().name}] [{exp_id}/{total_exps}] ✗ {policy} capacity={capacity}: {e}")

    finally:
        if manager is not None:
            manager.ClearAllCachesAndResetStats()
            del manager
        gc.collect()
        if temp_dir and os.path.exists(temp_dir):
            shutil.rmtree(temp_dir, ignore_errors=True)

    return result


def run_experiments_parallel(
    config_path: str,
    experiments: List[tuple],
    max_workers: int = 4,
    save_csv_dir: str = None,
    enable_lifecycle_tracking: bool = False,
) -> List[dict]:
    """
    并行运行实验列表。

    Args:
        experiments: [(capacity, policy), ...]

    Returns:
        实验结果列表（顺序不保证）
    """
    print(f"\n{'='*60}")
    print(f"Parallel Experiments: {len(experiments)} tasks, {max_workers} workers")
    print(f"{'='*60}\n")

    tasks = []
    for i, (capacity, policy) in enumerate(experiments):
        csv_subdir = None
        if save_csv_dir:
            policy_name = policy or "default_policy"
            csv_subdir = os.path.join(save_csv_dir, f"cap_{capacity}_{policy_name}")
        tasks.append((
            config_path, capacity, policy,
            i + 1, len(experiments),
            csv_subdir, enable_lifecycle_tracking,
        ))

    results = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(run_single_experiment, *t): t for t in tasks}
        for future in as_completed(futures):
            results.append(future.result())

    success = sum(1 for r in results if r["success"])
    print(f"\n{'='*60}")
    print(f"Done: {success}/{len(experiments)} succeeded")
    print(f"{'='*60}\n")
    return results
