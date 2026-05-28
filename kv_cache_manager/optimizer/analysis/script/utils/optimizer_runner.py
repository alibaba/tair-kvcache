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
import sys
import tempfile
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional, Tuple

from kv_cache_manager.optimizer.pybind import kvcm_py_optimizer

from .csv_loader import collect_instance_csvs, parse_instance_metrics


# ============================================================================
# Logger
# ============================================================================

def init_kvcm_logger(log_level: int = 4):
    """初始化 KVCM 日志系统。

    Python analysis targets are usually run through `bazel run`, where users
    expect progress and KVCM C++ logs to be visible in the terminal.  The C++
    default logger writes to logs/kv_cache_manager.log, so analysis scripts
    opt into console logging unless the caller explicitly sets
    KVCM_LOG_TO_CONSOLE=0.
    """
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(line_buffering=True)
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(line_buffering=True)
    os.environ.setdefault("KVCM_LOG_TO_CONSOLE", "1")
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
    enable_template_analysis: bool = False,
) -> str:
    """
    运行 optimizer，返回临时输出目录路径。

    调用方负责在使用完毕后删除临时目录。
    """
    temp_dir, _ = run_optimizer_with_config_explicit(
        config_path, capacity, policy, save_csv_to,
        enable_lifecycle_tracking, enable_template_analysis,
    )
    return temp_dir


def run_optimizer_with_config_explicit(
    config_path: str,
    capacity: int,
    policy: str = None,
    save_csv_to: str = None,
    enable_lifecycle_tracking: bool = False,
    enable_template_analysis: bool = False,
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
        # NOTE: In tiered mode (tier_strategy.hierarchical_eviction_enabled=true),
        # eviction decisions are based on each tier's independent storages[i].capacity,
        # not quota_capacity.
        # This function only modifies quota_capacity, so capacity sweeps in tradeoff
        # analysis are ineffective for tiered configurations.
        # quota_capacity in config is GB; convert blocks -> GB for C++ FromRapidValue (bytes internally)
        # Use instances[0] as representative for bytes_per_block calculation.
        # Assumption: all instances in the same group serve the same model and
        # therefore share identical block_size (enforced by C++ MismatchFields check
        # in RegisterInstance) and bytes_per_token (Python-only annotation field,
        # not validated by C++, but expected to be consistent within a group).
        inst = group.get("instances", [{}])[0] if group.get("instances") else {}
        bpb = inst.get("bytes_per_token", 0) * inst.get("block_size", 0)
        group["quota_capacity"] = -1 if capacity < 0 else capacity * bpb / (1024 ** 3) if bpb > 0 else capacity
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

    manager = kvcm_py_optimizer.OptimizerManager(config, enable_lifecycle_tracking, enable_template_analysis)
    manager.Init()
    manager.DirectRun()
    manager.AnalyzeResults()

    if save_csv_to:
        import glob
        os.makedirs(save_csv_to, exist_ok=True)
        patterns = ["*_hit_rates.csv", "*_template_prefix_traces.csv", "*_template_prefix_summary.csv"]
        for pattern in patterns:
            for csv_file in glob.glob(os.path.join(temp_dir, pattern)):
                shutil.copy(csv_file, save_csv_to)
        print(f"  → CSV saved to: {save_csv_to}")

    return temp_dir, manager


# ============================================================================
# 配置工具
# ============================================================================

def group_hierarchical_eviction_enabled(group: dict) -> bool:
    tier_strategy = group.get("tier_strategy", {})
    if isinstance(tier_strategy, dict) and "hierarchical_eviction_enabled" in tier_strategy:
        return bool(tier_strategy.get("hierarchical_eviction_enabled", False))
    return bool(group.get("hierarchical_eviction_enabled", False))


def extract_bytes_per_block_map(config_path: str) -> Dict[str, int]:
    """
    从 config JSON 提取每个 instance 的 bytes_per_block。

    bytes_per_block = block_size × bytes_per_token

    Note:
        bytes_per_token 是 Python 侧的配置注解字段，C++ 层（OptInstanceConfig）不解析该字段。
        block_size 在 KVCM C++ 层由 RegisterInstance/MismatchFields 强制要求同一 group 内一致；
        bytes_per_token 无 C++ 层强制约束，依赖配置人员保证同一 group 内的值一致（通常
        同一 group 服务同一模型，bytes_per_token 自然相同）。

    缺少配置时打印 warning 并将该 instance 的值设为 0（调用方需检查）。

    Returns:
        {instance_id: bytes_per_block}
    """
    with open(config_path, "r") as f:
        config_json = json.load(f)

    result: Dict[str, int] = {}
    for group in config_json.get("instance_groups", []):
        for inst in group.get("instances", []):
            bpt = inst.get("bytes_per_token", 0)
            bs = inst.get("block_size", 0)
            iid = inst.get("instance_id", "<unknown>")
            if bpt <= 0 or bs <= 0:
                print(
                    f"Warning: Instance '{iid}' is missing bytes_per_token or block_size "
                    "configuration. Storage will be displayed in blocks instead of GB."
                )
                result[iid] = 0
            else:
                result[iid] = bs * bpt
    return result


def has_hierarchical_storage(config_path: str) -> bool:
    """
    Return whether the optimizer config enables real multi-tier storage.

    A single storage entry can still produce Tier0 diagnostic columns in CSV,
    but per-tier comparison charts are only meaningful when a group has at
    least two configured tiers and hierarchical eviction is enabled.
    """
    with open(config_path, "r") as f:
        config_json = json.load(f)

    for group in config_json.get("instance_groups", []):
        if group_hierarchical_eviction_enabled(group) and len(group.get("storages", [])) > 1:
            return True
    return False


def extract_config_quota_gb_map(config_path: str) -> Dict[str, float]:
    """
    从 config JSON 提取每个 instance 所属 group 的 quota_capacity（单位：GB）。

    quota_capacity 是 group 级别的共享配额，同一 group 内所有 instances 共享该值。
    用于在 tradeoff 曲线上标注当前配置的实际部署容量位置（竖线参考线）。

    Returns:
        {instance_id: quota_capacity_gb}
        若某 group 未配置 quota_capacity，则其 instances 不会出现在结果中。
    """
    with open(config_path, "r") as f:
        config_json = json.load(f)

    result: Dict[str, float] = {}
    for group in config_json.get("instance_groups", []):
        quota_gb = group.get("quota_capacity")
        if quota_gb is None:
            continue
        for inst in group.get("instances", []):
            iid = inst.get("instance_id", "<unknown>")
            result[iid] = float(quota_gb)
    return result


# ============================================================================
# Warmup Pass
# ============================================================================

def warmup_pass(
    config_path: str,
    warmup_capacity: int,
    bytes_per_block_map: Dict[str, int],
    policy: str = None,
    enable_lifecycle_tracking: bool = False,
    enable_template_analysis: bool = False,
) -> int:
    """
    用大容量跑一遍，获取 group 内最大 block 数。

    Returns:
        max_blocks (int)
    """
    return warmup_pass_with_metrics(
        config_path,
        warmup_capacity,
        bytes_per_block_map,
        policy,
        enable_lifecycle_tracking,
        enable_template_analysis,
    )["max_blocks"]


def warmup_pass_with_metrics(
    config_path: str,
    warmup_capacity: int,
    bytes_per_block_map: Dict[str, int],
    policy: str = None,
    enable_lifecycle_tracking: bool = False,
    enable_template_analysis: bool = False,
) -> dict:
    """
    用大容量跑一遍，获取全量缓存容量和无限容量理论命中率。

    Returns:
        {
            "max_blocks": int,
            "instances": {
                instance_id: {
                    "total": float,
                    "local": float,
                    "remote": float,
                    "cached_gb": float,
                }
            },
        }
    """
    import pandas as pd

    print(f"Running warmup with capacity={warmup_capacity}...")
    temp_dir, manager = run_optimizer_with_config_explicit(
        config_path, warmup_capacity, policy,
        enable_lifecycle_tracking=enable_lifecycle_tracking,
        enable_template_analysis=enable_template_analysis,
    )

    try:
        csv_map = collect_instance_csvs(temp_dir)
        if not csv_map:
            raise RuntimeError("No CSV files found after warmup")

        max_blocks = 0
        instance_metrics = {}
        total_acc_write = 0
        for iid, csv_file in csv_map.items():
            df = pd.read_csv(csv_file)
            if df.empty:
                continue
            if "CachedBlocksAllInstances" not in df.columns:
                raise RuntimeError(f"{csv_file} missing required CachedBlocksAllInstances column")
            max_blocks = max(max_blocks, int(df["CachedBlocksAllInstances"].max()))
            total_acc_write = max(total_acc_write, int(df["AccWriteBlocks"].iloc[-1]) if "AccWriteBlocks" in df.columns else 0)
            bpb = bytes_per_block_map.get(iid, 0)
            metrics = parse_instance_metrics(csv_file, bpb)
            if metrics is None:
                continue
            instance_metrics[iid] = {
                "total": metrics["acc_total_hit_rate"],
                "local": metrics["acc_local_hit_rate"],
                "remote": metrics["acc_remote_hit_rate"],
                "cached_gb": metrics["cached_gb"],
            }

        rep_bpb = next(iter(bytes_per_block_map.values()), 0)
        max_gb = max_blocks * rep_bpb / (1024 ** 3) if rep_bpb > 0 else 0
        print(f"Warmup done. Max cached: {max_gb:.2f} GB ({max_blocks} blocks), AccWriteBlocks: {total_acc_write}")

        return {
            "max_blocks": max_blocks,
            "instances": instance_metrics,
        }
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
    bytes_per_block_map: Dict[str, int],
    save_csv_to: str = None,
    enable_lifecycle_tracking: bool = False,
    enable_template_analysis: bool = False,
) -> dict:
    """
    运行单个实验，返回结果字典。

    Returns:
        {
            "policy": str,
            "capacity": int,
            "instances": {instance_id: {"total": float, "local": float,
                                        "remote": float, "cached_gb": float}},
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
            enable_lifecycle_tracking, enable_template_analysis,
        )
        csv_map = collect_instance_csvs(temp_dir)
        if not csv_map:
            result["error"] = "No CSV files found"
            return result

        instance_metrics = {}
        for iid, csv_file in csv_map.items():
            bpb = bytes_per_block_map.get(iid, 0)
            metrics = parse_instance_metrics(csv_file, bpb)
            if metrics is None:
                continue
            instance_metrics[iid] = {
                "total": metrics["acc_total_hit_rate"],
                "local": metrics["acc_local_hit_rate"],
                "remote": metrics["acc_remote_hit_rate"],
                "cached_gb": metrics["cached_gb"],
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
    bytes_per_block_map: Dict[str, int],
    max_workers: int = 4,
    save_csv_dir: str = None,
    enable_lifecycle_tracking: bool = False,
    enable_template_analysis: bool = False,
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
            bytes_per_block_map,
            csv_subdir, enable_lifecycle_tracking,
            enable_template_analysis,
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
