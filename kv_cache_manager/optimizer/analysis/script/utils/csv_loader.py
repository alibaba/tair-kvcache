#!/usr/bin/env python3
"""
CSV 数据加载层

职责：
- 收集 optimizer 输出目录里的 hit_rates CSV
- 解析单个 instance 的指标（取最后一行累计值）
- 容量列表生成（指数分布采样）
- 从已有 CSV 目录加载 Pareto 曲线数据（--skip-run 模式）
"""

import os
import re
from typing import Dict, List, Optional

import numpy as np
import pandas as pd


# ============================================================================
# 基础 CSV 读取
# ============================================================================

def collect_instance_csvs(output_dir: str) -> Dict[str, str]:
    """
    扫描输出目录，收集所有 *_hit_rates.csv 文件。

    Returns:
        {instance_id: csv_file_path}
    """
    if not os.path.exists(output_dir):
        return {}
    return {
        fname.replace("_hit_rates.csv", ""): os.path.join(output_dir, fname)
        for fname in os.listdir(output_dir)
        if fname.endswith("_hit_rates.csv")
    }


def parse_instance_metrics(csv_file: str, bytes_per_block: int) -> Optional[dict]:
    """
    从单个 instance CSV 解析累计指标（取最后一行）。

    Args:
        csv_file:        hit_rates CSV 路径
        bytes_per_block: 每个 block 的字节数（= block_size × bytes_per_token），必须 > 0

    Returns:
        {"acc_total_hit_rate", "acc_local_hit_rate",
         "acc_remote_hit_rate", "cached_blocks_all", "cached_gb",
         "tier_names": [...], "acc_tier_hit_rates": {...}, "tier_block_nums": {...}}
        或 None（文件为空时）
    """
    df = pd.read_csv(csv_file)
    if df.empty:
        return None
    last = df.iloc[-1]
    cached_blocks = int(last["CachedBlocksAllInstance"])
    result = {
        "acc_total_hit_rate": float(last["AccHitRate"]),
        "acc_local_hit_rate": float(last["AccLocalHitRate"]),
        "acc_remote_hit_rate": float(last["AccRemoteHitRate"]),
        "cached_blocks_all": cached_blocks,
        "cached_gb": cached_blocks * bytes_per_block / (1024 ** 3) if bytes_per_block > 0 else 0,
    }
    
    # 解析 per-tier 数据
    tier_names = []
    acc_tier_hit_rates = {}
    tier_block_nums = {}
    
    # 从列名中提取 tier 信息
    for col in df.columns:
        if col.startswith("AccTier") and col.endswith("_HitRate"):
            # 提取 tier 名称: AccTier0(GPU)_HitRate -> GPU
            match = re.search(r'AccTier\d+\(([^)]+)\)_HitRate', col)
            if match:
                tier_name = match.group(1)
                tier_names.append(tier_name)
                acc_tier_hit_rates[tier_name] = float(last[col])
        elif col.startswith("Tier") and col.endswith("_BlockNum"):
            # 提取 tier 名称: Tier0(GPU)_BlockNum -> GPU
            match = re.search(r'Tier\d+\(([^)]+)\)_BlockNum', col)
            if match:
                tier_name = match.group(1)
                tier_block_nums[tier_name] = int(last[col])
    
    if tier_names:
        result["tier_names"] = tier_names
        result["acc_tier_hit_rates"] = acc_tier_hit_rates
        result["tier_block_nums"] = tier_block_nums
    
    return result


def _read_hit_rates_from_csv(csv_path: str, bytes_per_block: int) -> Optional[dict]:
    """
    读取单个 hit_rates CSV，兼容 Acc* 和非 Acc* 列名。

    Args:
        csv_path:        hit_rates CSV 路径
        bytes_per_block: 每个 block 的字节数（= block_size × bytes_per_token），必须 > 0

    Returns:
        {"total", "local", "remote", "cached_blocks_all", "cached_gb"}
        或 None
    """
    try:
        df = pd.read_csv(csv_path)
        if df.empty:
            return None
        last = df.iloc[-1]

        def _get(col_acc, col_fallback):
            if col_acc in df.columns:
                return float(last[col_acc])
            if col_fallback in df.columns:
                return float(last[col_fallback])
            return 0.0

        cached = int(last["CachedBlocksAllInstance"]) if "CachedBlocksAllInstance" in df.columns else 0
        return {
            "total": _get("AccHitRate", "HitRate"),
            "local": _get("AccLocalHitRate", "LocalHitRate"),
            "remote": _get("AccRemoteHitRate", "RemoteHitRate"),
            "cached_blocks_all": cached,
            "cached_gb": cached * bytes_per_block / (1024 ** 3) if bytes_per_block > 0 else 0,
        }
    except Exception as e:
        print(f"  Warning: Failed to read {csv_path}: {e}")
        return None


# ============================================================================
# 容量列表生成
# ============================================================================

def generate_capacity_list(
    max_blocks: int,
    num_points: int,
    min_capacity: int = 2000,
) -> List[int]:
    """
    指数分布采样容量列表，从小到大排序。

    Args:
        max_blocks:   warmup 获取的最大 block 数
        num_points:   采样点数
        min_capacity: 最小容量阈值（小于此值的点丢弃）
    """
    x = np.linspace(-4, 4, num_points)
    ratios = np.exp(x) / np.exp(4)
    return sorted({
        int(max_blocks * r)
        for r in ratios
        if int(max_blocks * r) > min_capacity
    })


# ============================================================================
# --skip-run 模式：从已有 CSV 目录加载 Pareto 曲线数据
# ============================================================================

def _parse_cap_dirname(dirname: str):
    """
    解析 cap_<capacity>_<policy> 目录名。

    Returns:
        (capacity: int, policy: str or None)
    """
    parts = dirname.split("_")
    if len(parts) < 2:
        return None, None
    try:
        capacity = int(parts[1])
    except ValueError:
        return None, None
    policy = "_".join(parts[2:]) if len(parts) > 2 else None
    return capacity, policy


def load_results_from_csv_dir(
    csv_save_dir: str,
    bytes_per_block_map: Dict[str, int],
) -> Dict[str, List[dict]]:
    """
    扫描 csv_save_dir/cap_<capacity>_<policy>/ 子目录，
    构建按策略分组的结果。

    Args:
        csv_save_dir:      保存 CSV 子目录的根目录
        bytes_per_block_map: {instance_id: bytes_per_block}，必须覆盖所有 instance

    Returns:
        {"policy_name": [{"capacity": int, "instances": {...}}, ...]}
        单策略时 key 为解析出的策略名或 "default_policy"。
        各策略的列表已按 capacity 升序排序。
    """
    if not os.path.exists(csv_save_dir):
        print(f"Error: CSV directory not found: {csv_save_dir}")
        return {}

    cap_dirs = [
        d for d in sorted(os.listdir(csv_save_dir))
        if os.path.isdir(os.path.join(csv_save_dir, d)) and d.startswith("cap_")
    ]

    if not cap_dirs:
        print(f"Error: No cap_* directories found in {csv_save_dir}")
        return {}

    print(f"Found {len(cap_dirs)} CSV directories\n")
    results_by_policy: Dict[str, List[dict]] = {}

    for dirname in cap_dirs:
        cap_dir = os.path.join(csv_save_dir, dirname)
        capacity, policy = _parse_cap_dirname(dirname)
        if capacity is None:
            print(f"Warning: Cannot parse capacity from {dirname}, skipping")
            continue

        policy = policy or "default_policy"
        print(f"Loading {policy} capacity={capacity} from {dirname}...")
        instances = {}

        for fname in os.listdir(cap_dir):
            if not fname.endswith(".csv"):
                continue
            instance_id = fname.replace("_hit_rates.csv", "")
            bpb = bytes_per_block_map.get(instance_id, 0)
            metrics = _read_hit_rates_from_csv(os.path.join(cap_dir, fname), bpb)
            if metrics:
                instances[instance_id] = metrics

        if instances:
            results_by_policy.setdefault(policy, []).append(
                {"capacity": capacity, "instances": instances}
            )
            print(f"  ✓ {len(instances)} instances loaded\n")
        else:
            print(f"  ✗ No valid data\n")

    for pol in results_by_policy:
        results_by_policy[pol].sort(key=lambda x: x["capacity"])

    total = sum(len(v) for v in results_by_policy.values())
    print(f"Loaded {total} capacity points across {len(results_by_policy)} policies\n")
    return results_by_policy
