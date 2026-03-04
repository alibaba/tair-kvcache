#!/usr/bin/env python3
"""
Block Lifecycle 分析核心

职责：
- 全量读取 lifecycle CSV
- 计算统计指标（lifespan / access / revival）
- 格式化打印报告
- 返回排序数组供绘图层使用

被 run/analyze_lifecycle.py 调用，不含参数解析和绘图逻辑。
"""

import os
import glob
from typing import Dict, Optional

import numpy as np
import pandas as pd


# ============================================================================
# 数据读取
# ============================================================================

def read_lifecycle_csv(csv_path: str) -> Optional[pd.DataFrame]:
    """读取 lifecycle CSV 并预处理，返回 DataFrame（失败返回 None）"""
    try:
        df = pd.read_csv(csv_path, comment="#")
        df.columns = df.columns.str.strip()

        for col in ["BlockKey", "BirthTimeUs", "DeathTimeUs", "LifespanUs",
                     "AccessCount", "LastAccessTimeUs"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        if "IsAlive" in df.columns:
            df["IsAlive"] = df["IsAlive"].astype(str).str.strip().str.lower() == "true"

        for col in ["BirthTimeUs", "DeathTimeUs", "LifespanUs", "LastAccessTimeUs"]:
            if col in df.columns:
                df[col[:-2] + "S"] = df[col] / 1e6

        if "LastAccessTimeUs" in df.columns and "BirthTimeUs" in df.columns:
            df["ActiveLifespanS"] = (df["LastAccessTimeUs"] - df["BirthTimeUs"]) / 1e6

        return df
    except Exception as e:
        print(f"错误: 无法读取 {csv_path}: {e}")
        return None


# ============================================================================
# 统计计算
# ============================================================================

def compute_statistics(df: pd.DataFrame) -> Dict:
    """计算全维度统计指标"""
    stats = {}
    total = len(df)
    stats["total_blocks"] = total

    def _dist(series, prefix):
        s = series.dropna()
        if len(s) == 0:
            return
        stats[f"{prefix}_mean"] = float(s.mean())
        stats[f"{prefix}_median"] = float(s.median())
        stats[f"{prefix}_std"] = float(s.std())
        stats[f"{prefix}_min"] = float(s.min())
        stats[f"{prefix}_max"] = float(s.max())
        for p in [25, 75, 90, 95, 99]:
            stats[f"{prefix}_p{p}"] = float(s.quantile(p / 100))

    _dist(df["LifespanS"], "lifespan")
    _dist(df["ActiveLifespanS"], "active_lifespan")
    _dist(df["AccessCount"], "access")

    stats["zero_lifespan_blocks"] = int((df["LifespanS"] == 0).sum())
    stats["zero_lifespan_ratio"] = stats["zero_lifespan_blocks"] / total if total > 0 else 0

    stats["zero_access_blocks"] = int((df["AccessCount"] == 0).sum())
    stats["zero_access_ratio"] = stats["zero_access_blocks"] / total if total > 0 else 0

    revival = df.groupby("BlockKey").size()
    unique = len(revival)
    revived = int((revival > 1).sum())
    stats["unique_block_keys"] = unique
    stats["revived_block_keys"] = revived
    stats["revival_ratio"] = revived / unique if unique > 0 else 0
    stats["max_revivals"] = int(revival.max() - 1) if unique > 0 else 0

    stats["utilization_rate"] = (
        stats["active_lifespan_mean"] / stats["lifespan_mean"]
        if stats.get("lifespan_mean", 0) > 0 else 0.0
    )

    return stats


# ============================================================================
# 绘图数据提取（排序数组，供 plot 层直接使用）
# ============================================================================

def extract_plot_data(df: pd.DataFrame) -> Dict[str, np.ndarray]:
    """从 DataFrame 提取排序后的数组，供绘图层使用"""
    evicted_mask = ~df["IsAlive"] if "IsAlive" in df.columns else pd.Series(False, index=df.index)
    return {
        "physical_all": np.sort(df["LifespanS"].dropna().values),
        "physical_evicted": np.sort(df.loc[evicted_mask, "LifespanS"].dropna().values),
        "active_all": np.sort(df["ActiveLifespanS"].dropna().values),
        "access_counts": df["AccessCount"].dropna().values,
    }


# ============================================================================
# 报告打印
# ============================================================================

def print_statistics(stats: Dict, instance_name: str):
    """格式化打印统计报告"""
    sep = "=" * 80
    print(f"\n{sep}")
    print(f"  Block Lifecycle 分析报告 - {instance_name}")
    print(f"{sep}\n")

    print("【总体概览】")
    print(f"  总 Block 数:        {stats['total_blocks']:>12,}")
    print(f"  唯一 BlockKey 数:   {stats['unique_block_keys']:>12,}")
    print(f"  复活 BlockKey 数:   {stats['revived_block_keys']:>12,}  (复活率: {stats['revival_ratio']*100:.1f}%)")
    print(f"  最多复活次数:       {stats['max_revivals']:>12}")
    print(f"  即逐 Block:         {stats['zero_lifespan_blocks']:>12,}  ({stats['zero_lifespan_ratio']*100:.1f}%)  写入即驱逐")
    print(f"  零访问 Block:       {stats['zero_access_blocks']:>12,}  ({stats['zero_access_ratio']*100:.1f}%)")
    print(f"  缓存利用率:         {stats['utilization_rate']*100:>11.1f}%  (Active/Total)")

    def _print_dist(label, prefix, unit="s"):
        print(f"\n【{label}】")
        for name, key in [("均值", "mean"), ("中位数", "median"), ("标准差", "std"),
                          ("最小值", "min"), ("最大值", "max"),
                          ("P25", "p25"), ("P75", "p75"), ("P90", "p90"),
                          ("P95", "p95"), ("P99", "p99")]:
            k = f"{prefix}_{key}"
            if k in stats:
                print(f"  {name:<8} {stats[k]:>12.2f}{unit}")

    _print_dist("Physical Lifespan - 秒", "lifespan")
    _print_dist("Active Lifespan - 秒", "active_lifespan")
    _print_dist("Access Count", "access", unit="")

    print(f"\n{sep}\n")


# ============================================================================
# 批量分析
# ============================================================================

def find_lifecycle_csvs(input_path: str):
    """根据输入路径返回 CSV 文件列表（支持单文件或目录）"""
    if os.path.isfile(input_path):
        return [input_path]
    return sorted(glob.glob(os.path.join(input_path, "*_lifecycle.csv")))
