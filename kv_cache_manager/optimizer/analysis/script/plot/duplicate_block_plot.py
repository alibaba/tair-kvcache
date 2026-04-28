#!/usr/bin/env python3
"""
全局重复 Block 时序图绘制

读取 global_duplicate_blocks.csv，绘制总 block 数量和重复 block 数量随时间变化的曲线。
"""

import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_duplicate_blocks(csv_dir):
    """
    绘制全局重复 block 时序图。

    Parameters
    ----------
    csv_dir : str
        含 global_duplicate_blocks.csv 的输出目录
    """
    csv_path = os.path.join(csv_dir, "global_duplicate_blocks.csv")
    if not os.path.exists(csv_path):
        print(f"Duplicate blocks CSV not found: {csv_path}, skipping")
        return

    try:
        df = pd.read_csv(csv_path, comment='#')
        df.columns = df.columns.str.strip()
    except Exception as e:
        print(f"Error reading {csv_path}: {e}")
        return

    required = ['TimestampUs', 'TotalBlockCopies', 'DuplicateBlockCopies']
    missing = [c for c in required if c not in df.columns]
    if missing:
        print(f"Missing columns {missing} in {csv_path}, skipping")
        return

    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors='coerce')
    df = df.dropna(subset=['TimestampUs']).sort_values('TimestampUs')

    if df.empty:
        print("No data in duplicate blocks CSV, skipping")
        return

    # 转换为秒
    t0 = df['TimestampUs'].iloc[0]
    t_sec = (df['TimestampUs'].to_numpy(float) - t0) / 1e6
    total_copies = df['TotalBlockCopies'].to_numpy(float)
    dup_copies = df['DuplicateBlockCopies'].to_numpy(float)
    unique_keys = df['TotalUniqueKeys'].to_numpy(float) if 'TotalUniqueKeys' in df.columns else None

    # 降采样：数据点过多时按间隔采样
    max_points = 5000
    if len(t_sec) > max_points:
        step = len(t_sec) // max_points
        idx = np.arange(0, len(t_sec), step)
        # 确保最后一个点被包含
        if idx[-1] != len(t_sec) - 1:
            idx = np.append(idx, len(t_sec) - 1)
        t_sec = t_sec[idx]
        total_copies = total_copies[idx]
        dup_copies = dup_copies[idx]
        if unique_keys is not None:
            unique_keys = unique_keys[idx]

    # ---- 绘图 ----
    fig, ax = plt.subplots(figsize=(16, 8))

    ax.plot(t_sec, total_copies, color='#1f77b4', linewidth=1.5,
            label='TotalBlockCopies', drawstyle='steps-post')
    ax.plot(t_sec, dup_copies, color='#d62728', linewidth=1.5,
            label='DuplicateBlockCopies', drawstyle='steps-post')
    if unique_keys is not None:
        ax.plot(t_sec, unique_keys, color='#2ca02c', linewidth=1.5,
                linestyle='--', label='TotalUniqueKeys', drawstyle='steps-post')

    ax.set_xlabel('Timestamp (s)', fontsize=12)
    ax.set_ylabel('Block Count', fontsize=12)
    ax.set_title('Global Duplicate Blocks Over Time', fontsize=15, fontweight='bold', pad=12)
    ax.legend(fontsize=11, loc='upper left')
    ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    ax.set_xlim(0, t_sec[-1] * 1.02)
    ax.set_ylim(bottom=0)

    fig.tight_layout()
    output_file = os.path.join(csv_dir, "global_duplicate_blocks.png")
    plt.savefig(output_file, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"Duplicate blocks chart saved to: {output_file}")
    plt.close()


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python duplicate_block_plot.py <csv_dir>")
        sys.exit(1)
    plot_duplicate_blocks(sys.argv[1])
