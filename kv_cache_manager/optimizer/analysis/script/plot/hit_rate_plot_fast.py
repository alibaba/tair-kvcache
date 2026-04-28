#!/usr/bin/env python3
"""
命中率时序图绘制（高性能版本）

相比 hit_rate_plot.py 的优化点：
- 固定间隔时间网格替代全时间戳并集，避免 base 随 instance 数量膨胀
- np.searchsorted 替代 pd.merge_asof，全向量化 ZOH 对齐
- window_hit_rate 全向量化，消除 Python for-loop
- 绘图前预降采样，减少 matplotlib 渲染点数

图表内容与原版一致：
- 上图：累计命中率（AccHitRate / AccExternalHitRate / AccRemoteHitRate）
- 下图：瞬时命中率（时间窗口内累积量差值）
- 左轴：InstanceGroup Storage
"""

import glob
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def read_csv_file(csv_file_path):
    try:
        df = pd.read_csv(csv_file_path, comment='#')
        df.columns = df.columns.str.strip()
        return df
    except Exception as e:
        print(f"Error reading {csv_file_path}: {str(e)}")
        return None


def _zoh_align(src_t, src_cols, grid_t):
    """
    Zero-order hold 对齐：对 grid_t 中每个点取 src_t 中 <= 它的最后一行。
    src_cols: (n_src, n_col) ndarray
    返回 (n_grid, n_col) ndarray，无对应值处为 nan。
    """
    idx = np.searchsorted(src_t, grid_t, side='right') - 1
    n_col = src_cols.shape[1] if src_cols.ndim == 2 else 1
    out = np.full((len(grid_t), n_col), np.nan)
    valid = idx >= 0
    out[valid] = src_cols[idx[valid]]
    return out


def _window_hit_rate_vec(src_t, src_acc_hit, src_acc_read,
                         eval_t, window_s):
    """
    全向量化窗口命中率：在 eval_t 处求值。

    对 eval_t[i]：
      end = src 中 <= eval_t[i] 的最后一条
      beg = src 中 >= (eval_t[i] - window_s) 的第一条（同段内）
      rate = (hit[end] - hit[beg]) / (read[end] - read[beg])

    段边界：相邻真实上报间隔 > window_s 时视为不连续。
    """
    n_eval = len(eval_t)
    rate = np.full(n_eval, np.nan)

    # 过滤有效上报点
    mask = ~(np.isnan(src_acc_read) | np.isnan(src_acc_hit))
    if not np.any(mask):
        return rate
    rt = src_t[mask]
    rh = src_acc_hit[mask]
    rr = src_acc_read[mask]

    if len(rt) == 0:
        return rate

    # 段划分
    gaps = np.diff(rt)
    seg_ids = np.zeros(len(rt), dtype=np.int32)
    seg_ids[1:] = np.cumsum(gaps > window_s)
    _, first_of_seg = np.unique(seg_ids, return_index=True)
    seg_start_lut = np.empty(seg_ids[-1] + 1, dtype=np.int32)
    seg_start_lut[_] = first_of_seg

    # end: eval_t 对应的最近真实点（ZOH）
    end_idx = np.searchsorted(rt, eval_t, side='right') - 1

    # 只处理 end_idx >= 0 的点
    ok = end_idx >= 0
    ei = end_idx[ok]

    # beg: eval_t - window_s 对应的最早真实点
    t_left = eval_t[ok] - window_s
    beg_idx = np.searchsorted(rt, t_left, side='left')
    beg_idx = np.clip(beg_idx, 0, len(rt) - 1)

    # 不越过段边界
    end_seg = seg_ids[ei]
    seg_floor = seg_start_lut[end_seg]
    beg_idx = np.maximum(beg_idx, seg_floor)

    # 计算差值
    delta_read = rr[ei] - rr[beg_idx]
    delta_hit = rh[ei] - rh[beg_idx]

    has_data = delta_read > 0
    result = np.full(len(ei), np.nan)
    result[has_data] = np.clip(delta_hit[has_data] / delta_read[has_data], 0.0, 1.0)

    rate[ok] = result
    return rate


def plot_multi_instance_analysis(csv_dir,
                                 grid_interval_s=1.0,
                                 plot_interval_s=5.0,
                                 window_s=10.0):
    """
    Parameters
    ----------
    csv_dir : str
        含 *_hit_rates.csv 的目录
    grid_interval_s : float
        ZOH 对齐的时间网格间隔（秒）。1 秒足以保留细节。
    plot_interval_s : float
        绘图降采样间隔（秒）。每 plot_interval_s 画一个点。
    window_s : float
        瞬时命中率的滑动窗口宽度（秒）。
    """
    csv_files = sorted(glob.glob(os.path.join(csv_dir, "*_hit_rates.csv")))
    if not csv_files:
        print(f"Error: No CSV files found in directory: {csv_dir}")
        return

    # ---- 1. 读取 CSV ----
    raw_instances = []        # list of (name, t_sec, col_dict)
    has_remote = True

    for csv_file in csv_files:
        df = read_csv_file(csv_file)
        if df is None:
            continue
        num_cols = ['TimestampUs', 'CachedBlocksAllInstance',
                    'AccHitRate', 'AccExternalHitRate', 'AccReadBlocks']
        for c in num_cols:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors='coerce')
        df = df.dropna(subset=['TimestampUs']).sort_values('TimestampUs')
        if df.empty:
            continue

        missing = [c for c in num_cols if c not in df.columns]
        if missing:
            print(f"Warning: {csv_file} missing {missing}, skipped")
            continue

        if 'AccRemoteHitRate' not in df.columns:
            has_remote = False

        name = os.path.splitext(os.path.basename(csv_file))[0]
        raw_instances.append((name, df))

    if not raw_instances:
        print("Error: No valid CSV data could be loaded")
        return

    n_inst = len(raw_instances)

    # ---- 2. 全局时间基准 ----
    min_ts = min(df['TimestampUs'].iloc[0] for _, df in raw_instances)
    max_ts = max(df['TimestampUs'].iloc[-1] for _, df in raw_instances)
    t_end = (max_ts - min_ts) / 1e6

    grid_t = np.arange(0.0, t_end + grid_interval_s, grid_interval_s)
    n_grid = len(grid_t)
    print(f"Instances: {n_inst}, Time span: {t_end:.1f}s, "
          f"Grid points: {n_grid} (interval={grid_interval_s}s)")

    # ---- 3. 逐 instance 对齐到网格 ----
    # 预分配 (n_inst, n_grid) 大矩阵
    acc_hit_rate_all = np.full((n_inst, n_grid), np.nan)
    acc_ext_rate_all = np.full((n_inst, n_grid), np.nan)
    acc_read_all = np.full((n_inst, n_grid), np.nan)
    acc_hit_bl_all = np.full((n_inst, n_grid), np.nan)
    acc_ext_bl_all = np.full((n_inst, n_grid), np.nan)
    acc_remote_bl_all = np.full((n_inst, n_grid), np.nan) if has_remote else None

    time_ranges = []
    names = []
    global_t_parts = []
    global_storage_parts = []

    for i, (name, df) in enumerate(raw_instances):
        t = (df['TimestampUs'].to_numpy(float) - min_ts) / 1e6
        t0, t1 = t[0], t[-1]
        time_ranges.append((t0, t1))
        names.append(name)

        hr = df['AccHitRate'].to_numpy(float)
        er = df['AccExternalHitRate'].to_numpy(float)
        rb = df['AccReadBlocks'].to_numpy(float)
        hb = hr * rb
        eb = er * rb

        if has_remote:
            rr = df['AccRemoteHitRate'].to_numpy(float)
            rmb = rr * rb
            cols = np.column_stack([hr, er, rb, hb, eb, rmb])
        else:
            cols = np.column_stack([hr, er, rb, hb, eb])

        aligned = _zoh_align(t, cols, grid_t)  # (n_grid, ncol)

        # 掩盖 instance 尚未开始的区间
        before = grid_t < t0
        aligned[before] = np.nan

        acc_hit_rate_all[i] = aligned[:, 0]
        acc_ext_rate_all[i] = aligned[:, 1]
        acc_read_all[i] = aligned[:, 2]
        acc_hit_bl_all[i] = aligned[:, 3]
        acc_ext_bl_all[i] = aligned[:, 4]
        if has_remote:
            acc_remote_bl_all[i] = aligned[:, 5]

        # 全局存储
        storage = df['CachedBlocksAllInstance'].to_numpy(float)
        global_t_parts.append(t)
        global_storage_parts.append(storage)

    # ---- 4. 全局存储 ZOH 对齐 ----
    g_t = np.concatenate(global_t_parts)
    g_v = np.concatenate(global_storage_parts)
    order = np.argsort(g_t, kind='mergesort')
    g_t, g_v = g_t[order], g_v[order]

    idx = np.searchsorted(g_t, grid_t, side='right') - 1
    total_storage = np.full(n_grid, np.nan)
    ok = idx >= 0
    total_storage[ok] = g_v[idx[ok]]

    # ---- 5. 计算瞬时命中率（在原始数据上计算，只在降采样点求值） ----
    ds_step = max(1, int(round(plot_interval_s / grid_interval_s)))
    ds_idx = np.arange(0, n_grid, ds_step)
    ds_t = grid_t[ds_idx]
    n_ds = len(ds_idx)
    print(f"Plot points per trace: {n_ds} (interval={plot_interval_s}s)")

    # 瞬时命中率直接在原始数据上计算，在 ds_t 处求值
    inst_hit_ds = np.full((n_inst, n_ds), np.nan)
    inst_ext_ds = np.full((n_inst, n_ds), np.nan)
    inst_remote_ds = np.full((n_inst, n_ds), np.nan) if has_remote else None

    for i, (name, df) in enumerate(raw_instances):
        t = (df['TimestampUs'].to_numpy(float) - min_ts) / 1e6
        rb = df['AccReadBlocks'].to_numpy(float)
        hr = df['AccHitRate'].to_numpy(float)
        er = df['AccExternalHitRate'].to_numpy(float)
        hb = hr * rb
        eb = er * rb

        inst_hit_ds[i] = _window_hit_rate_vec(t, hb, rb, ds_t, window_s)
        inst_ext_ds[i] = _window_hit_rate_vec(t, eb, rb, ds_t, window_s)

        if has_remote:
            rr = df['AccRemoteHitRate'].to_numpy(float)
            rmb = rr * rb
            inst_remote_ds[i] = _window_hit_rate_vec(t, rmb, rb, ds_t, window_s)

    # ---- 6. 画图 ----
    ds_storage = total_storage[ds_idx]

    fig, (ax_top, ax_bot) = plt.subplots(
        2, 1, figsize=(16, 20), sharex=True,
        gridspec_kw={'height_ratios': [1, 1], 'hspace': 0.12}
    )

    y_upper = np.nanmax(total_storage) * 1.15 if np.any(np.isfinite(total_storage)) else 1

    for ax in (ax_top, ax_bot):
        ax.set_ylabel('InstanceGroup Storage', color='#1f77b4', fontsize=12)
        ax.plot(ds_t, ds_storage, color='#1f77b4',
                label='InstanceGroup Storage', linewidth=2.2, alpha=0.9,
                drawstyle='steps-post')
        ax.set_ylim(0, y_upper)
        ax.tick_params(axis='y', labelcolor='#1f77b4')
        ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)

    ax_top_r = ax_top.twinx()
    ax_top_r.set_ylabel('Cumulative Hit Rate', color='#d62728', fontsize=12)
    ax_top_r.set_ylim(0, 1)
    ax_top_r.tick_params(axis='y', labelcolor='#d62728')

    ax_bot_r = ax_bot.twinx()
    ax_bot_r.set_ylabel('Instant Hit Rate (Per-trace)', color='#d62728', fontsize=12)
    ax_bot_r.set_ylim(0, 1)
    ax_bot_r.tick_params(axis='y', labelcolor='#d62728')

    ax_bot.set_xlabel('Timestamp (s)', fontsize=12)
    ax_bot.set_xlim(0, t_end * 1.05)

    colors = plt.cm.tab20(np.linspace(0.3, 0.9, n_inst))

    # 上图：降采样后的累计命中率
    ds_acc_hit = acc_hit_rate_all[:, ds_idx]     # (n_inst, n_ds)
    ds_acc_ext = acc_ext_rate_all[:, ds_idx]
    if has_remote:
        ds_acc_read = acc_read_all[:, ds_idx]
        ds_remote_bl = acc_remote_bl_all[:, ds_idx]

    for i in range(n_inst):
        t0, t1 = time_ranges[i]
        m = (ds_t >= t0) & (ds_t <= t1)

        ax_top_r.plot(ds_t[m], ds_acc_hit[i, m],
                      color=colors[i], linewidth=1.2, alpha=0.85,
                      drawstyle='steps-post')
        ax_top_r.plot(ds_t[m], ds_acc_ext[i, m],
                      color=colors[i], linestyle='--', alpha=0.6,
                      linewidth=0.8, drawstyle='steps-post')
        if has_remote:
            rd = ds_acc_read[i, m]
            remote_rate = np.where(rd > 0, ds_remote_bl[i, m] / rd, np.nan)
            ax_top_r.plot(ds_t[m], remote_rate,
                          color=colors[i], linestyle=':', alpha=0.7,
                          linewidth=1.0, drawstyle='steps-post')

    # 下图：降采样后的瞬时命中率
    for i in range(n_inst):
        t0, t1 = time_ranges[i]
        m = (ds_t >= t0) & (ds_t <= t1)

        ax_bot_r.plot(ds_t[m], inst_hit_ds[i, m],
                      color=colors[i], linewidth=1.0, alpha=0.85)
        ax_bot_r.plot(ds_t[m], inst_ext_ds[i, m],
                      color=colors[i], linestyle='--', alpha=0.6,
                      linewidth=0.8)
        if has_remote:
            ax_bot_r.plot(ds_t[m], inst_remote_ds[i, m],
                          color=colors[i], linestyle=':', alpha=0.7,
                          linewidth=0.8)

    ax_top.tick_params(axis='x', labelbottom=True)
    ax_top.set_xlabel('Timestamp (s)', fontsize=12)
    ax_top.set_title(f'Cache Analysis - {n_inst} Instances',
                     fontsize=15, fontweight='bold', pad=12)

    fig.tight_layout()
    output_file = os.path.join(csv_dir, "multi_instance_cache_analysis.png")
    plt.savefig(output_file, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"Chart saved to: {output_file}")
    plt.close()


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python hit_rate_plot_fast.py <csv_dir> "
              "[grid_interval_s] [plot_interval_s] [window_s]")
        sys.exit(1)
    csv_dir = sys.argv[1]
    grid_s = float(sys.argv[2]) if len(sys.argv) > 2 else 1.0
    plot_s = float(sys.argv[3]) if len(sys.argv) > 3 else 5.0
    win_s = float(sys.argv[4]) if len(sys.argv) > 4 else 10.0
    plot_multi_instance_analysis(csv_dir, grid_s, plot_s, win_s)
