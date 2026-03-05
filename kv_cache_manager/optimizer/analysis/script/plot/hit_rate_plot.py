#!/usr/bin/env python3
"""
命中率时序图绘制

职责：
- 读取多个 instance 的 hit_rates CSV
- 将各 instance 的命中率对齐到统一时间轴（ZOH 插值）
- 双子图：累计命中率 + 瞬时命中率（平滑）
- plot_multi_instance_analysis() 可被其他脚本直接 import

被 run/optimizer_run.py 和 run/tradeoff_by_*.py 调用。
"""

import glob
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd



def read_csv_file(csv_file_path):
    """
    读取单个CSV文件并返回DataFrame，包含错误处理
    """
    try:
        df = pd.read_csv(csv_file_path, comment='#')
        df.columns = df.columns.str.strip()
        return df
    except Exception as e:
        print(f"Error reading {csv_file_path}: {str(e)}")
        return None

def plot_multi_instance_analysis(csv_dir):
    csv_files = sorted(glob.glob(os.path.join(csv_dir, "*_hit_rates.csv")))
    if not csv_files:
        print(f"Error: No CSV files found in directory: {csv_dir}")
        return

    dataframes, instance_names = [], []
    for csv_file in csv_files:
        df = read_csv_file(csv_file)
        if df is None:
            continue

        # 数值化 + 排序 
        for c in ['TimestampUs', 'CachedBlocksAllInstance', 'HitRate', 'ExternalHitRate', 'AccHitRate', 'AccExternalHitRate']:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors='coerce')
        df = df.dropna(subset=['TimestampUs']).sort_values('TimestampUs')


        dataframes.append(df)
        instance_names.append(os.path.splitext(os.path.basename(csv_file))[0])

    if not dataframes:
        print("Error: No valid CSV data could be loaded")
        return

    required_cols = ['TimestampUs', 'CachedBlocksAllInstance', 'HitRate', 'ExternalHitRate', 'AccHitRate', 'AccExternalHitRate']
    for i, df in enumerate(dataframes):
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            print(f"Error: {instance_names[i]} is missing required columns: {missing}")
            return

    # 全局基准：最早起点
    min_timestamp = min(df['TimestampUs'].iloc[0] for df in dataframes)

    # “每个trace都画”：基准时间轴取所有instance的时间戳并集（秒）
    all_t = []
    for df in dataframes:
        all_t.append(((df['TimestampUs'] - min_timestamp) / 1e6).to_numpy())
    base_timestamps = np.unique(np.concatenate(all_t))
    base = pd.DataFrame({'t': base_timestamps})  # 用于merge_asof

    all_hit, all_external_hit, all_acc_hit, all_acc_external_hit, all_time_ranges = [], [], [], [], []
    global_updates_list = []
    for df in dataframes:
        d = df.copy()
        d['t'] = (d['TimestampUs'] - min_timestamp) / 1e6
        d = d.sort_values('t')

        global_updates_list.append(d[['t', 'CachedBlocksAllInstance']])
        t0, t1 = d['t'].iloc[0], d['t'].iloc[-1]
        all_time_ranges.append((t0, t1))

        # 真实对齐：取 <=t 的最后一次上报（ZOH），不插值
        aligned = pd.merge_asof(
            base,
            d[['t', 'HitRate', 'ExternalHitRate', 'AccHitRate', 'AccExternalHitRate']],
            on='t',
            direction='backward',
            allow_exact_matches=True
        )


        mask_out = (aligned['t'] < t0)
        aligned.loc[mask_out, ['HitRate', 'ExternalHitRate', 'AccHitRate', 'AccExternalHitRate']] = np.nan
        
        all_hit.append(aligned['HitRate'].to_numpy(float))
        all_external_hit.append(aligned['ExternalHitRate'].to_numpy(float))
        all_acc_hit.append(aligned['AccHitRate'].to_numpy(float))
        all_acc_external_hit.append(aligned['AccExternalHitRate'].to_numpy(float))

    global_updates = pd.concat(global_updates_list, ignore_index=True)
    global_updates = global_updates.dropna(subset=['t', 'CachedBlocksAllInstance']).sort_values('t')

    # 同一时刻可能多个instance都写了全局容量：聚合成一个值（median更稳健）
    global_updates = (global_updates
                    .groupby('t', as_index=False)['CachedBlocksAllInstance']
                    .median())

    # 对齐到base：在两次更新之间保持最后值（全局容量是状态量）
    global_aligned = pd.merge_asof(
        base,
        global_updates,
        on='t',
        direction='backward',
        allow_exact_matches=True
    )

    total_storage = global_aligned['CachedBlocksAllInstance'].to_numpy(float)    

    # ---- 画图 ----
    fig, (ax_top, ax_bot) = plt.subplots(
        2, 1, figsize=(16, 20), sharex=True,
        gridspec_kw={'height_ratios': [1, 1], 'hspace': 0.12}
    )

    def setup_left_axis(ax):
        ax.set_ylabel('InstanceGroup Storage', color='#1f77b4', fontsize=12)
        ax.plot(base_timestamps, total_storage, color='#1f77b4',
                label='InstanceGroup Storage', linewidth=2.2, alpha=0.9,
                drawstyle='steps-post')
        y_upper = np.nanmax(total_storage) * 1.15 if np.any(~np.isnan(total_storage)) else 1
        ax.set_ylim(0, y_upper)
        ax.tick_params(axis='y', labelcolor='#1f77b4')
        ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)

    def setup_right_axis(ax, ylabel):
        axr = ax.twinx()
        axr.set_ylabel(ylabel, color='#d62728', fontsize=12)
        axr.set_ylim(0, 1)
        axr.tick_params(axis='y', labelcolor='#d62728')
        return axr
    
    def smooth_by_time(timestamps, values, window_seconds=60):
        """
        按时间窗口平滑数据
        
        Args:
            timestamps: 时间戳数组（秒）
            values: 数值数组
            window_seconds: 时间窗口大小（秒），默认60秒
        
        Returns:
            平滑后的数组
        """
        # 创建临时DataFrame
        df = pd.DataFrame({'t': timestamps, 'v': values})
        df = df.sort_values('t').reset_index(drop=True)
        
        # 转换为时间索引（使用datetime）
        base_time = pd.Timestamp('2000-01-01')  # 任意基准时间
        df['t_dt'] = base_time + pd.to_timedelta(df['t'], unit='s')
        df = df.set_index('t_dt')
        
        # 使用rolling按时间窗口平滑
        smoothed = df['v'].rolling(
            window=f'{window_seconds}s',
            center=True,
            min_periods=1
        ).mean()
        
        return smoothed.values
    setup_left_axis(ax_top)
    ax_top_r = setup_right_axis(ax_top, 'Cumulative Hit Rate')

    setup_left_axis(ax_bot)
    ax_bot_r = setup_right_axis(ax_bot, 'Instant Hit Rate (Per-trace)')

    ax_bot.set_xlabel('Timestamp (s)', fontsize=12)
    ax_bot.set_xlim(base_timestamps.min(), base_timestamps.max() * 1.05)

    colors = plt.cm.tab20(np.linspace(0.3, 0.9, len(instance_names)))

    top_lines = [ax_top.lines[0]]
    bot_lines = [ax_bot.lines[0]]

    # 上图：累计命中率
    for i, name in enumerate(instance_names):
        t0, t1 = all_time_ranges[i]
        valid = (base_timestamps >= t0) & (base_timestamps <= t1)

        l1 = ax_top_r.plot(base_timestamps[valid], np.array(all_acc_hit[i])[valid],
                        color=colors[i], label=f'{name} - AccHitRate',
                        linewidth=2, alpha=0.85, drawstyle='steps-post')
        ax_top_r.plot(base_timestamps[valid], np.array(all_acc_external_hit[i])[valid],
                    color=colors[i], linestyle='--', alpha=0.6,
                    linewidth=1.5, drawstyle='steps-post')
        top_lines += l1

    # 下图：当前trace命中率（按时间平滑 + 按时间降采样）
    downsample_interval_s = 10  # 每隔 10 秒取一个代表点
    window_seconds = 2

    for i, name in enumerate(instance_names):
        t0, t1 = all_time_ranges[i]
        valid = (base_timestamps >= t0) & (base_timestamps <= t1)

        # 按时间平滑
        hit_sm = smooth_by_time(base_timestamps, np.array(all_hit[i]), window_seconds)
        ext_sm = smooth_by_time(base_timestamps, np.array(all_external_hit[i]), window_seconds)

        # 按时间降采样：在 valid 范围内，每隔 downsample_interval_s 秒保留最近的一个点
        valid_idx = np.flatnonzero(valid)
        if len(valid_idx) == 0:
            continue

        sampled = [valid_idx[0]]
        last_t = base_timestamps[valid_idx[0]]
        for vi in valid_idx[1:]:
            if base_timestamps[vi] - last_t >= downsample_interval_s:
                sampled.append(vi)
                last_t = base_timestamps[vi]
        idx = np.array(sampled)

        l2 = ax_bot_r.plot(base_timestamps[idx], hit_sm[idx],
                        color=colors[i], label=f'{name} - HitRate',
                        linewidth=2, alpha=0.85)

        ax_bot_r.plot(base_timestamps[idx], ext_sm[idx],
                    color=colors[i], linestyle='--', alpha=0.6,
                    linewidth=1.5)
        bot_lines += l2

    ax_top.legend(top_lines, [l.get_label() for l in top_lines],
                loc='upper left', bbox_to_anchor=(0.01, 0.99),
                framealpha=0.95, fontsize=9)


    ax_top.tick_params(axis='x', labelbottom=True)
    ax_top.set_xlabel('Timestamp (s)', fontsize=12) 
    ax_top.set_title(f'Cache Analysis - {len(instance_names)} Instances', fontsize=15, fontweight='bold', pad=12)

    fig.tight_layout()
    output_file = os.path.join(csv_dir, "multi_instance_cache_analysis.png")
    plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Chart saved to: {output_file}")
    plt.close()

