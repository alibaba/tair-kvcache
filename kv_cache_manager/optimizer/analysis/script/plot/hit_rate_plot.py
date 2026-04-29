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
import re
from typing import Dict, Optional

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


def _ensure_hit_rate_timestamp_ns(df: pd.DataFrame) -> pd.DataFrame:
    """命中 CSV 仅使用 TimestampNs（纳秒）。"""
    if "TimestampNs" not in df.columns:
        raise ValueError("hit_rates CSV 缺少 TimestampNs 列")
    df["TimestampNs"] = pd.to_numeric(df["TimestampNs"], errors="coerce")
    return df


def _load_sp_cumulative(csv_dir, instance_name):
    """
    从 template_prefix_traces.csv 计算 system prompt 累积命中率时序。

    返回 DataFrame: [TimestampNs, AccSpHitRate]
    AccSpHitRate = cumsum(min(hit, template_depth)) / cumsum(total_blocks)
    """
    basename = instance_name.replace("_hit_rates", "")
    sp_path = os.path.join(csv_dir, f"{basename}_template_prefix_traces.csv")
    if not os.path.exists(sp_path):
        return None

    df = pd.read_csv(sp_path)
    # trace_id format: trace_<instance>_<timestamp_ns>
    df["TimestampNs"] = df["TraceId"].str.rsplit("_", n=1).str[-1].astype(np.int64)
    df = df.sort_values("TimestampNs")

    sp_hits = np.where(
        (df['TemplateId'] != 'NONE') & (df['TemplateDepth'] > 0),
        np.minimum(df['HitBlocks'].values, df['TemplateDepth'].values),
        0
    )

    cum_sp_hits = np.cumsum(sp_hits)
    cum_total = np.cumsum(df['TotalBlocks'].values)

    acc_sp_rate = np.where(cum_total > 0, cum_sp_hits / cum_total, 0.0)

    return pd.DataFrame({
        "TimestampNs": df["TimestampNs"].values,
        "AccSpHitRate": acc_sp_rate,
    })


def plot_multi_instance_analysis(
    csv_dir,
    output_dir: str = None,
    show_template: bool = True,
    bytes_per_block_map: Optional[Dict[str, int]] = None,
):
    """
    读取 csv_dir 下的命中率 CSV，生成时序分析图。

    Args:
        csv_dir:             CSV 数据目录
        output_dir:          图表根输出目录，图表保存至 output_dir/timeseries/
                             默认为 csv_dir（向后兼容）
        show_template:       是否在图上显示 SP 累计命中率线（需要 template_prefix_traces.csv）
        bytes_per_block_map: {instance_id: bytes_per_block}，不提供时存储量以 blocks 显示
    """
    csv_files = sorted(glob.glob(os.path.join(csv_dir, "*_hit_rates.csv")))
    if not csv_files:
        print(f"Error: No CSV files found in directory: {csv_dir}")
        return

    dataframes, instance_names = [], []
    # per-tier BlockNum 列名 -> tier_name 映射（从第一个有 tier 列的 CSV 中提取）
    tier_block_cols = {}  # {col_name: tier_name}

    for csv_file in csv_files:
        df = read_csv_file(csv_file)
        if df is None:
            continue

        df = _ensure_hit_rate_timestamp_ns(df)

        # 数值化 + 排序
        for c in ['CachedBlocksAllInstance', 'AccHitRate', 'AccRemoteHitRate', 'AccReadBlocks']:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce")

        # 收集 per-tier BlockNum 列
        if not tier_block_cols:
            for col in df.columns:
                m = re.match(r'Tier\d+\(([^)]+)\)_BlockNum', col)
                if m:
                    tier_block_cols[col] = m.group(1)

        # 数值化 tier BlockNum 列
        for col in tier_block_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        df = df.dropna(subset=["TimestampNs"]).sort_values("TimestampNs")


        dataframes.append(df)
        instance_names.append(os.path.splitext(os.path.basename(csv_file))[0])

    if not dataframes:
        print("Error: No valid CSV data could be loaded")
        return

    required_cols = ['TimestampNs', 'CachedBlocksAllInstance', 'AccHitRate', 'AccRemoteHitRate', 'AccReadBlocks']
    for i, df in enumerate(dataframes):
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            print(f"Error: {instance_names[i]} is missing required columns: {missing}")
            return

    # 全局基准：最早起点
    min_timestamp = min(df["TimestampNs"].iloc[0] for df in dataframes)

    # "每个trace都画"：基准时间轴取所有instance的时间戳并集（秒）
    all_t = []
    for df in dataframes:
        all_t.append(((df["TimestampNs"] - min_timestamp) / 1e9).to_numpy())
    base_timestamps = np.unique(np.concatenate(all_t))
    base = pd.DataFrame({'t': base_timestamps})  # 用于merge_asof

    all_acc_sp_hit = []
    all_acc_hit, all_acc_remote_hit, all_time_ranges = [], [], []
    # 用于瞬时命中率计算：累积读块数 / 累积命中块数（反推）
    all_acc_read_blocks, all_acc_hit_blocks, all_acc_remote_hit_blocks = [], [], []
    all_acc_sp_hit = []

    global_updates_list = []
    tier_updates_lists = {col: [] for col in tier_block_cols}  # per-tier 更新列表

    for df in dataframes:
        d = df.copy()
        d["t"] = (d["TimestampNs"] - min_timestamp) / 1e9
        d = d.sort_values('t')

        # 反推累积命中块数：AccHitBlocks = AccHitRate × AccReadBlocks
        d['AccHitBlocks']       = d['AccHitRate']       * d['AccReadBlocks']
        d['AccRemoteHitBlocks'] = d['AccRemoteHitRate'] * d['AccReadBlocks']

        global_updates_list.append(d[['t', 'CachedBlocksAllInstance']])

        # 收集 per-tier 数据
        for col in tier_block_cols:
            if col in d.columns:
                tier_updates_lists[col].append(d[['t', col]])
        t0, t1 = d['t'].iloc[0], d['t'].iloc[-1]
        all_time_ranges.append((t0, t1))

        # 真实对齐：取 <=t 的最后一次上报（ZOH），不插值
        aligned = pd.merge_asof(
            base,
            d[['t', 'AccHitRate', 'AccRemoteHitRate',
               'AccReadBlocks', 'AccHitBlocks', 'AccRemoteHitBlocks']],
            on='t',
            direction='backward',
            allow_exact_matches=True
        )

        mask_out = (aligned['t'] < t0)
        acc_cols = ['AccHitRate', 'AccRemoteHitRate',
                    'AccReadBlocks', 'AccHitBlocks', 'AccRemoteHitBlocks']
        aligned.loc[mask_out, acc_cols] = np.nan

        all_acc_hit.append(aligned['AccHitRate'].to_numpy(float))
        all_acc_remote_hit.append(aligned['AccRemoteHitRate'].to_numpy(float))
        all_acc_read_blocks.append(aligned['AccReadBlocks'].to_numpy(float))
        all_acc_hit_blocks.append(aligned['AccHitBlocks'].to_numpy(float))
        all_acc_remote_hit_blocks.append(aligned['AccRemoteHitBlocks'].to_numpy(float))

    # ---- SP 累积命中率对齐 ----
    if show_template:
        for idx, name in enumerate(instance_names):
            sp_df = _load_sp_cumulative(csv_dir, name)
            if sp_df is None:
                all_acc_sp_hit.append(None)
                continue
            sp_df["t"] = (sp_df["TimestampNs"] - min_timestamp) / 1e9
            sp_df = sp_df.sort_values('t')
            sp_aligned = pd.merge_asof(
                base, sp_df[['t', 'AccSpHitRate']], on='t',
                direction='backward', allow_exact_matches=True
            )
            t0, _ = all_time_ranges[idx]
            sp_aligned.loc[sp_aligned['t'] < t0, 'AccSpHitRate'] = np.nan
            all_acc_sp_hit.append(sp_aligned['AccSpHitRate'].to_numpy(float))
    else:
        all_acc_sp_hit = [None] * len(instance_names)

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
    rep_bpb = None
    if bytes_per_block_map:
        rep_bpb = next(iter(bytes_per_block_map.values()))
        if rep_bpb and rep_bpb > 0:
            total_storage = total_storage * rep_bpb / (1024 ** 3)
            storage_label = 'InstanceGroup Storage (GB)'
        else:
            rep_bpb = None
            storage_label = 'InstanceGroup Storage (blocks)'
    else:
        storage_label = 'InstanceGroup Storage (blocks)'

    # ---- per-tier 存储对齐 ----
    tier_storage = {}  # {tier_name: aligned_array}
    for col, tier_name in tier_block_cols.items():
        if col not in tier_updates_lists or not tier_updates_lists[col]:
            continue
        tier_updates = pd.concat(tier_updates_lists[col], ignore_index=True)
        tier_updates = tier_updates.dropna(subset=['t', col]).sort_values('t')
        tier_updates = tier_updates.groupby('t', as_index=False)[col].median()
        tier_aligned = pd.merge_asof(
            base, tier_updates, on='t',
            direction='backward', allow_exact_matches=True
        )
        arr = tier_aligned[col].to_numpy(float)
        if rep_bpb and rep_bpb > 0:
            arr = arr * rep_bpb / (1024 ** 3)
        tier_storage[tier_name] = arr

    # ---- 画图 ----
    fig, (ax_top, ax_bot) = plt.subplots(
        2, 1, figsize=(16, 20), sharex=True,
        gridspec_kw={'height_ratios': [1, 1], 'hspace': 0.12},
        constrained_layout=True,
    )

    # per-tier 颜色
    TIER_COLORS = [
        '#2ca02c', '#ff7f0e', '#9467bd', '#8c564b', '#e377c2',
        '#7f7f7f', '#bcbd22', '#17becf', '#d62728', '#1f77b4',
    ]

    def setup_left_axis(ax):
        ax.set_ylabel(storage_label, color='#1f77b4', fontsize=12)
        # total 线（虚线+较粗），始终绘制
        ax.plot(base_timestamps, total_storage, color='#1f77b4',
                label=f'Total {storage_label}', linewidth=2.5, alpha=0.9,
                linestyle='--', drawstyle='steps-post')
        # per-tier 堆积面积图（有 tier 数据时）
        if tier_storage:
            tier_names_ordered = list(tier_storage.keys())
            tier_arrays = [tier_storage[n] for n in tier_names_ordered]
            # 用 stackplot 绘制堆积面积
            stacked = np.row_stack(tier_arrays)
            ax.stackplot(
                base_timestamps, stacked,
                labels=tier_names_ordered,
                colors=[TIER_COLORS[i % len(TIER_COLORS)] for i in range(len(tier_names_ordered))],
                alpha=0.35, step='post',
            )
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
    
    def window_hit_rate(timestamps, acc_hit_blocks, acc_read_blocks, window_seconds=60):
        """
        基于累积量差值计算时间窗口内的真实命中率。

        对每个采样点 t_i，窗口基准取 [t_i - window_seconds, t_i] 内
        最早的真实上报点，end 取 t_i 处的累积值：
            hit_rate = (hit[end] - hit[beg]) / (read[end] - read[beg])

        空缺处理：真实上报点之间若存在间隔 > window_seconds 的空缺，
        窗口基准不会越过该空缺，自动重置为空缺后的第一个真实上报点。
        这样空缺前的历史累积量不会污染空缺后的命中率计算。

        实现说明：纯 numpy 向量化（无 Python 循环），在百万量级采样点下
        相较逐点 for-loop 版本加速约 50 倍。
        """
        ts       = np.asarray(timestamps,      dtype=float)
        hit_arr  = np.asarray(acc_hit_blocks,  dtype=float)
        read_arr = np.asarray(acc_read_blocks, dtype=float)

        # 只保留真实上报点（非 nan）
        real_mask = ~np.isnan(read_arr)
        real_idx  = np.flatnonzero(real_mask)
        if len(real_idx) == 0:
            return np.full(len(ts), np.nan)

        real_ts   = ts[real_idx]
        real_hit  = hit_arr[real_idx]
        real_read = read_arr[real_idx]

        # 每个真实点所属"连续段"的起始索引（在 real_idx 中的位置）
        # 段边界：相邻真实点时间间隔 > window_seconds
        gaps = np.diff(real_ts)
        seg_starts_in_real = np.concatenate(([0], np.flatnonzero(gaps > window_seconds) + 1))

        # real_idx[k] 属于哪个段 → 该段在 real_idx 中的起始位置（向量化查表）
        k_arr = np.arange(len(real_idx))
        seg_of_real = seg_starts_in_real[
            np.searchsorted(seg_starts_in_real, k_arr, side='right') - 1
        ]

        # 窗口左边界对应的时间戳数组，并在 real_ts 中定位左端点
        t_left = real_ts - window_seconds
        beg_k_raw = np.searchsorted(real_ts, t_left, side='left')
        beg_k = np.maximum(seg_of_real, beg_k_raw)

        delta_read = real_read - real_read[beg_k]
        delta_hit  = real_hit  - real_hit[beg_k]
        # 避免 0 除；结果在 delta_read<=0 处为 NaN
        safe_read = np.where(delta_read > 0, delta_read, 1.0)
        rate_real = np.where(
            delta_read > 0,
            np.maximum(0.0, delta_hit / safe_read),
            np.nan,
        )

        rate = np.full(len(ts), np.nan)
        rate[real_idx] = rate_real
        return rate
    setup_left_axis(ax_top)
    ax_top_r = setup_right_axis(ax_top, 'Cumulative Hit Rate')

    setup_left_axis(ax_bot)
    ax_bot_r = setup_right_axis(ax_bot, 'Instant Hit Rate')

    ax_bot.set_xlabel('Timestamp (s)', fontsize=12)
    ax_bot.set_xlim(base_timestamps.min(), base_timestamps.max() * 1.05)

    colors = plt.cm.tab20(np.linspace(0.3, 0.9, len(instance_names)))

    # 不再手动收集 lines，最后统一从 axes 收集 handles/labels

    # 上图：累计命中率 + system prompt 累积命中率
    for i, name in enumerate(instance_names):
        t0, t1 = all_time_ranges[i]
        valid = (base_timestamps >= t0) & (base_timestamps <= t1)

        ax_top_r.plot(base_timestamps[valid], np.array(all_acc_hit[i])[valid],
                     color=colors[i], label=f'{name} - AccHitRate',
                     linewidth=2, alpha=0.85, drawstyle='steps-post')
        ax_top_r.plot(base_timestamps[valid], np.array(all_acc_remote_hit[i])[valid],
                     color=colors[i], linestyle='--', alpha=0.6,
                     label=f'{name} - AccRemoteHitRate',
                     linewidth=1.5, drawstyle='steps-post')
        if all_acc_sp_hit[i] is not None:
            ax_top_r.plot(
                base_timestamps[valid], np.array(all_acc_sp_hit[i])[valid],
                color=colors[i], linestyle=':', linewidth=2.5, alpha=0.9,
                label=f'{name} - SP AccHitRate',
                drawstyle='steps-post')

    # 下图：时间窗口内真实命中率（累积量差值）+ 按时间降采样
    downsample_interval_s = 10   # 每隔 10 秒取一个代表点
    window_seconds         = 10  # 窗口内累积命中率的统计时间跨度

    for i, name in enumerate(instance_names):
        t0, t1 = all_time_ranges[i]
        valid = (base_timestamps >= t0) & (base_timestamps <= t1)

        # 基于累积量差值计算窗口命中率（正确权重 = 请求数，而非上报次数）
        hit_sm = window_hit_rate(
            base_timestamps,
            all_acc_hit_blocks[i],
            all_acc_read_blocks[i],
            window_seconds,
        )
        remote_sm = window_hit_rate(
            base_timestamps,
            all_acc_remote_hit_blocks[i],
            all_acc_read_blocks[i],
            window_seconds,
        )

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

        ax_bot_r.plot(base_timestamps[idx], hit_sm[idx],
                     color=colors[i], label=f'{name} - HitRate',
                     linewidth=2, alpha=0.85)
        ax_bot_r.plot(base_timestamps[idx], remote_sm[idx],
                     color=colors[i], linestyle='--', alpha=0.6,
                     label=f'{name} - RemoteHitRate',
                     linewidth=1.5)
    # ---- 收集所有带 label 的 handles，放置图例到图外侧 ----
    def _collect_legend(ax_left, ax_right):
        h1, l1 = ax_left.get_legend_handles_labels()
        h2, l2 = ax_right.get_legend_handles_labels()
        return h1 + h2, l1 + l2

    h_top, l_top = _collect_legend(ax_top, ax_top_r)
    ax_top.legend(h_top, l_top,
                  bbox_to_anchor=(1.05, 1), loc='upper left',
                  fontsize=8, framealpha=0.95)

    h_bot, l_bot = _collect_legend(ax_bot, ax_bot_r)
    ax_bot.legend(h_bot, l_bot,
                  bbox_to_anchor=(1.05, 1), loc='upper left',
                  fontsize=8, framealpha=0.95)

    ax_top.tick_params(axis='x', labelbottom=True)
    ax_top.set_xlabel('Timestamp (s)', fontsize=12)
    ax_top.set_title(f'Cache Analysis - {len(instance_names)} Instances', fontsize=15, fontweight='bold', pad=12)
    ax_bot.set_title(f'Instant Hit Rate (window={window_seconds}s)', fontsize=15, fontweight='bold', pad=12)

    timeseries_dir = os.path.join(output_dir or csv_dir, "timeseries")
    os.makedirs(timeseries_dir, exist_ok=True)
    output_file = os.path.join(timeseries_dir, "multi_instance_cache_analysis.png")
    plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Chart saved to: {output_file}")
    plt.close()


# ============================================================================
# Per-Tier 时序图
# ============================================================================

def plot_per_tier_timeseries(csv_dir, output_dir: str = None):
    """
    读取 csv_dir 下的命中率 CSV，生成 per-tier 命中率时序图。

    同一 instance 的所有 tier 使用相同颜色、不同线型区分。
    图例格式：{instance_name} - {tier_name}

    如果 CSV 中没有 per-tier 列，优雅跳过（打印 info 并返回）。

    Args:
        csv_dir:    CSV 数据目录
        output_dir: 图表根输出目录，图表保存至 output_dir/timeseries/
    """
    csv_files = sorted(glob.glob(os.path.join(csv_dir, "*_hit_rates.csv")))
    if not csv_files:
        print(f"[INFO] No hit_rates CSV files found in {csv_dir}, skipping per-tier chart.")
        return

    # 读取所有 instance 的数据
    dataframes = []
    instance_names = []

    for csv_file in csv_files:
        df = read_csv_file(csv_file)
        if df is None:
            continue

        # 数值化 + 排序
        if 'TimestampNs' in df.columns:
            df['TimestampNs'] = pd.to_numeric(df['TimestampNs'], errors='coerce')
        else:
            continue

        tier_cols = [col for col in df.columns if col.startswith("AccTier") and col.endswith("_HitRate")]
        for c in tier_cols:
            df[c] = pd.to_numeric(df[c], errors='coerce')
        df = df.dropna(subset=['TimestampNs']).sort_values('TimestampNs')

        dataframes.append(df)
        instance_names.append(os.path.splitext(os.path.basename(csv_file))[0].replace("_hit_rates", ""))

    if not dataframes:
        print("[INFO] No valid CSV data could be loaded, skipping per-tier chart.")
        return

    # 提取 tier 列信息（idx -> (col_name, tier_name)）
    tier_info = []  # [(tier_idx, col_name, tier_name), ...]
    for col in dataframes[0].columns:
        m = re.search(r'AccTier(\d+)\(([^)]+)\)_HitRate', col)
        if m:
            tier_info.append((int(m.group(1)), col, m.group(2)))

    if not tier_info:
        print("[INFO] No per-tier columns (AccTier*_HitRate) found in CSV, skipping per-tier chart.")
        return

    tier_info.sort(key=lambda x: x[0])

    # 统一时间轴（向量化：numpy.unique 远快于 set.update + sorted）
    all_ts = np.unique(np.concatenate([
        df['TimestampNs'].to_numpy() for df in dataframes
    ]))
    min_timestamp = all_ts[0]
    base_timestamps = (all_ts - min_timestamp) / 1e9

    # 颜色：同一 instance 同色
    INSTANCE_COLORS = [
        "tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple",
        "tab:brown", "tab:pink", "tab:gray", "tab:olive", "tab:cyan",
    ]
    # 线型：不同 tier 不同线型
    TIER_LINESTYLES = ['-', '--', ':', '-.', (0, (3, 1, 1, 1))]

    fig, ax = plt.subplots(figsize=(16, 8))

    for inst_idx, (df, inst_name) in enumerate(zip(dataframes, instance_names)):
        color = INSTANCE_COLORS[inst_idx % len(INSTANCE_COLORS)]
        df_copy = df.copy()
        df_copy['t'] = (df_copy['TimestampNs'] - min_timestamp) / 1e9
        df_copy = df_copy.sort_values('t')

        for tier_order, (_, tier_col, tier_name) in enumerate(tier_info):
            if tier_col not in df_copy.columns:
                continue
            ls = TIER_LINESTYLES[tier_order % len(TIER_LINESTYLES)]

            # 对齐到统一时间轴
            df_aligned = pd.merge_asof(
                pd.DataFrame({'t': base_timestamps}),
                df_copy[['t', tier_col]],
                on='t',
                direction='backward',
                allow_exact_matches=True,
            )
            vals = df_aligned[tier_col].values

            ax.plot(base_timestamps, vals,
                    color=color, linestyle=ls, linewidth=2,
                    label=f"{inst_name} - {tier_name}",
                    alpha=0.85)

    ax.set_xlabel('Time (seconds)', fontsize=12)
    ax.set_ylabel('Accumulative Hit Rate', fontsize=12)
    ax.set_title(f'Per-Tier Hit Rate Over Time - {len(instance_names)} Instances',
                 fontsize=14, fontweight='bold')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8, framealpha=0.95)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.05)

    fig.tight_layout()
    timeseries_dir = os.path.join(output_dir or csv_dir, "timeseries")
    os.makedirs(timeseries_dir, exist_ok=True)
    output_file = os.path.join(timeseries_dir, "per_tier_timeseries.png")
    plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Per-tier timeseries chart saved to: {output_file}")
    plt.close()
