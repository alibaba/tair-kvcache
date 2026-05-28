#!/usr/bin/env python3
"""
绘图公共工具

职责：
- 颜色/标记常量
- 单策略 Pareto 曲线（每个 instance 一条线）
- 多策略对比子图（每个 instance 一个子图）

Note: Tradeoff（Pareto 曲线）分析仅适用于非分层模式
（hierarchical_eviction_enabled=false）。分层模式下各 tier
独立容量构成多维搜索空间，quota_capacity 扫描无意义。
"""

import os
from typing import Dict, List, Optional

import matplotlib.pyplot as plt


# ============================================================================
# 全局绘图风格
# ============================================================================

def setup_plot_style():
    """统一的 matplotlib 绘图配置，在脚本入口调用一次即可"""
    plt.rcParams.update({
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.labelsize": 12,
        "legend.fontsize": 10,
        "figure.dpi": 100,
    })
    try:
        import seaborn as sns
        sns.set_palette("husl")
    except ImportError:
        pass


# ============================================================================
# 常量
# ============================================================================

COLORS = [
    "tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple",
    "tab:brown", "tab:pink", "tab:gray", "tab:olive", "tab:cyan",
]
MARKERS = ["o", "s", "^", "D", "v", "<", ">", "p", "h", "*"]


# ============================================================================
# 单策略 Pareto 曲线
# ============================================================================

def plot_single_policy_curves(
    results: List[dict],
    output_dir: str,
    bytes_per_block_map: Dict[str, int],
    hit_rate_type: str = "total",
    title: str = None,
    axis_limits: dict = None,
    config_quota_gb_map: Optional[Dict[str, float]] = None,
):
    """
    绘制单策略的容量-命中率散点图，每个 instance 一条曲线。

    Args:
        results:             [{"capacity": int, "instances": {...}}, ...]
        output_dir:          图片根输出目录，图表保存至 output_dir/pareto/
        bytes_per_block_map: {instance_id: bytes_per_block}，用于容量 blocks → GB 转换
        hit_rate_type:       "total" | "local" | "remote"
        title:               图标题（None 则自动生成）
        axis_limits:         {"x_min", "x_max", "y_min", "y_max"}，单位 GB；None 表示不限制
        config_quota_gb_map: {instance_id: quota_capacity_gb}，用于标注当前配置容量位置。
                             每个唯一的 quota_capacity_gb 对应一根竖赋参考线。
    """
    if not results:
        print("No data to plot!")
        return

    instance_ids = list(results[0]["instances"].keys())
    rep_bpb = next(iter(bytes_per_block_map.values()))
    plt.figure(figsize=(12, 8))

    for idx, iid in enumerate(instance_ids):
        bpb = bytes_per_block_map.get(iid, rep_bpb)
        caps = [r["capacity"] * bpb / (1024 ** 3) for r in results if iid in r["instances"]]
        rates = [r["instances"][iid][hit_rate_type] for r in results if iid in r["instances"]]
        if not caps:
            continue
        plt.scatter(caps, rates,
                    color=COLORS[idx % len(COLORS)],
                    marker=MARKERS[idx % len(MARKERS)],
                    s=1, label=iid, alpha=0.8)

    # Draw vertical reference lines for configured quota_capacity
    if config_quota_gb_map:
        # Collect unique quota values; use first matching instance_id as label suffix
        # when multiple groups have different quotas.
        quota_to_repr: Dict[float, str] = {}
        for iid in instance_ids:
            quota_gb = config_quota_gb_map.get(iid)
            if quota_gb is not None and quota_gb not in quota_to_repr:
                quota_to_repr[quota_gb] = iid
        multi_quota = len(quota_to_repr) > 1
        vline_colors = ["black", "dimgray", "saddlebrown"]
        for vi, (quota_gb, repr_iid) in enumerate(sorted(quota_to_repr.items())):
            label = (
                f"Config quota ({repr_iid}): {quota_gb:.1f} GB"
                if multi_quota
                else f"Config quota: {quota_gb:.1f} GB"
            )
            plt.axvline(
                x=quota_gb,
                color=vline_colors[vi % len(vline_colors)],
                linestyle="--",
                linewidth=1.5,
                label=label,
                alpha=0.75,
            )

    plt.xlabel("Cache Capacity (GB)", fontsize=12)
    plt.ylabel(f"{hit_rate_type.capitalize()} Hit Rate", fontsize=12)
    plt.title(title or f"KVCache Trade-off Curve - {hit_rate_type.capitalize()} Hit Rate", fontsize=14)
    plt.legend(loc="lower right", fontsize=10)
    plt.grid(True, alpha=0.3)

    al = axis_limits or {}
    if al.get("x_min") is not None or al.get("x_max") is not None:
        plt.xlim(al.get("x_min"), al.get("x_max"))
    y_lo = al.get("y_min") if al.get("y_min") is not None else 0
    y_hi = al.get("y_max") if al.get("y_max") is not None else 1
    plt.ylim(y_lo, y_hi)

    plt.tight_layout()
    pareto_dir = os.path.join(output_dir, "pareto")
    os.makedirs(pareto_dir, exist_ok=True)
    out = os.path.join(pareto_dir, f"pareto_curve_{hit_rate_type}.png")
    plt.savefig(out, dpi=300, bbox_inches="tight")
    print(f"Saved: {out}")
    plt.close()


# ============================================================================
# 多策略对比子图
# ============================================================================

def plot_multi_policy_subplots(
    results_by_policy: Dict[str, List[dict]],
    output_dir: str,
    bytes_per_block_map: Dict[str, int],
    hit_rate_type: str = "total",
    config_quota_gb_map: Optional[Dict[str, float]] = None,
):
    """
    多策略对比：每个 instance 一个子图，每个子图里每条策略一条曲线。

    Args:
        results_by_policy:   {"policy": [{"capacity", "instances"}, ...]}
        output_dir:          图片根输出目录，图表保存至 output_dir/pareto/
        bytes_per_block_map: {instance_id: bytes_per_block}，用于存储量 blocks → GB 转换
        hit_rate_type:       "total" | "local" | "remote"
        config_quota_gb_map: {instance_id: quota_capacity_gb}，用于在各子图中标注配置容量位置。
    """
    if not results_by_policy:
        print("No data to plot!")
        return

    from collections import defaultdict

    # 重组：plot_data[instance_id][policy] = {"storage": [...], "hit_rates": [...]}
    plot_data = defaultdict(lambda: defaultdict(lambda: {"storage": [], "hit_rates": []}))
    for policy, results in results_by_policy.items():
        for r in results:
            for iid, metrics in r["instances"].items():
                plot_data[iid][policy]["storage"].append(metrics["cached_gb"])
                plot_data[iid][policy]["hit_rates"].append(metrics[hit_rate_type])

    instance_ids = sorted(plot_data.keys())
    n = len(instance_ids)

    # 自适应子图布局
    if n == 1:
        nrows, ncols = 1, 1
    elif n <= 4:
        nrows, ncols = 2, 2
    elif n <= 6:
        nrows, ncols = 2, 3
    elif n <= 9:
        nrows, ncols = 3, 3
    else:
        ncols = 3
        nrows = (n + 2) // 3

    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 5 * nrows))
    axes_flat = [axes] if n == 1 else axes.flatten()

    policies = sorted(next(iter(plot_data.values())).keys())

    for idx, iid in enumerate(instance_ids):
        ax = axes_flat[idx]
        for pi, policy_name in enumerate(policies):
            data = plot_data[iid][policy_name]
            ax.scatter(data["storage"], data["hit_rates"],
                       label=policy_name,
                       color=COLORS[pi % len(COLORS)],
                       marker=MARKERS[pi % len(MARKERS)],
                       s=1, alpha=0.8)
        # Draw vertical reference line for this instance's configured quota_capacity
        if config_quota_gb_map:
            quota_gb = config_quota_gb_map.get(iid)
            if quota_gb is not None:
                ax.axvline(
                    x=quota_gb,
                    color="black",
                    linestyle="--",
                    linewidth=1.5,
                    label=f"Config quota: {quota_gb:.1f} GB",
                    alpha=0.75,
                )
        ax.set_xlabel("Group Total Storage (GB)", fontsize=11)
        ax.set_ylabel(f"{hit_rate_type.capitalize()} Hit Rate", fontsize=11)
        ax.set_ylim(0, 1.05)
        ax.grid(True, alpha=0.3, linestyle="--")
        ax.set_title(iid, fontsize=13, fontweight="bold")
        ax.legend(loc="lower right", fontsize=9, framealpha=0.9)

    for idx in range(n, len(axes_flat)):
        axes_flat[idx].set_visible(False)

    plt.suptitle(
        f"Multi-Policy Comparison - {hit_rate_type.capitalize()} Hit Rate",
        fontsize=16, fontweight="bold", y=0.998,
    )
    plt.tight_layout()

    pareto_dir = os.path.join(output_dir, "pareto")
    os.makedirs(pareto_dir, exist_ok=True)
    out = os.path.join(pareto_dir, f"multi_policy_{hit_rate_type}.png")
    plt.savefig(out, dpi=300, bbox_inches="tight")
    print(f"\nSaved: {out}")
    plt.close()
