#!/usr/bin/env python3
"""
绘图公共工具

职责：
- 颜色/标记常量
- 单策略 Pareto 曲线（每个 instance 一条线）
- 多策略对比子图（每个 instance 一个子图）
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
    hit_rate_type: str = "total",
    title: str = None,
    axis_limits: dict = None,
):
    """
    绘制单策略的容量-命中率散点图，每个 instance 一条曲线。

    Args:
        results:       [{"capacity": int, "instances": {...}}, ...]
        output_dir:    图片输出目录
        hit_rate_type: "total" | "internal" | "external"
        title:         图标题（None 则自动生成）
        axis_limits:   {"x_min", "x_max", "y_min", "y_max"}，None 表示不限制
    """
    if not results:
        print("No data to plot!")
        return

    instance_ids = list(results[0]["instances"].keys())
    plt.figure(figsize=(12, 8))

    for idx, iid in enumerate(instance_ids):
        caps = [r["capacity"] for r in results if iid in r["instances"]]
        rates = [r["instances"][iid][hit_rate_type] for r in results if iid in r["instances"]]
        if not caps:
            continue
        plt.scatter(caps, rates,
                    color=COLORS[idx % len(COLORS)],
                    marker=MARKERS[idx % len(MARKERS)],
                    s=1, label=iid, alpha=0.8)

    plt.xlabel("Cache Capacity (blocks)", fontsize=12)
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
    os.makedirs(output_dir, exist_ok=True)
    out = os.path.join(output_dir, f"pareto_curve_{hit_rate_type}.png")
    plt.savefig(out, dpi=300, bbox_inches="tight")
    print(f"Saved: {out}")
    plt.close()


# ============================================================================
# 多策略对比子图
# ============================================================================

def plot_multi_policy_subplots(
    results_by_policy: Dict[str, List[dict]],
    output_dir: str,
    hit_rate_type: str = "total",
):
    """
    多策略对比：每个 instance 一个子图，每个子图里每条策略一条曲线。

    Args:
        results_by_policy: {"policy": [{"capacity", "instances"}, ...]}
        output_dir:        图片输出目录
        hit_rate_type:     "total" | "internal" | "external"
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
                plot_data[iid][policy]["storage"].append(metrics["cached_blocks_all"])
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
        ax.set_xlabel("Group Total Storage (blocks)", fontsize=11)
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

    os.makedirs(output_dir, exist_ok=True)
    out = os.path.join(output_dir, f"multi_policy_{hit_rate_type}.png")
    plt.savefig(out, dpi=300, bbox_inches="tight")
    print(f"\nSaved: {out}")
    plt.close()
