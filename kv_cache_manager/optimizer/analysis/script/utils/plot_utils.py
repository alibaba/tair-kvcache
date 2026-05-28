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

def _fmt_gb(value: float) -> str:
    if value >= 1000:
        return f"{value:,.0f}"
    return f"{value:,.1f}"


def _trim_after_target(curve: List[tuple], target: float) -> List[tuple]:
    trimmed = []
    for point in sorted(curve, key=lambda item: item[0]):
        trimmed.append(point)
        if point[1] >= target:
            break
    return trimmed


def _drop_descending_points(curve: List[tuple], label: str) -> List[tuple]:
    """Drop dominated points where hit rate falls below the best previously kept point."""
    kept = []
    best_hit = None
    best_cap = None
    for cap, hit in sorted(curve, key=lambda item: item[0]):
        if best_hit is not None and hit < best_hit:
            print(
                "Drop descending Pareto point [{}]: capacity={:.6f} GB, hit_rate={:.6f}% "
                "< best_so_far={:.6f}% at {:.6f} GB".format(
                    label,
                    cap,
                    hit * 100,
                    best_hit * 100,
                    best_cap,
                )
            )
            continue
        kept.append((cap, hit))
        best_hit = hit
        best_cap = cap
    return kept


def _interpolate_capacity(curve: List[tuple], target: float) -> tuple:
    ordered = sorted(curve, key=lambda item: item[0])
    prev_cap, prev_hit = 0.0, 0.0
    for cap, hit in ordered:
        if hit >= target:
            if hit <= prev_hit:
                return cap, target
            ratio = (target - prev_hit) / (hit - prev_hit)
            return prev_cap + ratio * (cap - prev_cap), target
        prev_cap, prev_hit = cap, hit
    return ordered[-1] if ordered else (0.0, 0.0)


def plot_single_policy_curves(
    results: List[dict],
    output_dir: str,
    bytes_per_block_map: Dict[str, int],
    hit_rate_type: str = "total",
    title: str = None,
    axis_limits: dict = None,
    config_quota_gb_map: Optional[Dict[str, float]] = None,
    theoretical_hit_rates: Optional[Dict[str, dict]] = None,
):
    """
    绘制单策略的容量-命中率 Pareto 曲线，每个 instance 一条曲线。

    Args:
        results:             [{"capacity": int, "instances": {...}}, ...]
        output_dir:          图片根输出目录，图表保存至 output_dir/pareto/
        bytes_per_block_map: {instance_id: bytes_per_block}，用于容量 blocks → GB 转换
        hit_rate_type:       "total" | "local" | "remote"
        title:               图标题（None 则自动生成）
        axis_limits:         {"x_min", "x_max", "y_min", "y_max"}，单位 GB；None 表示不限制
        config_quota_gb_map: {instance_id: quota_capacity_gb}，用于标注当前配置容量位置。
                             每个唯一的 quota_capacity_gb 对应一根竖赋参考线。
        theoretical_hit_rates:
                             {instance_id: {"total"|"local"|"remote": hit_rate}}，用于插值标注
                             95%/99% 理论命中率。
    """
    if not results:
        print("No data to plot!")
        return

    instance_ids = list(results[0]["instances"].keys())
    rep_bpb = next(iter(bytes_per_block_map.values()))
    fig, ax = plt.subplots(figsize=(12.8, 7.40625), dpi=160)
    labeled_95 = False
    labeled_99 = False

    for idx, iid in enumerate(instance_ids):
        bpb = bytes_per_block_map.get(iid, rep_bpb)
        curve = [
            (r["capacity"] * bpb / (1024 ** 3), r["instances"][iid][hit_rate_type])
            for r in results
            if iid in r["instances"]
        ]
        if not curve:
            continue
        theory = None
        if theoretical_hit_rates:
            theory = theoretical_hit_rates.get(iid, {}).get(hit_rate_type)
        curve = _drop_descending_points(curve, f"{iid}:{hit_rate_type}")
        if theory is not None:
            curve = _trim_after_target(curve, float(theory) * 0.99)
        plot_curve = [(0.0, 0.0)] + sorted(curve, key=lambda item: item[0])
        caps = [p[0] for p in plot_curve]
        rates = [p[1] * 100 for p in plot_curve]
        color = COLORS[idx % len(COLORS)]
        curve_label = "Pareto" if len(instance_ids) == 1 else iid
        ax.plot(caps, rates, marker="o", linewidth=2.0, markersize=5, color=color, label=curve_label)

        if theory is not None:
            target95 = float(theory) * 0.95
            target99 = float(theory) * 0.99
            cap95, hit95 = _interpolate_capacity(plot_curve, target95)
            cap99, hit99 = _interpolate_capacity(plot_curve, target99)

            label95 = "95%" if not labeled_95 else None
            label99 = "99%" if not labeled_99 else None
            ax.axhline(hit95 * 100, color="tab:orange", linestyle="--", linewidth=1.4, label=label95)
            ax.axvline(cap95, color="tab:orange", linestyle=":", linewidth=1.4)
            ax.axhline(hit99 * 100, color="tab:red", linestyle="--", linewidth=1.4, label=label99)
            ax.axvline(cap99, color="tab:red", linestyle=":", linewidth=1.4)
            labeled_95 = True
            labeled_99 = True

            ax.scatter([cap95], [hit95 * 100], facecolors="none", edgecolors="tab:orange",
                       linewidths=2.0, s=120, zorder=5)
            ax.scatter([cap99], [hit99 * 100], facecolors="none", edgecolors="tab:red",
                       linewidths=2.0, s=120, zorder=5)
            ax.annotate(
                f"95%: {_fmt_gb(cap95)} GB\n{hit95 * 100:.2f}%",
                xy=(cap95, hit95 * 100),
                xytext=(10, -30),
                textcoords="offset points",
                fontsize=10,
                color="tab:orange",
            )
            ax.annotate(
                f"99%: {_fmt_gb(cap99)} GB\n{hit99 * 100:.2f}%",
                xy=(cap99, hit99 * 100),
                xytext=(10, 12),
                textcoords="offset points",
                fontsize=10,
                color="tab:red",
            )

    # Draw vertical reference lines for configured quota_capacity
    if config_quota_gb_map and not theoretical_hit_rates:
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
            ax.axvline(
                x=quota_gb,
                color=vline_colors[vi % len(vline_colors)],
                linestyle="--",
                linewidth=1.5,
                label=label,
                alpha=0.75,
            )

    ax.set_xlabel("Capacity (GB)", fontsize=12)
    ax.set_ylabel("HitRate (%)", fontsize=12)
    default_title = f"{instance_ids[0]} Pareto" if len(instance_ids) == 1 else "Pareto"
    ax.set_title(title or default_title, fontsize=14)
    ax.legend(loc="lower right", fontsize=10)
    ax.grid(True, alpha=0.3)

    al = axis_limits or {}
    if al.get("x_min") is not None or al.get("x_max") is not None:
        ax.set_xlim(al.get("x_min"), al.get("x_max"))
    else:
        ax.set_xlim(left=0)
    y_lo = al.get("y_min") if al.get("y_min") is not None else 0
    y_hi = al.get("y_max") if al.get("y_max") is not None else 100
    ax.set_ylim(y_lo, y_hi)

    fig.tight_layout()
    pareto_dir = os.path.join(output_dir, "pareto")
    os.makedirs(pareto_dir, exist_ok=True)
    out = os.path.join(pareto_dir, f"pareto_curve_{hit_rate_type}.png")
    fig.savefig(out, dpi=300, bbox_inches="tight")
    print(f"Saved: {out}")
    plt.close(fig)


# ============================================================================
# 多策略对比子图
# ============================================================================

def plot_multi_policy_subplots(
    results_by_policy: Dict[str, List[dict]],
    output_dir: str,
    bytes_per_block_map: Dict[str, int],
    hit_rate_type: str = "total",
    title: str = None,
    axis_limits: dict = None,
    config_quota_gb_map: Optional[Dict[str, float]] = None,
    theoretical_hit_rates_by_policy: Optional[Dict[str, Dict[str, dict]]] = None,
):
    """
    多策略对比：每个 instance 一个子图，每个子图里每条策略一条曲线。

    Args:
        results_by_policy:   {"policy": [{"capacity", "instances"}, ...]}
        output_dir:          图片根输出目录，图表保存至 output_dir/pareto/
        bytes_per_block_map: {instance_id: bytes_per_block}，用于存储量 blocks → GB 转换
        hit_rate_type:       "total" | "local" | "remote"
        title:               图标题（None 则自动生成）
        axis_limits:         {"x_min", "x_max", "y_min", "y_max"}，单位 GB；None 表示不限制
        config_quota_gb_map: {instance_id: quota_capacity_gb}，用于在各子图中标注配置容量位置。
        theoretical_hit_rates_by_policy:
                             {policy: {instance_id: {"total"|"local"|"remote": hit_rate}}}
    """
    if not results_by_policy:
        print("No data to plot!")
        return

    from collections import defaultdict

    # 重组：plot_data[instance_id][policy] = [(capacity_gb, hit_rate), ...]
    rep_bpb = next(iter(bytes_per_block_map.values()))
    plot_data = defaultdict(lambda: defaultdict(list))
    for policy, results in results_by_policy.items():
        for r in results:
            for iid, metrics in r["instances"].items():
                bpb = bytes_per_block_map.get(iid, rep_bpb)
                capacity_gb = r["capacity"] * bpb / (1024 ** 3)
                plot_data[iid][policy].append((capacity_gb, metrics[hit_rate_type]))

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
            curve = plot_data[iid][policy_name]
            if not curve:
                continue
            theory = None
            if theoretical_hit_rates_by_policy:
                theory = theoretical_hit_rates_by_policy.get(policy_name, {}).get(iid, {}).get(hit_rate_type)
            curve = _drop_descending_points(curve, f"{iid}:{policy_name}:{hit_rate_type}")
            if theory is not None:
                curve = _trim_after_target(curve, float(theory) * 0.99)
            plot_curve = [(0.0, 0.0)] + sorted(curve, key=lambda item: item[0])
            caps = [p[0] for p in plot_curve]
            rates = [p[1] * 100 for p in plot_curve]
            color = COLORS[pi % len(COLORS)]
            ax.plot(
                caps,
                rates,
                label=policy_name,
                color=color,
                marker=MARKERS[pi % len(MARKERS)],
                linewidth=2.0,
                markersize=4.5,
                alpha=0.92,
            )

            if theory is not None:
                target95 = float(theory) * 0.95
                target99 = float(theory) * 0.99
                cap95, hit95 = _interpolate_capacity(plot_curve, target95)
                cap99, hit99 = _interpolate_capacity(plot_curve, target99)
                ax.axhline(hit95 * 100, color=color, linestyle="--", linewidth=1.0, alpha=0.32)
                ax.axvline(cap95, color=color, linestyle="--", linewidth=1.0, alpha=0.32)
                ax.axhline(hit99 * 100, color=color, linestyle=":", linewidth=1.2, alpha=0.45)
                ax.axvline(cap99, color=color, linestyle=":", linewidth=1.2, alpha=0.45)
                ax.scatter([cap95], [hit95 * 100], facecolors="none", edgecolors=color,
                           linewidths=1.7, s=90, zorder=5)
                ax.scatter([cap99], [hit99 * 100], facecolors="none", edgecolors=color,
                           linewidths=2.0, s=125, zorder=5)
                # Stagger labels by policy index to keep multiple strategies readable.
                y_offset = -32 - (pi % 3) * 18
                ax.annotate(
                    f"{policy_name} 95%: {_fmt_gb(cap95)} GB\n{hit95 * 100:.2f}%",
                    xy=(cap95, hit95 * 100),
                    xytext=(10, y_offset),
                    textcoords="offset points",
                    fontsize=8.5,
                    color=color,
                )
                ax.annotate(
                    f"{policy_name} 99%: {_fmt_gb(cap99)} GB\n{hit99 * 100:.2f}%",
                    xy=(cap99, hit99 * 100),
                    xytext=(10, 12 + (pi % 3) * 18),
                    textcoords="offset points",
                    fontsize=8.5,
                    color=color,
                )
        # Draw vertical reference line for this instance's configured quota_capacity
        if config_quota_gb_map and not theoretical_hit_rates_by_policy:
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
        ax.set_xlabel("Capacity (GB)", fontsize=11)
        ax.set_ylabel("HitRate (%)", fontsize=11)
        al = axis_limits or {}
        if al.get("x_min") is not None or al.get("x_max") is not None:
            ax.set_xlim(al.get("x_min"), al.get("x_max"))
        else:
            ax.set_xlim(left=0)
        y_lo = al.get("y_min") if al.get("y_min") is not None else 0
        y_hi = al.get("y_max") if al.get("y_max") is not None else 100
        ax.set_ylim(y_lo, y_hi)
        ax.grid(True, alpha=0.3)
        ax.set_title(iid, fontsize=13)
        ax.legend(loc="lower right", fontsize=9, framealpha=0.9)

    for idx in range(n, len(axes_flat)):
        axes_flat[idx].set_visible(False)

    plt.suptitle(title or f"Multi-Policy Pareto - {hit_rate_type.capitalize()}", fontsize=16, y=0.998)
    plt.tight_layout()

    pareto_dir = os.path.join(output_dir, "pareto")
    os.makedirs(pareto_dir, exist_ok=True)
    out = os.path.join(pareto_dir, f"multi_policy_{hit_rate_type}.png")
    plt.savefig(out, dpi=300, bbox_inches="tight")
    print(f"\nSaved: {out}")
    plt.close()
