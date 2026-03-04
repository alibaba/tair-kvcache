#!/usr/bin/env python3
"""
Lifecycle 可视化模块

职责：
- Physical Lifespan CDF
- Active Lifespan CDF
- Access Count 直方图

接收排序后的 numpy 数组，不做数据处理。
"""

import matplotlib.pyplot as plt
import numpy as np


def _annotate_percentiles(ax, sorted_data, percentiles, x_max):
    """在 CDF 图上标注百分位点，返回百分位值列表"""
    vals = np.percentile(sorted_data, percentiles)
    n = len(sorted_data)
    for pct, val in zip(percentiles, vals):
        val = float(val)
        cdf_y = pct
        ax.axhline(cdf_y, color="red", linestyle=":", alpha=0.3, linewidth=1)
        ax.axvline(val, color="red", linestyle=":", alpha=0.3, linewidth=1)
        ax.plot(val, cdf_y, "ro", markersize=6, zorder=5)

        ha = "left" if val < x_max * 0.5 else "right"
        dy = 2 if val < x_max * 0.5 else -2
        ax.text(val, cdf_y + dy, f"P{pct}: {val:.1f}s", fontsize=9, ha=ha,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7))
    return vals


def plot_physical_lifespan_cdf(all_sorted, evicted_sorted, instance_name, output_path):
    """Physical Lifespan CDF: 全量 + Evicted"""
    n_all, n_ev = len(all_sorted), len(evicted_sorted)
    if n_all == 0:
        print("  跳过 Physical Lifespan CDF: 无数据")
        return

    cdf_all = np.arange(1, n_all + 1) / n_all * 100
    p99 = float(np.percentile(all_sorted, 99))
    x_max = min(p99 * 1.5, float(all_sorted[-1])) if all_sorted[-1] > 0 else 1.0

    fig, ax = plt.subplots(figsize=(14, 8))

    ax.plot(all_sorted, cdf_all, color="steelblue", linewidth=2.5,
            label=f"All Blocks (n={n_all:,})", alpha=0.9)
    ax.fill_between(all_sorted, cdf_all, alpha=0.1, color="steelblue")

    if n_ev > 0:
        cdf_ev = np.arange(1, n_ev + 1) / n_ev * 100
        ax.plot(evicted_sorted, cdf_ev, color="darkred", linewidth=2.5,
                label=f"Evicted Only (n={n_ev:,})", alpha=0.8)
        ax.fill_between(evicted_sorted, cdf_ev, alpha=0.1, color="red")

    _annotate_percentiles(ax, all_sorted, [50, 75, 90, 95, 99], x_max)

    mean_val = float(np.mean(all_sorted))
    median_val = float(np.median(all_sorted))
    ax.axvline(mean_val, color="blue", linestyle="--", linewidth=2,
               label=f"Mean: {mean_val:.1f}s", alpha=0.7)
    ax.axvline(median_val, color="orange", linestyle="--", linewidth=2,
               label=f"Median: {median_val:.1f}s", alpha=0.7)

    ax.set_xlim([0, x_max])
    ax.set_ylim([0, 105])
    ax.set_xlabel("Physical Lifespan (seconds)", fontweight="bold", fontsize=13)
    ax.set_ylabel("Cumulative Percentage (%)", fontweight="bold", fontsize=13)
    ax.set_title(f"Physical Lifespan CDF - {instance_name} (n={n_all:,})",
                 fontweight="bold", fontsize=14)
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.legend(loc="upper left", fontsize=11, framealpha=0.9)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    print(f"  ✓ {output_path}")
    plt.close()


def plot_active_lifespan_cdf(active_sorted, instance_name, output_path):
    """Active Lifespan CDF"""
    n = len(active_sorted)
    if n == 0:
        print("  跳过 Active Lifespan CDF: 无数据")
        return

    cdf_y = np.arange(1, n + 1) / n * 100
    zero_pct = float((active_sorted == 0).sum()) / n * 100
    p99 = float(np.percentile(active_sorted, 99))
    x_max = min(p99 * 1.5, float(active_sorted[-1])) if active_sorted[-1] > 0 else 1.0

    fig, ax = plt.subplots(figsize=(14, 8))

    ax.plot(active_sorted, cdf_y, color="darkgreen", linewidth=2.5,
            label=f"Active Lifespan CDF (n={n:,})", alpha=0.9)
    ax.fill_between(active_sorted, cdf_y, alpha=0.15, color="green")

    _annotate_percentiles(ax, active_sorted, [50, 75, 90, 95, 99], x_max)

    if zero_pct > 0:
        ax.axhline(zero_pct, color="purple", linestyle="--", linewidth=2.5,
                   label=f"Zero Access: {zero_pct:.1f}%", zorder=4)

    mean_val = float(np.mean(active_sorted))
    median_val = float(np.median(active_sorted))
    ax.axvline(mean_val, color="blue", linestyle="--", linewidth=2,
               label=f"Mean: {mean_val:.1f}s", alpha=0.7)
    ax.axvline(median_val, color="orange", linestyle="--", linewidth=2,
               label=f"Median: {median_val:.1f}s", alpha=0.7)

    ax.set_xlim([0, x_max])
    ax.set_ylim([0, 105])
    ax.set_xlabel("Active Lifespan (seconds)", fontweight="bold", fontsize=13)
    ax.set_ylabel("Cumulative Percentage (%)", fontweight="bold", fontsize=13)
    ax.set_title(f"Active Lifespan CDF - {instance_name}\n"
                 f"(Total: {n:,} blocks, Zero-access: {zero_pct:.1f}%)",
                 fontweight="bold", fontsize=14)
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.legend(loc="lower right", fontsize=11, framealpha=0.9)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    print(f"  ✓ {output_path}")
    plt.close()


def _draw_access_hist(ax, data, max_bin, title, color):
    """在指定 ax 上绘制 access count 直方图"""
    clipped = np.clip(data, 0, max_bin)
    bins = np.arange(0, max_bin + 2) - 0.5
    ax.hist(clipped, bins=bins, color=color, edgecolor="black", linewidth=0.3, alpha=0.8)

    mean_val = float(np.mean(data))
    median_val = float(np.median(data))
    ax.axvline(mean_val, color="blue", linestyle="--", linewidth=2,
               label=f"Mean: {mean_val:.1f}", alpha=0.7)
    ax.axvline(median_val, color="green", linestyle="--", linewidth=2,
               label=f"Median: {median_val:.0f}", alpha=0.7)

    ax.set_xlabel("Access Count", fontweight="bold", fontsize=12)
    ax.set_ylabel("Block Count", fontweight="bold", fontsize=12)
    ax.set_title(title, fontweight="bold", fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis="y")


def plot_access_count_histogram(access_counts, instance_name, output_path, max_bin=100):
    """Access Count 直方图：上图全量，下图去除 zero-access"""
    if len(access_counts) == 0:
        print("  跳过 Access Count 直方图: 无数据")
        return

    nonzero = access_counts[access_counts > 0]
    zero_count = len(access_counts) - len(nonzero)
    total = len(access_counts)

    fig, (ax_all, ax_nz) = plt.subplots(2, 1, figsize=(12, 10))

    _draw_access_hist(ax_all, access_counts, max_bin,
                      f"Access Count (All) - {instance_name}", "coral")
    ax_all.text(0.97, 0.95,
                f"Total: {total:,}\nZero Access: {zero_count:,} ({zero_count/total*100:.1f}%)",
                transform=ax_all.transAxes, fontsize=11, ha="right", va="top",
                bbox=dict(boxstyle="round,pad=0.5", facecolor="lightyellow", alpha=0.9))

    if len(nonzero) > 0:
        _draw_access_hist(ax_nz, nonzero, max_bin,
                          f"Access Count (Excluding Zero) - {instance_name}", "steelblue")
        ax_nz.text(0.97, 0.95,
                   f"Count: {len(nonzero):,}",
                   transform=ax_nz.transAxes, fontsize=11, ha="right", va="top",
                   bbox=dict(boxstyle="round,pad=0.5", facecolor="lightcyan", alpha=0.9))
    else:
        ax_nz.text(0.5, 0.5, "All blocks have zero access",
                   transform=ax_nz.transAxes, fontsize=14, ha="center", va="center")
        ax_nz.set_title(f"Access Count (Excluding Zero) - {instance_name}",
                        fontweight="bold", fontsize=13)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    print(f"  ✓ {output_path}")
    plt.close()
