#!/usr/bin/env python3
"""
Block Lifecycle 分析入口

统一入口：读取 → 统计打印 → CDF 绘图 + Access 直方图。
支持单 CSV 文件或目录批量分析。

用法:
  python run/analyze_lifecycle.py -i lifecycle.csv
  python run/analyze_lifecycle.py -i output_dir/ -o results/
  python run/analyze_lifecycle.py -i lifecycle.csv --stats-only
"""

import argparse
import os
import sys
from pathlib import Path

from analysis.lifecycle_analysis import (
    read_lifecycle_csv,
    compute_statistics,
    extract_plot_data,
    print_statistics,
    find_lifecycle_csvs,
)
from plot.lifecycle_plot import (
    plot_physical_lifespan_cdf,
    plot_active_lifespan_cdf,
    plot_access_count_histogram,
)


def analyze_single(csv_path: str, output_dir: str, stats_only: bool = False):
    """分析单个 lifecycle CSV"""
    name = Path(csv_path).stem.replace("_lifecycle", "")
    print(f"\n{'='*60}")
    print(f"分析: {name}")
    print(f"{'='*60}")

    df = read_lifecycle_csv(csv_path)
    if df is None or len(df) == 0:
        print("  跳过 (无有效数据)")
        return

    stats = compute_statistics(df)
    print_statistics(stats, name)

    if stats_only:
        return

    os.makedirs(output_dir, exist_ok=True)
    plot_data = extract_plot_data(df)

    print(f"\n生成图表:")
    plot_physical_lifespan_cdf(
        plot_data["physical_all"], plot_data["physical_evicted"],
        name, os.path.join(output_dir, f"{name}_physical_lifespan_cdf.png"))

    plot_active_lifespan_cdf(
        plot_data["active_all"],
        name, os.path.join(output_dir, f"{name}_active_lifespan_cdf.png"))

    plot_access_count_histogram(
        plot_data["access_counts"],
        name, os.path.join(output_dir, f"{name}_access_count.png"))


def main():
    parser = argparse.ArgumentParser(
        description="Block Lifecycle 分析（统计 + CDF + 直方图）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python run/analyze_lifecycle.py -i lifecycle.csv
  python run/analyze_lifecycle.py -i output_dir/ -o results/
  python run/analyze_lifecycle.py -i lifecycle.csv --stats-only
        """,
    )
    parser.add_argument("-i", "--input", required=True,
                        help="lifecycle CSV 文件或包含 *_lifecycle.csv 的目录")
    parser.add_argument("-o", "--output-dir", default=None,
                        help="图表输出目录 (默认: 输入文件所在目录)")
    parser.add_argument("--stats-only", action="store_true",
                        help="只打印统计信息，不生成图表")
    args = parser.parse_args()

    if args.output_dir is None:
        input_path = Path(args.input)
        args.output_dir = str(input_path if input_path.is_dir() else input_path.parent)

    csvs = find_lifecycle_csvs(args.input)
    if not csvs:
        print(f"错误: 未找到 lifecycle CSV: {args.input}")
        sys.exit(1)

    print(f"找到 {len(csvs)} 个 lifecycle 文件")

    for csv_path in csvs:
        analyze_single(csv_path, args.output_dir, args.stats_only)

    print(f"\n完成!")


if __name__ == "__main__":
    main()
