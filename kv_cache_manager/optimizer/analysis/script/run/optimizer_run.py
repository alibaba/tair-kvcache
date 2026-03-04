#!/usr/bin/env python3
"""
单次运行 optimizer 并可选生成时序图

用法:
  python run/optimizer_run.py -c config.json
  python run/optimizer_run.py -c config.json --draw-chart
  python run/optimizer_run.py -c config.json --export-lifecycle
"""

import argparse
import sys
import time

from kv_cache_manager.optimizer.pybind import kvcm_py_optimizer

from utils.optimizer_runner import init_kvcm_logger
from plot.hit_rate_plot import plot_multi_instance_analysis


def parse_args():
    parser = argparse.ArgumentParser(description="运行优化器分析并可选地生成图表")
    parser.add_argument("-c", "--config", required=True,
                        help="优化器启动配置文件路径 (JSON格式)")
    parser.add_argument("--draw-chart", action="store_true", default=False,
                        help="是否生成时序命中率图表 (默认: 不生成)")
    parser.add_argument("--export-lifecycle", action="store_true", default=False,
                        help="导出 lifecycle CSV（警告：可能生成超大文件）")
    return parser.parse_args()


def main():
    args = parse_args()
    total_start = time.time()

    print("\n" + "=" * 80)
    print("  Optimizer Run")
    print("=" * 80 + "\n")

    t0 = time.time()
    init_kvcm_logger()
    print("[0/4] Logger init: {:.2f}s".format(time.time() - t0))

    t1 = time.time()
    print("\n[1/4] Loading config: {}".format(args.config))
    config_loader = kvcm_py_optimizer.OptimizerConfigLoader()
    if not config_loader.load(args.config):
        print("Failed to load config")
        sys.exit(1)
    config = config_loader.config()
    print("      Config loaded: {:.2f}s".format(time.time() - t1))

    t2 = time.time()
    print("\n[2/4] Creating OptimizerManager...")
    # 关键: 如果要导出lifecycle，必须在构造时开启tracking
    manager = kvcm_py_optimizer.OptimizerManager(config, enable_lifecycle_tracking=args.export_lifecycle)
    if manager is None:
        print("Failed to create OptimizerManager")
        sys.exit(1)
    manager.Init()
    output_path = config.output_result_path()
    print("      Manager init: {:.2f}s".format(time.time() - t2))
    print("      Output: {}".format(output_path))
    if args.export_lifecycle:
        print("      ⚠️  Lifecycle tracking enabled (will use ~10GB extra memory)")

    t3 = time.time()
    print("\n[3/4] Running simulation...")
    print("      Trace: {}".format(config.trace_file_path()))
    manager.DirectRun()
    dur_run = time.time() - t3
    print("      Simulation done: {:.2f}s".format(dur_run))

    t4 = time.time()
    print("\n[4/4] Analyzing results...")
    manager.AnalyzeResults()
    dur_analysis = time.time() - t4
    print("      Analysis done: {:.2f}s".format(dur_analysis))

    if args.draw_chart:
        t5 = time.time()
        print("\n[5/5] Generating charts...")
        plot_multi_instance_analysis(output_path)
        print("      Charts done: {:.2f}s".format(time.time() - t5))
    else:
        print("\n[5/5] Skipping chart generation.")

    total = time.time() - total_start
    print("\n" + "=" * 80)
    print("  Summary: total={:.2f}s  sim={:.2f}s ({:.1f}%)  analysis={:.2f}s ({:.1f}%)".format(
        total, dur_run, dur_run / total * 100, dur_analysis, dur_analysis / total * 100))
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
