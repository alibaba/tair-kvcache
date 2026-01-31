#!/usr/bin/env python3

import os
import sys
import argparse

from kv_cache_manager.optimizer.pybind import kvcm_py_optimizer
from kv_cache_manager.optimizer.analysis.script.plot_hit_rate_with_storage import plot_multi_instance_analysis as generate_plot

def parse_args():
    parser = argparse.ArgumentParser(
        description='运行优化器分析并可选地生成图表'
    )
    parser.add_argument(
        '-c', '--config',
        type=str,
        required=True,
        help='优化器启动配置文件路径 (JSON格式)'
    )
    parser.add_argument(
        '--draw-chart',
        action='store_true',
        default=False,
        help='是否生成图表 (默认: 不生成)'
    )
    return parser.parse_args()


def main():
    args = parse_args()

    kvcm_py_optimizer.LoggerBroker.InitLogger("")
    kvcm_py_optimizer.LoggerBroker.SetLogLevel(4)
    draw_chart = args.draw_chart

    print(f"Loading config from {args.config}")
    config_loader = kvcm_py_optimizer.OptimizerConfigLoader()
    if not config_loader.load(args.config):
        print("Failed to load config")
        sys.exit(1)
    config = config_loader.config()

    manager = kvcm_py_optimizer.OptimizerManager(config)
    if manager is None:
        print("Failed to create OptimizerManager")
        sys.exit(1)

    manager.Init()
    output_result_path = config.output_result_path()
    print(f"Running optimizer analysis, output will be saved to {output_result_path}")

    manager.DirectRun()

    manager.AnalyzeResults()

    if draw_chart:
        generate_plot(output_result_path)
    else:
        print("Skipping chart generation.")

    print("All done.")


if __name__ == "__main__":
    main()