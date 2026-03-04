#!/usr/bin/env python3
"""
前缀树导出与可视化入口

运行 optimizer 后导出 RadixTree 结构，保存为 JSON 并生成可视化图。

用法:
  python run/export_tree.py -c config.json --show-hot-paths
  python run/export_tree.py -c config.json --stats-only
  python plot/radix_tree_plot.py -i tree.json --show-hot-paths  # 从已有 JSON 直接可视化
"""

import argparse
import json
import sys
from pathlib import Path

from kv_cache_manager.optimizer.pybind import kvcm_py_optimizer

from utils.optimizer_runner import init_kvcm_logger
from plot.radix_tree_plot import RadixTreeVisualizer


def export_radix_tree_to_json(export_data, output_path):
    """将导出的前缀树数据序列化为 JSON 文件"""
    serializable = {}
    for instance_id, tree_export in export_data.items():
        nodes_list = [
            {
                "node_id": node.node_id,
                "access_count": node.access_count,
                "last_access_time": node.last_access_time,
                "total_blocks": list(node.total_blocks),
                "cached_blocks": list(node.cached_blocks),
                "is_leaf": node.is_leaf,
                "parent_id": node.parent_id,
            }
            for node in tree_export.nodes
        ]
        edges_list = [(e[0], e[1]) for e in tree_export.edges]
        serializable[instance_id] = {
            "instance_id": tree_export.instance_id,
            "nodes": nodes_list,
            "edges": edges_list,
        }

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(serializable, f, indent=2)
    print("[INFO] Exported radix tree to: {}".format(out))
    return out


def main():
    parser = argparse.ArgumentParser(
        description="Export and visualize radix tree from optimizer",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run/export_tree.py -c config.json --show-hot-paths --hot-nodes 20
  python run/export_tree.py -c config.json --stats-only
        """,
    )
    parser.add_argument("-c", "--config", required=True, help="Optimizer config file")
    parser.add_argument("-o", "--output-dir", default=None, help="Output directory")
    parser.add_argument("--hot-nodes", type=int, default=10)
    parser.add_argument("--show-hot-paths", action="store_true",
                        help="只可视化热点路径（推荐大树使用）")
    parser.add_argument("--show-blocks", action="store_true",
                        help="打印热点路径的完整 block 序列")
    parser.add_argument("--max-blocks", type=int, default=100)
    parser.add_argument("--stats-only", action="store_true",
                        help="只打印统计信息，不生成图片")
    parser.add_argument("--layout", choices=["auto", "graphviz", "custom"], default="auto")
    parser.add_argument("--node-size", type=int, default=2000)
    args = parser.parse_args()

    init_kvcm_logger()

    config_loader = kvcm_py_optimizer.OptimizerConfigLoader()
    if not config_loader.load(args.config):
        print("[ERROR] Failed to load config")
        sys.exit(1)
    config = config_loader.config()

    output_dir = Path(args.output_dir) if args.output_dir else Path(config.output_result_path())
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("Radix Tree Export and Visualization")
    print("=" * 80)
    print("Config: {}".format(args.config))
    print("Output: {}".format(output_dir))
    print()

    optimizer = kvcm_py_optimizer.OptimizerManager(config)
    if not optimizer.Init():
        print("[ERROR] Failed to initialize optimizer")
        sys.exit(1)

    print("[INFO] Running optimizer...")
    optimizer.DirectRun()
    optimizer.AnalyzeResults()
    print("[INFO] Optimizer done\n")

    print("[INFO] Exporting radix trees...")
    export_data = optimizer.ExportRadixTrees()
    if not export_data:
        print("[WARNING] No radix tree data exported")
        sys.exit(0)
    print("[INFO] Exported {} instance(s)\n".format(len(export_data)))

    # 保存 JSON
    for instance_id, tree_export in export_data.items():
        export_radix_tree_to_json(
            {instance_id: tree_export},
            output_dir / "{}_radix_tree.json".format(instance_id),
        )

    # 可视化和统计
    print("=" * 80)
    print("STATISTICS & VISUALIZATION")
    print("=" * 80)

    for instance_id, tree_export in export_data.items():
        print("\n" + "-" * 80)
        print("Instance: {}".format(instance_id))
        print("-" * 80)

        json_path = output_dir / "{}_radix_tree.json".format(instance_id)
        with open(json_path, "r") as f:
            data = json.load(f)

        viz = RadixTreeVisualizer(data[instance_id])
        viz.print_statistics()
        viz.print_hot_nodes(args.hot_nodes)

        if args.show_blocks:
            viz.print_hot_paths_with_blocks(args.hot_nodes, args.max_blocks)

        if args.stats_only:
            continue

        print("\n[INFO] Generating visualization for {}...".format(instance_id))
        force_layout = None if args.layout == "auto" else args.layout

        if args.show_hot_paths:
            out_img = output_dir / "{}_hot_paths.png".format(instance_id)
            viz.visualize_hot_paths(output_path=str(out_img), top_k=args.hot_nodes)
        else:
            out_img = output_dir / "{}_radix_tree.png".format(instance_id)
            viz.visualize_tree(
                output_path=str(out_img),
                show_labels=True,
                node_size=args.node_size,
                highlight_hot_nodes=True,
                top_k_hot=args.hot_nodes,
                force_layout=force_layout,
            )

    print("\n" + "=" * 80)
    print("Done! Output: {}".format(output_dir))
    print("=" * 80)

    kvcm_py_optimizer.LoggerBroker.DestroyLogger()


if __name__ == "__main__":
    main()
