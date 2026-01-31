#!/usr/bin/env python3
"""
导出并可视化前缀树的完整示例脚本

使用流程：
1. 加载配置文件
2. 运行 optimizer 分析
3. 导出前缀树数据
4. 可视化前缀树
"""

import argparse
import json
import sys
from pathlib import Path

try:
    from kv_cache_manager.optimizer.pybind import kvcm_py_optimizer
except ImportError:
    print("Error: kvcm_py_optimizer module not found. Please build the project first.")
    sys.exit(1)

import plot_radix_tree
import optimizer_analysis_utils as utils


def export_radix_tree_to_json(export_data, output_path):
    """
    将导出的前缀树数据保存为 JSON 文件

    Args:
        export_data: 导出的前缀树数据
        output_path: 输出文件路径
    """
    # 转换为可序列化的格式
    serializable_data = {}

    for instance_id, tree_export in export_data.items():
        # 转换 nodes
        nodes_list = []
        for node in tree_export.nodes:
            node_dict = {
                'node_id': node.node_id,
                'access_count': node.access_count,
                'last_access_time': node.last_access_time,
                'total_blocks': list(node.total_blocks),
                'cached_blocks': list(node.cached_blocks),
                'is_leaf': node.is_leaf,
                'parent_id': node.parent_id
            }
            nodes_list.append(node_dict)

        # 转换 edges
        edges_list = [(edge[0], edge[1]) for edge in tree_export.edges]

        serializable_data[instance_id] = {
            'instance_id': tree_export.instance_id,
            'nodes': nodes_list,
            'edges': edges_list
        }

    # 保存到文件
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump(serializable_data, f, indent=2)

    print(f"[INFO] Exported radix tree data to: {output_path}")
    return output_path


def main():
    parser = argparse.ArgumentParser(
        description='Export and visualize radix tree from optimizer',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run optimizer and visualize hot paths
  %(prog)s -c config.yaml --show-hot-paths --hot-nodes 20
  
  # Show detailed block sequences
  %(prog)s -c config.yaml --show-hot-paths --show-blocks --hot-nodes 10
  
  # Only export and visualize (skip optimizer run)
  %(prog)s -c config.yaml --skip-run --show-hot-paths
  
  # Use custom layout
  %(prog)s -c config.yaml --show-hot-paths --layout custom
        """
    )
    parser.add_argument('-c', '--config', required=True,
                        help='Optimizer config file path')
    parser.add_argument('-o', '--output-dir', default=None,
                        help='Output directory for exported data and visualizations')
    parser.add_argument('--hot-nodes', type=int, default=10,
                        help='Number of hot nodes to highlight (default: 10)')
    parser.add_argument('--show-hot-paths', action='store_true',
                        help='Visualize hot paths only (recommended for large trees)')
    parser.add_argument('--show-blocks', action='store_true',
                        help='Show detailed block sequences for hot paths')
    parser.add_argument('--max-blocks', type=int, default=100,
                        help='Maximum blocks to display per node (default: 100)')
    parser.add_argument('--skip-run', action='store_true',
                        help='Skip running optimizer, only export and visualize existing data')
    parser.add_argument('--stats-only', action='store_true',
                        help='Only show statistics, no visualization')
    parser.add_argument('--layout', choices=['auto', 'graphviz', 'custom'], default='auto',
                        help='Layout algorithm: auto (default), graphviz, or custom')
    parser.add_argument('--node-size', type=int, default=2000,
                        help='Base node size for visualization (default: 2000)')

    args = parser.parse_args()

    # 确定输出目录
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        # 使用配置文件中的输出路径
        config_loader = kvcm_py_optimizer.OptimizerConfigLoader()
        if not config_loader.load(args.config):
            print("[ERROR] Failed to load config")
            sys.exit(1)
        config = config_loader.config()
        output_dir = Path(config.output_result_path())

    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("Radix Tree Export and Visualization")
    print("=" * 80)
    print(f"Config: {args.config}")
    print(f"Output directory: {output_dir}")
    print(f"Layout: {args.layout}")
    print()

    # 初始化 logger
    utils.init_kvcm_logger()

    # 加载配置
    config_loader = kvcm_py_optimizer.OptimizerConfigLoader()
    if not config_loader.load(args.config):
        print("[ERROR] Failed to load config")
        sys.exit(1)
    config = config_loader.config()

    # 创建 OptimizerManager
    optimizer = kvcm_py_optimizer.OptimizerManager(config)

    # 初始化
    if not optimizer.Init():
        print("[ERROR] Failed to initialize optimizer")
        sys.exit(1)

    # 运行 optimizer（如果需要）
    if not args.skip_run:
        print("[INFO] Running optimizer analysis...")
        optimizer.DirectRun()
        print("[INFO] Optimizer analysis completed")
        print()

        # 分析结果
        print("[INFO] Analyzing results...")
        optimizer.AnalyzeResults()
        print("[INFO] Results analysis completed")
        print()

    # 导出前缀树
    print("[INFO] Exporting radix trees...")
    export_data = optimizer.ExportRadixTrees()

    if not export_data:
        print("[WARNING] No radix tree data exported")
        sys.exit(0)

    print(f"[INFO] Exported {len(export_data)} instance(s)")
    print()

    # 保存为 JSON
    for instance_id, tree_export in export_data.items():
        json_path = output_dir / f"{instance_id}_radix_tree.json"
        export_radix_tree_to_json({instance_id: tree_export}, json_path)

    # 可视化和统计
    print("=" * 80)
    print("GLOBAL STATISTICS (Full Tree)")
    print("=" * 80)

    for instance_id, tree_export in export_data.items():
        print(f"\n{'─' * 80}")
        print(f"Instance: {instance_id}")
        print(f"{'─' * 80}")

        # 直接使用已保存的 JSON 文件
        json_path = output_dir / f"{instance_id}_radix_tree.json"

        # 加载数据
        with open(json_path, 'r') as f:
            data = json.load(f)

        visualizer = plot_radix_tree.RadixTreeVisualizer(data[instance_id])

        # 打印全局统计信息
        visualizer.print_statistics()
        
        # 打印热点节点
        visualizer.print_hot_nodes(args.hot_nodes)

        # 显示详细的 block 序列（如果启用）
        if args.show_blocks:
            visualizer.print_hot_paths_with_blocks(args.hot_nodes, args.max_blocks)

        # 如果只显示统计信息，跳过可视化
        if args.stats_only:
            print(f"\n[INFO] Statistics displayed for {instance_id}")
            continue

        # 生成可视化
        print(f"\n[INFO] Generating visualization for {instance_id}...")
        
        if args.show_hot_paths:
            output_image = output_dir / f"{instance_id}_hot_paths.png"
            visualizer.visualize_hot_paths(
                output_path=str(output_image),
                top_k=args.hot_nodes
            )
        else:
            output_image = output_dir / f"{instance_id}_radix_tree.png"
            
            # 根据 layout 参数决定强制使用的布局
            force_layout = None if args.layout == 'auto' else args.layout
            
            visualizer.visualize_tree(
                output_path=str(output_image),
                show_labels=True,
                node_size=args.node_size,
                highlight_hot_nodes=True,
                top_k_hot=args.hot_nodes,
                force_layout=force_layout
            )

    print("\n" + "=" * 80)
    print("Export and visualization completed!")
    print("=" * 80)
    print(f"Output directory: {output_dir}")
    print("\nGenerated files:")
    for instance_id in export_data.keys():
        json_file = output_dir / f"{instance_id}_radix_tree.json"
        print(f"  - {json_file}")
        
        if not args.stats_only:
            if args.show_hot_paths:
                img_file = output_dir / f"{instance_id}_hot_paths.png"
            else:
                img_file = output_dir / f"{instance_id}_radix_tree.png"
            print(f"  - {img_file}")
    
    print("=" * 80)

    # 清理 logger
    kvcm_py_optimizer.LoggerBroker.DestroyLogger()


if __name__ == "__main__":
    main()
