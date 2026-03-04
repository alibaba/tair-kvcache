#!/usr/bin/env python3
"""
前缀树可视化工具
"""

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional
import numpy as np

try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    from matplotlib.patches import Patch
    import networkx as nx
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("Warning: matplotlib or networkx not installed. Install with: pip install matplotlib networkx")


class RadixTreeVisualizer:
    def __init__(self, export_data):
        """
        初始化可视化器

        Args:
            export_data: 导出的前缀树数据 (dict)
        """
        self.data = export_data
        self.instance_id = export_data.get('instance_id', 'unknown')
        self.nodes = export_data.get('nodes', [])
        self.edges = export_data.get('edges', [])

        # 创建 NetworkX 图
        self.graph = nx.DiGraph()
        
        # 创建快速查找映射
        self.node_map: Dict[str, dict] = {}  # node_id -> node_data
        self.parent_map: Dict[str, str] = {}  # child_id -> parent_id
        self.children_map: Dict[str, List[str]] = {}  # parent_id -> [child_ids]
        
        self._build_graph()
        self._build_lookup_maps()

    def _build_graph(self):
        """构建图结构"""
        # 添加节点
        for node_data in self.nodes:
            node_id = node_data.get('node_id', '')
            self.graph.add_node(node_id, **node_data)

        # 添加边
        for source, target in self.edges:
            self.graph.add_edge(source, target)

    def _build_lookup_maps(self):
        """构建快速查找映射，优化性能"""
        for node_data in self.nodes:
            node_id = node_data.get('node_id', '')
            parent_id = node_data.get('parent_id', '')
            
            self.node_map[node_id] = node_data
            
            if parent_id:
                self.parent_map[node_id] = parent_id
                if parent_id not in self.children_map:
                    self.children_map[parent_id] = []
                self.children_map[parent_id].append(node_id)
    
    def _get_block_count(self, node_data: dict, key: str) -> int:
        """
        获取 block 数量，兼容新旧格式
        
        Args:
            node_data: 节点数据
            key: 'cached_blocks' 或 'total_blocks'
        
        Returns:
            block 数量
        """
        value = node_data.get(key, 0)
        if isinstance(value, (int, float)):
            return int(value)
        else:
            return len(value) if value else 0
    
    def _get_block_list(self, node_data: dict, key: str) -> List:
        """
        获取 block 列表，兼容新旧格式
        
        Args:
            node_data: 节点数据
            key: 'cached_blocks' 或 'total_blocks'
        
        Returns:
            block 列表
        """
        value = node_data.get(key, [])
        if isinstance(value, list):
            return value
        else:
            return []

    def get_root_node(self) -> Optional[str]:
        """获取根节点ID"""
        for node_id in self.graph.nodes():
            if self.graph.in_degree(node_id) == 0:
                return node_id
        return None

    def get_node_depth(self, node_id: str) -> int:
        """获取节点深度（从根节点开始）"""
        depth = 0
        current = node_id
        while current in self.parent_map:
            depth += 1
            current = self.parent_map[current]
        return depth

    def get_path_to_root(self, node_id: str) -> List[str]:
        """获取从节点到根的路径（优化版）"""
        path = [node_id]
        current = node_id
        while current in self.parent_map:
            current = self.parent_map[current]
            path.append(current)
        return path[::-1]  # 反转，从根到节点

    def get_hot_nodes(self, top_k=10, by='access_count'):
        """
        获取热点节点

        Args:
            top_k: 返回前 k 个热点节点
            by: 排序依据 ('access_count', 'cached_blocks', 'total_blocks', 'cache_ratio')

        Returns:
            热点节点列表
        """
        if by == 'cache_ratio':
            # 特殊处理缓存率
            sorted_nodes = sorted(
                self.nodes,
                key=lambda x: self._get_block_count(x, 'cached_blocks') / max(1, self._get_block_count(x, 'total_blocks')),
                reverse=True
            )
        else:
            sorted_nodes = sorted(
                self.nodes,
                key=lambda x: x.get(by, 0),
                reverse=True
            )
        return sorted_nodes[:top_k]

    def calculate_statistics(self) -> Dict:
        """计算树的统计信息"""
        stats = {
            'total_nodes': len(self.nodes),
            'total_edges': len(self.edges),
            'total_access': 0,
            'total_blocks': 0,
            'total_cached_blocks': 0,
            'leaf_nodes': 0,
            'non_leaf_nodes': 0,
            'max_depth': 0,
            'avg_depth': 0,
            'nodes_with_access': 0,
            'cache_ratio': 0.0,
            'depth_distribution': {},
        }

        depths = []
        for node in self.nodes:
            node_id = node.get('node_id', '')
            access_count = node.get('access_count', 0)
            total_blocks = self._get_block_count(node, 'total_blocks')
            cached_blocks = self._get_block_count(node, 'cached_blocks')
            is_leaf = node.get('is_leaf', False)

            stats['total_access'] += access_count
            stats['total_blocks'] += total_blocks
            stats['total_cached_blocks'] += cached_blocks

            if is_leaf:
                stats['leaf_nodes'] += 1
            else:
                stats['non_leaf_nodes'] += 1

            if access_count > 0:
                stats['nodes_with_access'] += 1

            # 深度统计
            depth = self.get_node_depth(node_id)
            depths.append(depth)
            stats['depth_distribution'][depth] = stats['depth_distribution'].get(depth, 0) + 1

        if depths:
            stats['max_depth'] = max(depths)
            stats['avg_depth'] = np.mean(depths)

        if stats['total_blocks'] > 0:
            stats['cache_ratio'] = stats['total_cached_blocks'] / stats['total_blocks']

        return stats

    def print_statistics(self):
        """打印详细统计信息"""
        stats = self.calculate_statistics()

        print(f"\n{'=' * 80}")
        print(f"Radix Tree Statistics - Instance: {self.instance_id}")
        print(f"{'=' * 80}")
        
        print(f"\n[Tree Structure]")
        print(f"  Total Nodes: {stats['total_nodes']}")
        print(f"  Total Edges: {stats['total_edges']}")
        print(f"  Leaf Nodes: {stats['leaf_nodes']}")
        print(f"  Non-Leaf Nodes: {stats['non_leaf_nodes']}")
        print(f"  Max Depth: {stats['max_depth']}")
        print(f"  Avg Depth: {stats['avg_depth']:.2f}")

        print(f"\n[Access Statistics]")
        print(f"  Total Access: {stats['total_access']}")
        print(f"  Active Nodes: {stats['nodes_with_access']} ({stats['nodes_with_access']/max(1, stats['total_nodes'])*100:.1f}%)")

        print(f"\n[Cache Statistics]")
        print(f"  Total Blocks: {stats['total_blocks']}")
        print(f"  Cached Blocks: {stats['total_cached_blocks']}")
        print(f"  Cache Ratio: {stats['cache_ratio']*100:.2f}%")

        print(f"\n[Depth Distribution]")
        for depth in sorted(stats['depth_distribution'].keys())[:10]:
            count = stats['depth_distribution'][depth]
            print(f"  Depth {depth}: {count} nodes")
        if len(stats['depth_distribution']) > 10:
            print(f"  ... ({len(stats['depth_distribution'])} levels total)")

    def print_hot_nodes(self, top_k=10):
        """打印热点节点信息"""
        print(f"\n{'=' * 80}")
        print(f"Top {top_k} Hot Nodes (by access_count)")
        print(f"{'=' * 80}")

        for i, node in enumerate(self.get_hot_nodes(top_k, 'access_count'), 1):
            nid = node.get('node_id', 'unknown')
            nc = self._get_block_count(node, 'cached_blocks')
            nt = self._get_block_count(node, 'total_blocks')
            ratio = nc / max(1, nt)

            print(f"\n#{i} Node ID: {nid}")
            print(f"  Access: {node.get('access_count', 0)}  Depth: {self.get_node_depth(nid)}  Leaf: {node.get('is_leaf', False)}")
            print(f"  Cached: {nc}/{nt} ({ratio*100:.1f}%)  Last Access: {node.get('last_access_time', 0)}")

            for key, label in [('total_blocks', 'Total'), ('cached_blocks', 'Cached')]:
                blocks = self._get_block_list(node, key)
                if blocks:
                    preview = str(blocks[:20])
                    if len(blocks) > 20:
                        preview += f" ... ({len(blocks)} total)"
                    print(f"  {label} Blocks: {preview}")
    
    _NODE_TYPE_LABELS = {
        "hot": "HOT NODE", "root": "ROOT", "leaf": "LEAF", "internal": "INTERNAL"
    }

    def print_hot_paths_with_blocks(self, top_k=10, max_blocks_per_node=50):
        """打印热点路径信息（含每节点 block 序列）"""
        print(f"\n{'=' * 80}\nTop {top_k} Hot Paths with Block Sequences\n{'=' * 80}")

        for i, node in enumerate(self.get_hot_nodes(top_k, 'access_count'), 1):
            nid = node.get('node_id', 'unknown')
            print(f"\n{'─' * 80}\nHot Path #{i} (Access: {node.get('access_count', 0)})\n{'─' * 80}")

            path = self.get_path_to_root(nid)
            path.reverse()
            accumulated = []

            for depth, pid in enumerate(path):
                if pid not in self.node_map:
                    continue
                nd = self.node_map[pid]
                kind = "hot" if pid == nid else ("root" if depth == 0 else ("leaf" if nd.get('is_leaf') else "internal"))
                blocks = self._get_block_list(nd, 'total_blocks')[:max_blocks_per_node]
                new_blocks = [b for b in blocks if b not in accumulated]

                print(f"\n  [{self._NODE_TYPE_LABELS[kind]}] Depth {depth}  ID: {pid}")
                print(f"  Access: {nd.get('access_count', 0)}  Cached: {self._get_block_count(nd, 'cached_blocks')}/{self._get_block_count(nd, 'total_blocks')}")

                if blocks:
                    if depth == 0:
                        print(f"  Blocks ({len(self._get_block_list(nd, 'total_blocks'))} total): {blocks}")
                    else:
                        print(f"  New: {new_blocks or '(none)'}  Accumulated: {accumulated + new_blocks}")
                    full_len = len(self._get_block_list(nd, 'total_blocks'))
                    if full_len > max_blocks_per_node:
                        print(f"  ... and {full_len - max_blocks_per_node} more")
                    accumulated.extend(new_blocks)

            print(f"\n  {'─' * 76}\n  Path Blocks ({len(accumulated)}): {accumulated}\n  {'─' * 76}")

    def _hierarchical_layout_improved(self, graph: nx.DiGraph = None) -> Dict[str, Tuple[float, float]]:
        """子树宽度驱动的层次化布局（避免节点重叠）"""
        if graph is None:
            graph = self.graph

        root = next((n for n in graph.nodes() if graph.in_degree(n) == 0), list(graph.nodes())[0])

        subtree_sizes: Dict[str, int] = {}
        def calc_size(nid: str) -> int:
            if nid in subtree_sizes:
                return subtree_sizes[nid]
            children = list(graph.successors(nid))
            subtree_sizes[nid] = sum(calc_size(c) for c in children) if children else 1
            return subtree_sizes[nid]
        calc_size(root)

        pos: Dict[str, Tuple[float, float]] = {}
        vertical_spacing = 3.0
        horizontal_spacing = 2.5
        next_x = [0]

        def assign_positions(node_id: str, depth: int) -> Tuple[float, float]:
            """后序遍历分配位置, 返回 (left_x, right_x)
            """
            children = list(graph.successors(node_id))
            
            if not children:
                # 叶子节点：使用下一个可用的x位置
                x = next_x[0]
                next_x[0] += horizontal_spacing
                y = -depth * vertical_spacing
                pos[node_id] = (x, y)
                return x, x
            
            # 非叶子节点：先处理所有子节点
            child_positions = []
            for child in children:
                left_x, right_x = assign_positions(child, depth + 1)
                child_positions.append((left_x, right_x))
            
            # 父节点位于所有子节点的中心
            leftmost = child_positions[0][0]
            rightmost = child_positions[-1][1]
            x = (leftmost + rightmost) / 2
            y = -depth * vertical_spacing
            pos[node_id] = (x, y)
            
            return leftmost, rightmost
        
        assign_positions(root, 0)
        
        return pos

    def _get_best_layout(self) -> str:
        """选择最佳布局算法: pygraphviz > pydot > 自定义"""
        test_g = nx.DiGraph()
        test_g.add_edge(1, 2)

        for name, layout_fn in [
            ("graphviz_agraph", lambda g: nx.nx_agraph.graphviz_layout(g, prog="dot")),
            ("graphviz_pydot",  lambda g: nx.nx_pydot.graphviz_layout(g, prog="dot")),
        ]:
            try:
                layout_fn(test_g)
                return name
            except Exception:
                pass

        return "custom_hierarchical"

    def _compute_layout(self, layout_type: str, graph: nx.DiGraph = None) -> Dict:
        """计算节点布局位置"""
        if graph is None:
            graph = self.graph
        
        num_nodes = len(graph.nodes())

        layout_fns = {
            "graphviz_agraph": lambda: nx.nx_agraph.graphviz_layout(
                graph, prog="dot", args="-Granksep=2.0 -Gnodesep=1.5"),
            "graphviz_pydot": lambda: nx.nx_pydot.graphviz_layout(graph, prog="dot"),
        }

        if layout_type in layout_fns:
            try:
                pos = layout_fns[layout_type]()
                if pos and len(pos) == num_nodes:
                    print(f"[INFO] Using {layout_type} layout ({num_nodes} nodes)")
                    return pos
                print(f"[WARNING] {layout_type} returned invalid layout")
            except Exception as e:
                print(f"[WARNING] {layout_type} failed: {e}")

        print(f"[INFO] Using custom hierarchical layout ({num_nodes} nodes)")
        return self._hierarchical_layout_improved(graph)

    # 热度阈值 -> 颜色映射（降序匹配）
    _HEAT_COLORS = [(0.7, '#d32f2f'), (0.4, '#f57c00'), (0.2, '#fbc02d'), (0.0, '#1976d2')]

    def _prepare_node_styles(self, hot_node_ids: Set[str], base_size: int, graph: nx.DiGraph = None) -> Tuple[List, List, List, List, List]:
        """准备节点的颜色/大小/透明度/边框颜色/边框宽度"""
        if graph is None:
            graph = self.graph

        max_access = max((graph.nodes[n].get('access_count', 0) for n in graph.nodes()), default=1) or 1
        node_colors, node_sizes, node_alphas, edge_colors, edge_widths = [], [], [], [], []

        for node_id in graph.nodes():
            data = graph.nodes[node_id]
            acc = data.get('access_count', 0)
            cached = self._get_block_count(data, 'cached_blocks')

            node_sizes.append(base_size * (0.5 + math.log(cached + 1) / 3 if cached > 0 else 0.3))

            if acc > 0:
                heat = acc / max_access
                color = next(c for t, c in self._HEAT_COLORS if heat > t)
                node_colors.append(color)
                node_alphas.append(0.9)
            else:
                node_colors.append('#bdbdbd')
                node_alphas.append(0.5)

            is_hot = node_id in hot_node_ids
            edge_colors.append('#FFD700' if is_hot else 'white')
            edge_widths.append(4 if is_hot else 2)

        return node_colors, node_sizes, node_alphas, edge_colors, edge_widths

    def _generate_node_labels(self, hot_node_ids: Set[str], graph: nx.DiGraph = None) -> Dict[str, str]:
        """生成节点标签"""
        if graph is None:
            graph = self.graph
        
        # 找到根节点
        root_id = None
        for node_id in graph.nodes():
            if graph.in_degree(node_id) == 0:
                root_id = node_id
                break
            
        labels = {}
        for node_id in graph.nodes():
            node_data = graph.nodes[node_id]
            access_count = node_data.get('access_count', 0)
            cached_blocks = self._get_block_count(node_data, 'cached_blocks')
            total_blocks = self._get_block_count(node_data, 'total_blocks')

            # 根节点特殊显示
            if node_id == root_id:
                labels[node_id] = "ROOT"
            # 为热点节点显示详细信息（红边框已经标识了，不需要HOT文字）
            elif node_id in hot_node_ids:
                labels[node_id] = f"Acc:{access_count}\n{cached_blocks}/{total_blocks}"
            elif access_count > 0:
                labels[node_id] = f"{access_count}"
            else:
                labels[node_id] = ""  # 无访问的节点不显示标签

        return labels

    def _build_legend(self, ax, graph, top_k, marker_size=12):
        """构建访问热度图例（full tree 和 hot paths 共用）"""
        from matplotlib.lines import Line2D

        all_access = [graph.nodes[n].get('access_count', 0) for n in graph.nodes()]
        max_acc = max(all_access) if all_access else 1
        ms = marker_size

        elements = [
            Line2D([0], [0], marker='o', color='w', markerfacecolor='#d32f2f',
                   markersize=ms, label=f'Very Hot (>{int(max_acc*0.7)} acc, >70%)'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='#f57c00',
                   markersize=ms, label=f'Hot ({int(max_acc*0.4)}-{int(max_acc*0.7)} acc, 40-70%)'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='#fbc02d',
                   markersize=ms, label=f'Warm ({int(max_acc*0.2)}-{int(max_acc*0.4)} acc, 20-40%)'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='#1976d2',
                   markersize=ms, label=f'Cold (<{int(max_acc*0.2)} acc, <20%)'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='#bdbdbd',
                   markersize=ms - 2, label='Inactive (0 access)'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='#d32f2f',
                   markeredgecolor='#FFD700', markeredgewidth=3,
                   markersize=ms, label=f'Top {top_k} Hot Nodes (Gold Border)'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='gray',
                   markersize=ms + 4, label='Node Size = Cached Blocks'),
        ]
        ax.legend(handles=elements, loc='upper left', fontsize=8,
                  title=f'Legend (Max Access: {max_acc})', title_fontsize=9)

    def visualize_tree(self, output_path=None, show_labels=True, node_size=2000,
                       highlight_hot_nodes=True, top_k_hot=10, max_nodes=500, force_layout=None):
        """
        可视化前缀树 - 使用树形布局

        Args:
            output_path: 输出图片路径，如果为 None 则显示
            show_labels: 是否显示节点标签
            node_size: 基础节点大小
            highlight_hot_nodes: 是否高亮热点节点
            top_k_hot: 高亮前 k 个热点节点
            max_nodes: 最大显示节点数（超过则警告）
            force_layout: 强制使用的布局类型 ('graphviz', 'custom', None=auto)
        """
        if not HAS_MATPLOTLIB:
            print("Error: matplotlib not installed. Cannot visualize.")
            return

        num_nodes = len(self.graph.nodes())
        
        # 大规模树警告
        if num_nodes > max_nodes:
            print(f"\n[WARNING] Tree has {num_nodes} nodes, exceeds recommended {max_nodes}")
            print(f"   Suggest using --show-hot-paths to visualize hot paths only")
            response = input("   Continue to draw full tree? (y/N): ")
            if response.lower() != 'y':
                print("   Visualization cancelled")
                return

        # 根据节点数量自适应调整图形大小
        figsize_x = max(20, min(50, num_nodes * 0.3))
        figsize_y = max(15, min(40, num_nodes * 0.25))

        fig, ax = plt.subplots(figsize=(figsize_x, figsize_y))

        # 使用层次布局（树形）
        if force_layout == 'custom':
            layout_used = 'custom_hierarchical'
        elif force_layout == 'graphviz':
            layout_used = 'graphviz_pydot'  # 尝试使用 graphviz
        else:
            layout_used = self._get_best_layout()
        pos = self._compute_layout(layout_used)

        # 获取热点节点
        hot_node_ids = set()
        if highlight_hot_nodes:
            hot_nodes = self.get_hot_nodes(top_k_hot, 'access_count')
            hot_node_ids = {node.get('node_id') for node in hot_nodes}

        # 绘制边（先绘制，确保在节点下方）
        nx.draw_networkx_edges(self.graph, pos, alpha=0.4, arrows=True,
                              arrowsize=15, edge_color='#666666', width=1.5,
                              connectionstyle='arc3,rad=0.1', ax=ax)

        # 准备节点颜色和大小
        node_colors, node_sizes, node_alphas, edge_colors, edge_widths = self._prepare_node_styles(
            hot_node_ids, node_size
        )

        # 绘制节点
        nx.draw_networkx_nodes(self.graph, pos, node_color=node_colors,
                              node_size=node_sizes, alpha=node_alphas,
                              edgecolors=edge_colors, linewidths=edge_widths, ax=ax)

        # 绘制标签
        if show_labels:
            labels = self._generate_node_labels(hot_node_ids)
            nx.draw_networkx_labels(self.graph, pos, labels, font_size=7,
                                   font_weight='bold', font_color='black',
                                   bbox=dict(facecolor='white', edgecolor='gray',
                                           boxstyle='round,pad=0.3', alpha=0.8), ax=ax)

        # 添加标题和统计信息
        stats = self.calculate_statistics()
        title_text = (f"Radix Tree Visualization - Instance: {self.instance_id}\n"
                     f"Layout: {layout_used} | Nodes: {num_nodes} | Edges: {len(self.edges)} | "
                     f"Max Depth: {stats['max_depth']}\n"
                     f"Total Access: {stats['total_access']} | Cache Ratio: {stats['cache_ratio']:.2%}")

        ax.set_title(title_text, fontsize=14, fontweight='bold', pad=20)

        self._build_legend(ax, self.graph, top_k_hot, marker_size=12)

        ax.axis('off')
        plt.tight_layout()

        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"Tree visualization saved to: {output_path}")
        else:
            plt.show()

        plt.close()

    def visualize_hot_paths(self, output_path=None, top_k=10):
        """
        可视化热点路径（从根到热点节点的路径）

        Args:
            output_path: 输出图片路径
            top_k: 显示前 k 条热点路径
        """
        if not HAS_MATPLOTLIB:
            print("Error: matplotlib not installed. Cannot visualize.")
            return

        hot_nodes = self.get_hot_nodes(top_k, 'access_count')

        # 创建子图，只包含热点路径
        subgraph = nx.DiGraph()
        nodes_to_include = set()

        for node in hot_nodes:
            node_id = node.get('node_id')
            # 使用优化的路径查找
            path = self.get_path_to_root(node_id)
            nodes_to_include.update(path)

        # 添加节点和边到子图
        for node_id in nodes_to_include:
            if node_id in self.node_map:
                subgraph.add_node(node_id, **self.node_map[node_id])

        for source, target in self.edges:
            if source in nodes_to_include and target in nodes_to_include:
                subgraph.add_edge(source, target)

        # 可视化
        num_nodes = len(subgraph.nodes())
        figsize_x = max(16, min(35, num_nodes * 0.5))
        figsize_y = max(12, min(28, num_nodes * 0.4))

        fig, ax = plt.subplots(figsize=(figsize_x, figsize_y))

        # 使用层次布局
        layout_used = self._get_best_layout()
        pos = self._compute_layout(layout_used, subgraph)

        # 获取热点节点ID
        hot_node_ids = {node.get('node_id') for node in hot_nodes}

        # 绘制边
        nx.draw_networkx_edges(subgraph, pos, alpha=0.6, arrows=True,
                              arrowsize=18, edge_color='#666666', width=2.5,
                              connectionstyle='arc3,rad=0.1', ax=ax)

        # 准备节点样式
        node_colors, node_sizes, node_alphas, edge_colors, edge_widths = self._prepare_node_styles(
            hot_node_ids, 3000, subgraph
        )

        # 绘制节点
        nx.draw_networkx_nodes(subgraph, pos, node_color=node_colors,
                              node_size=node_sizes, alpha=node_alphas,
                              edgecolors=edge_colors, linewidths=edge_widths, ax=ax)

        labels = self._generate_node_labels(hot_node_ids, subgraph)

        nx.draw_networkx_labels(subgraph, pos, labels, font_size=9,
                               font_weight='bold', font_color='black',
                               bbox=dict(facecolor='white', edgecolor='gray',
                                       boxstyle='round,pad=0.4', alpha=0.9), ax=ax)

        # 添加标题（包含全局统计信息）
        # 热点节点的统计
        hot_total_access = sum(node.get('access_count', 0) for node in hot_nodes)
        
        # 全局统计（从完整的图获取）
        global_stats = self.calculate_statistics()
        
        title_text = (f"Top {top_k} Hot Paths - Instance: {self.instance_id}\n"
                     f"Layout: {layout_used} | "
                     f"Showing {num_nodes}/{global_stats['total_nodes']} nodes | "
                     f"Hot Access: {hot_total_access}/{global_stats['total_access']} "
                     f"({hot_total_access/max(1, global_stats['total_access'])*100:.1f}%)\n"
                     f"Global: {global_stats['total_nodes']} nodes, "
                     f"{global_stats['total_cached_blocks']}/{global_stats['total_blocks']} cached "
                     f"({global_stats['cache_ratio']:.1%})")

        ax.set_title(title_text, fontsize=14, fontweight='bold', pad=20)

        self._build_legend(ax, subgraph, top_k, marker_size=14)

        ax.axis('off')
        plt.tight_layout()

        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"Hot paths visualization saved to: {output_path}")
        else:
            plt.show()

        plt.close()


def load_export_data(json_path):
    """加载导出的 JSON 数据"""
    with open(json_path, 'r') as f:
        return json.load(f)


def main():
    parser = argparse.ArgumentParser(
        description='Radix Tree Visualization Tool (Fixed Version)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Show statistics only
  %(prog)s -i tree_export.json --stats
  
  # Draw full tree structure
  %(prog)s -i tree_export.json -o tree_full.png
  
  # Draw hot paths only (recommended)
  %(prog)s -i tree_export.json -o tree_hot.png --show-hot-paths --hot-nodes 15
        """
    )
    parser.add_argument('-i', '--input', required=True,
                        help='Input JSON file path (exported radix tree data)')
    parser.add_argument('-o', '--output', default=None,
                        help='Output image path (default: interactive display)')
    parser.add_argument('--hot-nodes', type=int, default=10,
                        help='Number of hot nodes (default: 10)')
    parser.add_argument('--show-hot-paths', action='store_true',
                        help='Visualize hot paths only (recommended for large trees)')
    parser.add_argument('--node-size', type=int, default=2000,
                        help='Base node size (default: 2000)')
    parser.add_argument('--no-labels', action='store_true',
                        help='Do not show node labels')
    parser.add_argument('--stats', action='store_true',
                        help='Show statistics only, no visualization')
    parser.add_argument('--max-nodes', type=int, default=500,
                        help='Warning threshold for full tree visualization (default: 500)')
    parser.add_argument('--layout', choices=['auto', 'graphviz', 'custom'], default='auto',
                        help='Layout algorithm: auto (default), graphviz (dot), or custom (hierarchical)')
    parser.add_argument('--show-blocks', action='store_true',
                        help='Show detailed block sequences for hot paths')
    parser.add_argument('--max-blocks', type=int, default=50,
                        help='Maximum blocks to display per node (default: 50)')

    args = parser.parse_args()

    if not Path(args.input).exists():
        print(f"[ERROR] Input file not found: {args.input}")
        sys.exit(1)

    # Load data
    print(f"[INFO] Loading data from: {args.input}")
    try:
        export_data = load_export_data(args.input)
    except json.JSONDecodeError as e:
        print(f"[ERROR] JSON parsing failed: {e}")
        sys.exit(1)

    # Create visualizer
    visualizer = RadixTreeVisualizer(export_data)

    # Show statistics (always global, even in hot-paths mode)
    print("\n" + "="*80)
    print("GLOBAL STATISTICS (Full Tree)")
    print("="*80)
    visualizer.print_statistics()
    visualizer.print_hot_nodes(args.hot_nodes)
    
    # Show detailed block sequences if requested
    if args.show_blocks:
        visualizer.print_hot_paths_with_blocks(args.hot_nodes, args.max_blocks)

    # If stats only, exit
    if args.stats:
        print("\n[DONE] Statistics displayed")
        return

    # Visualize tree structure
    if args.show_hot_paths:
        output_path = args.output
        if output_path and not output_path.endswith('_hot_paths.png'):
            base, ext = output_path.rsplit('.', 1) if '.' in output_path else (output_path, 'png')
            output_path = f"{base}_hot_paths.{ext}"

        visualizer.visualize_hot_paths(
            output_path=output_path,
            top_k=args.hot_nodes
        )
    else:
        visualizer.visualize_tree(
            output_path=args.output,
            show_labels=not args.no_labels,
            node_size=args.node_size,
            highlight_hot_nodes=True,
            top_k_hot=args.hot_nodes,
            max_nodes=args.max_nodes,
            force_layout=None if args.layout == 'auto' else args.layout
        )

    print("\n[DONE] Visualization complete!")


if __name__ == "__main__":
    main()
