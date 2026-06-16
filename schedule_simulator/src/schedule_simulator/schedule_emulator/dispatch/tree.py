import threading
import heapq
from collections import defaultdict
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class Node:
    """基数树节点"""

    # children 的 key 现在是 int (token_id)
    children: Dict[int, "Node"] = field(default_factory=dict)
    # text 现在是 List[int]
    text: List[int] = field(default_factory=list)
    tenant_last_access_time: Dict[str, int] = field(default_factory=dict)
    parent: Optional["Node"] = None
    lock: threading.RLock = field(default_factory=threading.RLock)


@dataclass
class EvictionEntry:
    """驱逐条目"""

    timestamp: int
    tenant: str
    node: Node

    def __lt__(self, other):
        return self.timestamp < other.timestamp


def _shared_prefix_count(a: List[int], b: List[int]) -> int:
    """计算两个列表的共享前缀长度（按 token）"""
    i = 0
    min_len = min(len(a), len(b))
    while i < min_len and a[i] == b[i]:
        i += 1
    return i


class MultiTenantRadixTree:
    """
    线程安全的多租户基数树 (Token ID 版本)

    功能：
    1. 存储多租户数据（多个基数树的重叠）
    2. 节点级锁支持并发访问
    3. 基于租户访问时间的叶子 LRU 驱逐
    """

    def __init__(self):
        """初始化基数树"""
        self.root = Node()
        # 重命名为 token_count 以准确反映数据类型
        self.tenant_token_count: Dict[str, int] = defaultdict(int)
        self.lock = threading.RLock()

    def insert(self, text: List[int], tenant: str, timestamp: float) -> None:
        """
        插入文本到树中

        Args:
            text: 要插入的 token 列表
            tenant: 租户标识
        """

        with self.lock:
            curr = self.root
            curr_idx = 0

            # 更新根节点访问时间
            with curr.lock:
                curr.tenant_last_access_time[tenant] = timestamp

            self.tenant_token_count[tenant] = self.tenant_token_count.get(tenant, 0)

            prev = self.root
            text_count = len(text)

            while curr_idx < text_count:
                first_token = text[curr_idx]
                curr = prev

                with curr.lock:
                    if first_token not in curr.children:
                        # 无匹配，创建新节点
                        curr_text = text[curr_idx:text_count]
                        curr_text_count = len(curr_text)

                        new_node = Node(text=curr_text, parent=curr)
                        new_node.tenant_last_access_time[tenant] = timestamp

                        # 更新 token 计数
                        self.tenant_token_count[tenant] += curr_text_count

                        curr.children[first_token] = new_node
                        prev = new_node
                        curr_idx = text_count
                    else:
                        # 有匹配
                        matched_node = curr.children[first_token]

                        with matched_node.lock:
                            matched_node_text = matched_node.text
                            matched_node_text_count = len(matched_node_text)

                            curr_text = text[curr_idx:text_count]
                            shared_count = _shared_prefix_count(
                                matched_node_text, curr_text
                            )

                            if shared_count < matched_node_text_count:
                                # 分割匹配的节点
                                matched_text = matched_node_text[:shared_count]
                                contracted_text = matched_node_text[shared_count:]
                                matched_text_count = len(matched_text)

                                new_node = Node(text=matched_text, parent=curr)
                                # 复制租户访问时间
                                new_node.tenant_last_access_time = (
                                    matched_node.tenant_last_access_time.copy()
                                )

                                first_new_token = (
                                    contracted_text[0] if contracted_text else -1
                                )
                                if first_new_token != -1:
                                    new_node.children[first_new_token] = matched_node
                                    matched_node.parent = new_node
                                    matched_node.text = contracted_text

                                curr.children[first_token] = new_node
                                prev = new_node

                                # 更新租户信息
                                if tenant not in prev.tenant_last_access_time:
                                    self.tenant_token_count[tenant] += (
                                        matched_text_count
                                    )
                                prev.tenant_last_access_time[tenant] = timestamp

                                curr_idx += shared_count
                            else:
                                # 移动到下一个节点
                                prev = matched_node

                                # 更新租户信息
                                if tenant not in prev.tenant_last_access_time:
                                    self.tenant_token_count[tenant] += (
                                        matched_node_text_count
                                    )
                                prev.tenant_last_access_time[tenant] = timestamp

                                curr_idx += shared_count

    def prefix_match(self, text: List[int], timestamp: float) -> Tuple[List[int], str]:
        """
        前缀匹配

        Args:
            text: 要匹配的 token 列表
            timestamp: 匹配请求的时间戳

        Returns:
            Tuple[匹配的 token 列表，租户标识]
        """
        with self.lock:
            curr = self.root
            curr_idx = 0
            prev = self.root
            text_count = len(text)

            while curr_idx < text_count:
                first_token = text[curr_idx]
                curr_text = text[curr_idx:text_count]

                curr = prev

                with curr.lock:
                    if first_token in curr.children:
                        matched_node = curr.children[first_token]

                        with matched_node.lock:
                            matched_text_guard = matched_node.text
                            shared_count = _shared_prefix_count(
                                matched_text_guard, curr_text
                            )
                            matched_node_text_count = len(matched_text_guard)

                            if shared_count == matched_node_text_count:
                                curr_idx += shared_count
                                prev = matched_node
                            else:
                                curr_idx += shared_count
                                prev = matched_node
                                break
                    else:
                        break

            curr = prev

            # 选择第一个租户
            with curr.lock:
                tenant = next(iter(curr.tenant_last_access_time.keys()), "empty")

            # 更新路径上的时间戳
            if tenant != "empty":
                current_node = curr
                while current_node is not None:
                    with current_node.lock:
                        current_node.tenant_last_access_time[tenant] = timestamp
                    current_node = current_node.parent

            ret_text = text[:curr_idx]
            return (ret_text, tenant)

    def prefix_match_tenant(
        self, text: List[int], tenant: str, timestamp: float
    ) -> List[int]:
        """
        指定租户的前缀匹配

        Args:
            text: 要匹配的 token 列表
            tenant: 租户标识
            timestamp: 匹配请求的时间戳

        Returns:
            匹配的 token 列表
        """
        with self.lock:
            curr = self.root
            curr_idx = 0
            prev = self.root
            text_count = len(text)

            while curr_idx < text_count:
                first_token = text[curr_idx]
                curr_text = text[curr_idx:text_count]

                curr = prev

                with curr.lock:
                    if first_token in curr.children:
                        matched_node = curr.children[first_token]

                        with matched_node.lock:
                            # 只继续匹配如果节点属于指定租户
                            if tenant not in matched_node.tenant_last_access_time:
                                break

                            matched_text_guard = matched_node.text
                            shared_count = _shared_prefix_count(
                                matched_text_guard, curr_text
                            )
                            matched_node_text_count = len(matched_text_guard)

                            if shared_count == matched_node_text_count:
                                curr_idx += shared_count
                                prev = matched_node
                            else:
                                curr_idx += shared_count
                                prev = matched_node
                                break
                    else:
                        break

            curr = prev

            # 只更新找到匹配的租户时间戳
            with curr.lock:
                if tenant in curr.tenant_last_access_time:
                    current_node = curr
                    while current_node is not None:
                        with current_node.lock:
                            current_node.tenant_last_access_time[tenant] = timestamp
                        current_node = current_node.parent

            return text[:curr_idx]

    def _leaf_of(self, node: Node) -> List[str]:
        """
        返回节点的叶子租户列表

        Args:
            node: 要检查的节点

        Returns:
            叶子租户列表
        """
        with node.lock:
            candidates: Dict[str, bool] = {
                tenant: True for tenant in node.tenant_last_access_time.keys()
            }

            for child in node.children.values():
                with child.lock:
                    for tenant in child.tenant_last_access_time.keys():
                        candidates[tenant] = False

            return [tenant for tenant, is_leaf in candidates.items() if is_leaf]

    def evict_tenant_by_size(self, max_size: int) -> None:
        """
        按大小驱逐租户

        Args:
            max_size: 最大允许的 token 数
        """
        with self.lock:
            stack = [self.root]
            pq = []

            while stack:
                curr = stack.pop()

                with curr.lock:
                    for child in curr.children.values():
                        stack.append(child)

                    # 添加叶子到优先队列
                    for tenant in self._leaf_of(curr):
                        if tenant in curr.tenant_last_access_time:
                            timestamp = curr.tenant_last_access_time[tenant]
                            heapq.heappush(
                                pq,
                                EvictionEntry(
                                    timestamp=timestamp, tenant=tenant, node=curr
                                ),
                            )

            logger.debug("Before eviction - Used size per tenant:")
            for tenant, size in self.tenant_token_count.items():
                logger.debug(f"Tenant: {tenant}, Size: {size}")

            # 处理驱逐
            while pq:
                entry = heapq.heappop(pq)
                tenant = entry.tenant
                node = entry.node

                used_size = self.tenant_token_count.get(tenant, 0)
                if used_size <= max_size:
                    continue

                # 从节点移除租户
                with node.lock:
                    if tenant in node.tenant_last_access_time:
                        node_len = len(node.text)
                        self.tenant_token_count[tenant] = max(
                            0, self.tenant_token_count[tenant] - node_len
                        )
                        del node.tenant_last_access_time[tenant]

                    # 移除空节点
                    if not node.children and not node.tenant_last_access_time:
                        if node.parent:
                            with node.parent.lock:
                                # 确保 text 不为空才能用 text[0] 作为 key
                                if node.text and node.text[0] in node.parent.children:
                                    del node.parent.children[node.text[0]]

                    # 如果父节点成为叶子，添加到队列
                    if node.parent:
                        parent = node.parent
                        if tenant in self._leaf_of(parent):
                            with parent.lock:
                                if tenant in parent.tenant_last_access_time:
                                    timestamp = parent.tenant_last_access_time[tenant]
                                    heapq.heappush(
                                        pq,
                                        EvictionEntry(
                                            timestamp=timestamp,
                                            tenant=tenant,
                                            node=parent,
                                        ),
                                    )

            logger.debug("After eviction - Used size per tenant:")
            for tenant, size in self.tenant_token_count.items():
                logger.debug(f"Tenant: {tenant}, Size: {size}")

    def remove_tenant(self, tenant: str) -> None:
        """
        移除租户

        Args:
            tenant: 要移除的租户标识
        """
        with self.lock:
            # 找到租户的所有叶子
            stack = [self.root]
            queue = []

            while stack:
                curr = stack.pop()

                with curr.lock:
                    for child in curr.children.values():
                        stack.append(child)

                    if tenant in self._leaf_of(curr):
                        queue.append(curr)

            # 从叶子向上遍历移除租户
            while queue:
                curr = queue.pop(0)

                with curr.lock:
                    if tenant in curr.tenant_last_access_time:
                        del curr.tenant_last_access_time[tenant]

                    # 移除空节点
                    if not curr.children and not curr.tenant_last_access_time:
                        if curr.parent:
                            with curr.parent.lock:
                                # 确保 text 不为空才能用 text[0] 作为 key
                                if curr.text and curr.text[0] in curr.parent.children:
                                    del curr.parent.children[curr.text[0]]

                    # 如果父节点成为叶子，添加到队列
                    if curr.parent:
                        parent = curr.parent
                        if tenant in self._leaf_of(parent):
                            queue.append(parent)

            # 从计数字典移除租户
            if tenant in self.tenant_token_count:
                del self.tenant_token_count[tenant]

    def get_tenant_token_count(self) -> Dict[str, int]:
        """
        获取租户 Token 计数

        Returns:
            租户到 Token 数的映射字典
        """
        with self.lock:
            return dict(self.tenant_token_count)

    def get_smallest_tenant(self) -> str:
        """
        获取使用空间最小的租户

        Returns:
            租户标识，如果没有租户则返回"empty"
        """
        with self.lock:
            if not self.tenant_token_count:
                return "empty"

            min_tenant = None
            min_count = float("inf")

            for tenant, count in self.tenant_token_count.items():
                if count < min_count:
                    min_count = count
                    min_tenant = tenant

            return min_tenant if min_tenant else "empty"

    def get_used_size_per_tenant(self) -> Dict[str, int]:
        """
        获取每个租户的使用大小 (Token 数)

        Returns:
            租户到使用大小的映射字典
        """
        with self.lock:
            used_size_per_tenant: Dict[str, int] = defaultdict(int)
            stack = [self.root]

            while stack:
                curr = stack.pop()

                with curr.lock:
                    text_count = len(curr.text)

                    for tenant in curr.tenant_last_access_time.keys():
                        used_size_per_tenant[tenant] += text_count

                    for child in curr.children.values():
                        stack.append(child)

            return dict(used_size_per_tenant)

    def pretty_print(self) -> str:
        """
        美化打印树结构

        Returns:
            树的字符串表示
        """
        if not self.root.children:
            return ""

        result = ""
        children = list(self.root.children.values())
        child_count = len(children)

        for i, child in enumerate(children):
            is_last = i == child_count - 1
            result += self._node_to_string(child, "", is_last)

        return result

    def _node_to_string(self, node: Node, prefix: str, is_last: bool) -> str:
        """递归生成节点的字符串表示"""
        result = ""

        result += prefix
        result += "└── " if is_last else "├── "

        with node.lock:
            # 打印 List[int]
            result += f"{node.text} ["

            tenant_info = []
            for tenant_id, timestamp in node.tenant_last_access_time.items():
                tenant_info.append(f"{tenant_id} | {timestamp}")

            result += ", ".join(tenant_info)
            result += "]\n"

            children = list(node.children.values())
            child_count = len(children)

            for i, child in enumerate(children):
                is_last_child = i == child_count - 1
                new_prefix = prefix + ("    " if is_last else "│   ")
                result += self._node_to_string(child, new_prefix, is_last_child)

        return result
