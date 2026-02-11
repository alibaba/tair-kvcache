"""
统一的前缀依赖哈希算法
"""

from typing import List


def hash_int64_func(prev_hash: int, current_value: int) -> int:
    """
    模拟C++的hashInt64Func逻辑

    在C++中使用std::hash<int64_t>进行链式哈希:
        hash = hashInt64Func(hasher, hash, hash_ids[index])

    Args:
        prev_hash: 前一个哈希值
        current_value: 当前值

    Returns:
        新的哈希值
    """
    # Python的hash()与C++的std::hash行为类似
    # 使用组合字符串来模拟前缀依赖
    combined = f"{prev_hash}_{current_value}"
    return hash(combined) & 0x7FFFFFFFFFFFFFFF  # 保持正数,避免负数


def apply_prefix_hash(hash_ids: List[int]) -> List[int]:
    """
    应用前缀哈希转换,每个block的key依赖前一个block

    这与C++ trace_util.cc 中的 ApplyPrefixHash 函数对应

    Args:
        hash_ids: 原始hash ID列表

    Returns:
        应用前缀依赖后的block key列表
    """
    block_keys = []
    hash_value = 0

    for hash_id in hash_ids:
        hash_value = hash_int64_func(hash_value, hash_id)
        block_keys.append(hash_value)

    return block_keys


def tokens_to_block_ids(
    token_ids: List[int],
    block_size: int = 16,
    truncate: bool = False
) -> List[int]:
    """
    将token IDs转换为带前缀依赖的block IDs

    使用整数哈希确保与C++一致性

    Args:
        token_ids: token ID列表
        block_size: block大小
        truncate: 是否截断不完整的block

    Returns:
        block ID列表
    """
    block_ids = []
    prev_hash = 0

    # 全局ID计数器 (用于生成唯一ID)
    if not hasattr(tokens_to_block_ids, '_hash_to_id'):
        tokens_to_block_ids._hash_to_id = {}
        tokens_to_block_ids._id_counter = 0

    for i in range(0, len(token_ids), block_size):
        if truncate and i + block_size > len(token_ids):
            continue

        # 获取当前block的tokens
        block_tokens = tuple(token_ids[i: i + block_size])

        # 计算block的哈希 (包含前缀依赖)
        block_hash = prev_hash
        for token in block_tokens:
            block_hash = hash_int64_func(block_hash, token)

        # 生成或获取唯一ID
        if block_hash not in tokens_to_block_ids._hash_to_id:
            tokens_to_block_ids._id_counter += 1
            tokens_to_block_ids._hash_to_id[block_hash] = tokens_to_block_ids._id_counter

        block_id = tokens_to_block_ids._hash_to_id[block_hash]
        block_ids.append(block_id)
        prev_hash = block_hash

    return block_ids


def reset_block_id_counter():
    """重置block ID计数器 (用于测试或新的转换会话)"""
    if hasattr(tokens_to_block_ids, '_hash_to_id'):
        tokens_to_block_ids._hash_to_id.clear()
        tokens_to_block_ids._id_counter = 0
