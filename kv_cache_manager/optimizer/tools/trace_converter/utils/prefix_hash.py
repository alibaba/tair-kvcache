"""
统一的前缀依赖哈希算法
"""

from typing import List


def hash_int64_func(prev_hash: int, current_value: int) -> int:
    """
    精确实现C++的HashIntFunc (Jenkins Hash变种)
    
    C++ 原始实现 (hash_util.h):
        hash ^= hasher(value) + 0x9e3779b97f4a7c15 + (hash << 12) + (hash >> 32);
    
    Jenkins Hash 核心:
    - 异或混合 std::hash<int64_t>(value)
    - 加上黄金比例常数 0x9e3779b97f4a7c15
    - 位移混淆: (hash << 12) + (hash >> 32)
    
    Args:
        prev_hash: 前一个哈希值
        current_value: 当前值 (int64)
    
    Returns:
        新的哈希值 (int64)
    """
    # std::hash<int64_t>(value) 在多数平台直接返回值本身
    value_hash = current_value
    
    # Jenkins Hash: hash ^= hasher(value) + 0x9e3779b97f4a7c15 + (hash << 12) + (hash >> 32)
    # Python整数无限精度,C++的int64_t会自然截断,我们需要手动模拟
    GOLDEN_RATIO = 0x9e3779b97f4a7c15
    
    # 将输入转为无符号64位进行位运算 (模拟C++的二补数表示)
    hash_unsigned = prev_hash & 0xFFFFFFFFFFFFFFFF
    value_unsigned = value_hash & 0xFFFFFFFFFFFFFFFF
    
    # 计算各部分 (全部在无符号64位范围内)
    left_shift = (hash_unsigned << 12) & 0xFFFFFFFFFFFFFFFF
    right_shift = hash_unsigned >> 32
    rhs = (value_unsigned + GOLDEN_RATIO + left_shift + right_shift) & 0xFFFFFFFFFFFFFFFF
    
    # 异或操作
    result = hash_unsigned ^ rhs
    
    # 截断到64位 (模拟C++ int64_t的自然溢出)
    result &= 0xFFFFFFFFFFFFFFFF
    
    # 转换回Python的有符号表示 (模拟int64_t的二补数)
    if result >= 0x8000000000000000:
        result -= 0x10000000000000000
    
    return result


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
    
    直接使用哈希值作为block ID,确保完全无状态:
    - 相同输入保证相同输出 (幂等性)
    - 天然支持多进程并行 (无共享状态)
    - 与C++ ApplyPrefixHash行为一致
    
    Args:
        token_ids: token ID列表
        block_size: block大小
        truncate: 是否截断不完整的block
    
    Returns:
        block ID列表 (哈希值)
    """
    block_ids = []
    prev_hash = 0

    for i in range(0, len(token_ids), block_size):
        if truncate and i + block_size > len(token_ids):
            continue

        # 获取当前block的tokens
        block_tokens = token_ids[i: i + block_size]

        # 计算block的哈希 (包含前缀依赖)
        block_hash = prev_hash
        for token in block_tokens:
            block_hash = hash_int64_func(block_hash, token)

        # 直接使用哈希值作为block ID
        block_ids.append(block_hash)
        prev_hash = block_hash

    return block_ids
