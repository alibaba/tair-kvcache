import torch

from kv_cache_manager.py_connector.kernel.batch_gather_scatter_helper import (
    batch_gather_kv_caches,
    batch_scatter_kv_caches,
)


def test_batch_gather_kv_caches():
    """
    测试 batch_gather_kv_caches 函数的功能正确性
    """
    torch.manual_seed(42)

    # 设置测试参数
    num_layers = 4
    total_token_in_kvcache = 10240
    num_tokens_per_block = 16
    hidden_size_per_token_per_layer = 1024
    total_blocks = 128
    total_buffer_blocks = 1024
    kv_count = 2  # K and V

    # 生成测试数据 - 创建KV caches
    # 根据函数实现，我们需要将每层的K和V分别作为单独的tensor
    kv_caches = []
    for layer_idx in range(num_layers):
        # 每个cache是[2, total_token_in_kvcache, hidden_size_per_token_per_layer]
        cache = torch.randn(
            kv_count,
            total_token_in_kvcache,
            hidden_size_per_token_per_layer,
            device="cuda",
            dtype=torch.bfloat16
        )
        # 将K和V分开添加到列表中，这样每个都是单独的指针
        kv_caches.append(cache[0])  # K
        kv_caches.append(cache[1])  # V

    # 创建KV缓存指针tensor - 现在应该有num_layers * 2个指针
    kv_caches_ptrs_tensor = torch.tensor(
        [cache.data_ptr() for cache in kv_caches],
        device="cuda",
        dtype=torch.int64
    )

    # 更新num_kvcache_ptrs为正确的值
    num_kvcache_ptrs = num_layers * kv_count  # 这应该是2 * num_layers

    # 生成目标tensor (用于存储gather结果)
    # 根据kernel实现，dst_tensor形状应为[total_buffer_blocks, num_layers*2, num_tokens_per_block, hidden_size_per_token_per_layer]
    dst_tensor = torch.empty(
        total_buffer_blocks,
        num_kvcache_ptrs,
        num_tokens_per_block,
        hidden_size_per_token_per_layer,
        device="cpu",
        dtype=torch.bfloat16,
        pin_memory=True
    )
    dst_tensor.zero_()

    # 生成block token indices
    # 每个block有num_tokens_per_block个token，总共total_blocks个block
    block_token_indices = []
    for block_idx in range(total_blocks):
        # 每个block的token indices需要在[0, total_token_in_kvcache)范围内
        block_indices = torch.randperm(total_token_in_kvcache)[:num_tokens_per_block].tolist()
        block_token_indices.extend(block_indices)

    # 生成dst block indices
    dst_block_indices = torch.randperm(total_buffer_blocks)[:total_blocks].tolist()

    # 调用 batch_gather_kv_caches 函数
    batch_gather_kv_caches(
        kv_caches_ptrs_tensor,
        dst_tensor,
        block_token_indices,
        dst_block_indices,
        num_tokens_per_block,
        hidden_size_per_token_per_layer,
    )

    kv_caches_cpu = []
    for kv_cache in kv_caches:
        kv_caches_cpu.append(kv_cache.to(device="cpu"))

    # 修正验证逻辑
    # block_token_indices是总列表，包含所有block的token indices
    # 每个block有num_tokens_per_block个token indices
    for block_idx in range(total_blocks):
        # 获取当前block的起始token index
        start_idx = block_idx * num_tokens_per_block
        dst_block_idx = dst_block_indices[block_idx]
        for token_pos in range(num_tokens_per_block):
            token_idx = block_token_indices[start_idx + token_pos]
            for layer_idx in range(num_layers):
                # 检查K值 - K存储在layer_idx*2的位置
                gathered_k = dst_tensor[dst_block_idx, layer_idx * 2, token_pos, :]  # 当前block，第layer_idx层，K部分
                # 注意：现在kv_caches是已分离的K和V，K在偶数索引(0, 2, 4...)
                expected_k = kv_caches_cpu[layer_idx * 2][token_idx, :]  # 对应的原始K值
                # expected_k = expected_k.to(device=gathered_k.device)
                torch.testing.assert_close(gathered_k, expected_k,
                                           msg="Mismatch in K values for block {}, layer {}, token {} {} {}".format(
                                               block_idx, layer_idx, token_idx, gathered_k, expected_k))

                # 检查V值 - V存储在layer_idx*2+1的位置
                gathered_v = dst_tensor[dst_block_idx, layer_idx * 2 + 1, token_pos, :]  # 当前block，第layer_idx层，V部分
                # 注意：现在kv_caches是已分离的K和V，V在奇数索引(1, 3, 5...)
                expected_v = kv_caches_cpu[layer_idx * 2 + 1][token_idx, :]  # 对应的原始V值
                # expected_v = expected_v.to(device=gathered_v.device)
                torch.testing.assert_close(gathered_v, expected_v,
                                           msg="Mismatch in V values for block {}, layer {}, token {}".format(
                                               block_idx, layer_idx, token_idx))

    print("Batch gather KV caches test passed!")


def test_batch_gather_kv_caches_with_reference():
    """
    使用参考实现验证 batch_gather_kv_caches 的正确性
    """
    torch.manual_seed(42)

    # 设置测试参数
    num_layers = 4
    total_token_in_kvcache = 10240
    num_tokens_per_block = 16
    hidden_size_per_token_per_layer = 1024
    total_blocks = 128
    total_buffer_blocks = 1024
    kv_count = 2  # K and V

    # 生成测试数据
    # 与第一个测试一致：将每层的K和V分别存储
    kv_caches = []
    original_caches = []  # 保存原始cache用于验证
    for layer_idx in range(num_layers):
        cache = torch.randn(
            kv_count,
            total_token_in_kvcache,
            hidden_size_per_token_per_layer,
            device="cuda",
            dtype=torch.bfloat16
        )
        original_caches.append(cache.clone())  # 保存原始值用于验证

        # 将K和V分开添加到列表中，这样每个都是单独的指针
        kv_caches.append(cache[0])  # K
        kv_caches.append(cache[1])  # V

    # 创建目标tensor
    # 与第一个测试一致，dst_tensor的形状是[total_buffer_blocks, num_layers*2, num_tokens_per_block, hidden_size_per_token_per_layer]
    dst_tensor = torch.zeros(
        total_buffer_blocks,
        num_layers * kv_count,  # num_layers * 2 (K和V)
        num_tokens_per_block,
        hidden_size_per_token_per_layer,
        device="cuda",
        dtype=torch.bfloat16
    )

    # 创建目标tensor用于参考实现
    reference_dst_tensor = torch.zeros(
        total_buffer_blocks,
        num_layers * kv_count,  # num_layers * 2 (K和V)
        num_tokens_per_block,
        hidden_size_per_token_per_layer,
        device="cuda",
        dtype=torch.bfloat16
    )

    # 为每个block生成token indices
    all_block_token_indices = []
    block_token_indices_per_block = []
    for block_idx in range(total_blocks):
        block_indices = torch.randperm(total_token_in_kvcache)[:num_tokens_per_block].tolist()
        all_block_token_indices.extend(block_indices)
        block_token_indices_per_block.append(block_indices)

    dst_block_indices = torch.randperm(total_buffer_blocks)[:total_blocks].tolist()

    # 调用批量gather函数
    kv_caches_ptrs_tensor = torch.tensor(
        [cache.data_ptr() for cache in kv_caches],
        device="cuda",
        dtype=torch.int64
    )

    batch_gather_kv_caches(
        kv_caches_ptrs_tensor,
        dst_tensor,
        all_block_token_indices,
        dst_block_indices,
        num_tokens_per_block,
        hidden_size_per_token_per_layer,
        num_layers * kv_count  # 修正参数
    )

    # 使用PyTorch实现参考实现
    for block_idx in range(total_blocks):
        block_token_indices = block_token_indices_per_block[block_idx]
        dst_block_idx = dst_block_indices[block_idx]
        for token_pos, token_idx in enumerate(block_token_indices):
            for layer_idx in range(num_layers):
                # 计算参考实现结果 - K值
                reference_k = original_caches[layer_idx][0, token_idx, :]
                reference_dst_tensor[dst_block_idx, layer_idx * 2, token_pos, :] = reference_k

                # 计算参考实现结果 - V值
                reference_v = original_caches[layer_idx][1, token_idx, :]
                reference_dst_tensor[dst_block_idx, layer_idx * 2 + 1, token_pos, :] = reference_v

    # 直接比较整个dst_tensor和参考实现的结果
    torch.testing.assert_close(dst_tensor, reference_dst_tensor,
                               msg="Mismatch between kernel and reference implementation for dst_tensor")

    print("Batch gather KV caches test with reference passed!")


def test_batch_gather_edge_cases():
    """
    测试边界情况
    """
    torch.manual_seed(42)

    # 测试单个token，单个block的情况
    num_layers = 1
    total_token_in_kvcache = 32
    num_tokens_per_block = 1
    hidden_size_per_token_per_layer = 64
    total_blocks = 1
    kv_count = 2  # K and V

    # 创建KV caches
    # 与前面测试一致，将每层的K和V分别存储
    kv_caches = []
    for layer_idx in range(num_layers):
        cache = torch.randn(
            kv_count,
            total_token_in_kvcache,
            hidden_size_per_token_per_layer,
            device="cuda",
            dtype=torch.bfloat16
        )
        # 将K和V分开添加到列表中，这样每个都是单独的指针
        kv_caches.append(cache[0])  # K
        kv_caches.append(cache[1])  # V

    # 创建目标tensor
    # 与前面测试一致，dst_tensor的形状是[total_blocks, num_layers*2, num_tokens_per_block, hidden_size_per_token_per_layer]
    dst_tensor = torch.zeros(
        total_blocks,
        num_layers * kv_count,  # num_layers * 2 (K和V)
        num_tokens_per_block,
        hidden_size_per_token_per_layer,
        device="cuda",
        dtype=torch.bfloat16
    )

    # 定义token indices和block indices
    block_token_indices = [5]  # 只有一个token，位置为5
    dst_block_indices = [0]  # 只有一个block，位置为0

    # 保存特定位置的原始值用于验证
    # cache[0]是K，cache[1]是V，我们已经将它们分离到不同的tensor中
    original_cache = torch.stack(
        [kv_caches[0], kv_caches[1]])  # 重新构成[2, total_token_in_kvcache, hidden_size_per_token_per_layer]
    expected_k = original_cache[0, 5, :].clone()  # 第0层，K部分，位置5
    expected_v = original_cache[1, 5, :].clone()  # 第0层，V部分，位置5

    # 调用批量gather函数
    kv_caches_ptrs_tensor = torch.tensor(
        [cache.data_ptr() for cache in kv_caches],
        device="cuda",
        dtype=torch.int64
    )

    batch_gather_kv_caches(
        kv_caches_ptrs_tensor,
        dst_tensor,
        block_token_indices,
        dst_block_indices,
        num_tokens_per_block,
        hidden_size_per_token_per_layer,
        num_layers * kv_count  # 修正参数
    )

    # 验证结果
    actual_k = dst_tensor[0, 0, 0, :]  # 第0个block，第0个指针(K)，第0个token位置
    actual_v = dst_tensor[0, 1, 0, :]  # 第0个block，第1个指针(V)，第0个token位置

    torch.testing.assert_close(expected_k, actual_k,
                               msg="Mismatch in K values for edge case")
    torch.testing.assert_close(expected_v, actual_v,
                               msg="Mismatch in V values for edge case")

    print("Gather edge case test passed!")


def test_batch_scatter_kv_caches():
    """
    测试 batch_scatter_kv_caches 函数的功能正确性
    """
    torch.manual_seed(42)

    # 设置测试参数
    num_layers = 4
    total_token_in_kvcache = 10240
    num_tokens_per_block = 16
    hidden_size_per_token_per_layer = 1024
    total_blocks = 128
    total_buffer_blocks = 1024
    kv_count = 2  # K and V

    # 创建KV caches用于接收数据
    kv_caches = []
    original_caches = []  # 保存原始cache用于验证
    for layer_idx in range(num_layers):
        cache = torch.zeros(
            kv_count,
            total_token_in_kvcache,
            hidden_size_per_token_per_layer,
            device="cuda",
            dtype=torch.bfloat16
        )
        original_caches.append(cache.clone())  # 保存零值作为原始状态
        # 将K和V分开添加到列表中，这样每个都是单独的指针
        kv_caches.append(cache[0])  # K
        kv_caches.append(cache[1])  # V

    # 创建KV缓存指针tensor
    kv_caches_ptrs_tensor = torch.tensor(
        [cache.data_ptr() for cache in kv_caches],
        device="cuda",
        dtype=torch.int64
    )

    # 创建源tensor，包含要scatter的数据
    # 形状应为[total_blocks, num_layers*2, num_tokens_per_block, hidden_size_per_token_per_layer]
    src_tensor = torch.randn(
        total_buffer_blocks,
        num_layers * kv_count,  # num_layers * 2 (K和V)
        num_tokens_per_block,
        hidden_size_per_token_per_layer,
        device="cpu",  # 注意：scatter函数期望host memory (pinned memory)
        dtype=torch.bfloat16,
        pin_memory=True  # 设置为pinned memory
    )

    # 生成block token indices
    # 每个block有num_tokens_per_block个token，总共total_blocks个block
    block_token_indices = torch.randperm(total_token_in_kvcache)[:num_tokens_per_block * total_blocks].tolist()

    # 生成src block indices
    src_block_indices = torch.randperm(total_buffer_blocks)[:total_blocks].tolist()

    # 调用 batch_scatter_kv_caches 函数
    batch_scatter_kv_caches(
        kv_caches_ptrs_tensor,
        src_tensor,
        block_token_indices,
        src_block_indices,
        num_tokens_per_block,
        hidden_size_per_token_per_layer,
        sm_count=3
    )

    kv_caches_cpu = []
    for kv_cache in kv_caches:
        kv_caches_cpu.append(kv_cache.to(device="cpu"))

    # 验证结果 - 检查KV缓存是否正确更新
    for block_idx in range(total_blocks):
        # 获取当前block的起始token index
        start_idx = block_idx * num_tokens_per_block
        src_block_idx = src_block_indices[block_idx]
        for token_pos in range(num_tokens_per_block):
            token_idx = block_token_indices[start_idx + token_pos]
            for layer_idx in range(num_layers):
                # 检查K值
                scattered_k = kv_caches_cpu[layer_idx * 2][token_idx, :]  # 层layer_idx的K部分
                expected_k = src_tensor[src_block_idx, layer_idx * 2, token_pos, :].cpu()  # 对应的源K值
                torch.testing.assert_close(scattered_k, expected_k,
                                           msg="Mismatch in K values for block {}, layer {}, token {}".format(
                                               block_idx, layer_idx, token_idx))

                # 检查V值
                scattered_v = kv_caches_cpu[layer_idx * 2 + 1][token_idx, :]  # 层layer_idx的V部分
                expected_v = src_tensor[src_block_idx, layer_idx * 2 + 1, token_pos, :].cpu()  # 对应的源V值
                torch.testing.assert_close(scattered_v, expected_v,
                                           msg="Mismatch in V values for block {}, layer {}, token {}".format(
                                               block_idx, layer_idx, token_idx))

    print("Batch scatter KV caches test passed!")


def test_batch_scatter_kv_caches_with_reference():
    """
    使用参考实现验证 batch_scatter_kv_caches 的正确性
    """
    torch.manual_seed(42)

    # 设置测试参数
    num_layers = 4
    total_token_in_kvcache = 10240
    num_tokens_per_block = 16
    hidden_size_per_token_per_layer = 1024
    total_blocks = 128
    total_buffer_blocks = 1024
    kv_count = 2  # K and V

    # 创建KV caches用于接收数据
    kv_caches = []
    original_caches = []  # 保存原始cache用于验证
    for layer_idx in range(num_layers):
        cache = torch.zeros(
            kv_count,
            total_token_in_kvcache,
            hidden_size_per_token_per_layer,
            device="cuda",
            dtype=torch.bfloat16
        )
        original_caches.append(cache.clone())  # 保存原始值用于验证
        # 将K和V分开添加到列表中，这样每个都是单独的指针
        kv_caches.append(cache[0])  # K
        kv_caches.append(cache[1])  # V

    # 创建源tensor
    src_tensor = torch.randn(
        total_buffer_blocks,
        num_layers * kv_count,  # num_layers * 2 (K和V)
        num_tokens_per_block,
        hidden_size_per_token_per_layer,
        device="cpu",  # scatter函数期望host memory
        dtype=torch.bfloat16,
        pin_memory=True  # 设置为pinned memory
    )

    # 为每个block生成token indices
    all_block_token_indices = torch.randperm(total_token_in_kvcache)[:num_tokens_per_block * total_blocks].tolist()

    src_block_indices = torch.randperm(total_buffer_blocks)[:total_blocks].tolist()

    # 创建KV caches的参考副本用于参考实现
    reference_kv_caches = []
    for layer_idx in range(num_layers):
        cache = torch.zeros(
            kv_count,
            total_token_in_kvcache,
            hidden_size_per_token_per_layer,
            device="cuda",
            dtype=torch.bfloat16
        )
        # 将K和V分开添加到列表中，这样每个都是单独的指针
        reference_kv_caches.append(cache[0])  # K
        reference_kv_caches.append(cache[1])  # V

    # 调用批量scatter函数
    kv_caches_ptrs_tensor = torch.tensor(
        [cache.data_ptr() for cache in kv_caches],
        device="cuda",
        dtype=torch.int64
    )

    batch_scatter_kv_caches(
        kv_caches_ptrs_tensor,
        src_tensor,
        all_block_token_indices,
        src_block_indices,
        num_tokens_per_block,
        hidden_size_per_token_per_layer,
        sm_count=3
    )

    # 使用PyTorch实现参考实现
    for block_idx in range(total_blocks):
        start_idx = block_idx * num_tokens_per_block
        src_block_idx = src_block_indices[block_idx]
        for token_pos in range(num_tokens_per_block):
            token_idx = all_block_token_indices[start_idx + token_pos]
            for layer_idx in range(num_layers):
                # 参考实现 - 更新K值
                src_k = src_tensor[src_block_idx, layer_idx * 2, token_pos, :].to(device=reference_kv_caches[layer_idx * 2].device, dtype=reference_kv_caches[layer_idx * 2].dtype)
                reference_kv_caches[layer_idx * 2][token_idx, :] = src_k

                # 参考实现 - 更新V值
                src_v = src_tensor[src_block_idx, layer_idx * 2 + 1, token_pos, :].to(device=reference_kv_caches[layer_idx * 2 + 1].device, dtype=reference_kv_caches[layer_idx * 2 + 1].dtype)
                reference_kv_caches[layer_idx * 2 + 1][token_idx, :] = src_v

    # 直接比较整个KV缓存和参考实现的结果
    for idx, (actual_cache, reference_cache) in enumerate(zip(kv_caches, reference_kv_caches)):
        torch.testing.assert_close(actual_cache, reference_cache,
                                   msg="Mismatch between kernel and reference implementation for kvcache index {}".format(idx))

    print("Batch scatter KV caches test with reference passed!")


def test_batch_scatter_edge_cases():
    """
    测试边界情况
    """
    torch.manual_seed(42)

    # 测试单个token，单个block的情况
    num_layers = 1
    total_token_in_kvcache = 32
    num_tokens_per_block = 1
    hidden_size_per_token_per_layer = 64
    total_blocks = 1
    kv_count = 2  # K and V

    # 创建KV caches
    kv_caches = []
    original_caches = []  # 保存原始cache用于验证
    for layer_idx in range(num_layers):
        cache = torch.zeros(
            kv_count,
            total_token_in_kvcache,
            hidden_size_per_token_per_layer,
            device="cuda",
            dtype=torch.bfloat16
        )
        original_caches.append(cache.clone())  # 保存原始值用于验证
        # 将K和V分开添加到列表中，这样每个都是单独的指针
        kv_caches.append(cache[0])  # K
        kv_caches.append(cache[1])  # V

    # 创建源tensor
    src_tensor = torch.randn(
        total_blocks,
        num_layers * kv_count,  # num_layers * 2 (K和V)
        num_tokens_per_block,
        hidden_size_per_token_per_layer,
        device="cpu",  # scatter函数期望host memory
        dtype=torch.bfloat16,
        pin_memory=True  # 设置为pinned memory
    )

    # 定义token indices和block indices
    block_token_indices = [5]  # 只有一个token，位置为5
    src_block_indices = [0]  # 只有一个block，位置为0

    # 保存源tensor的值用于验证
    expected_k = src_tensor[0, 0, 0, :].clone()  # 第0个block，第0个指针(K)，第0个token位置
    expected_v = src_tensor[0, 1, 0, :].clone()  # 第0个block，第1个指针(V)，第0个token位置

    # 调用批量scatter函数
    kv_caches_ptrs_tensor = torch.tensor(
        [cache.data_ptr() for cache in kv_caches],
        device="cuda",
        dtype=torch.int64
    )

    batch_scatter_kv_caches(
        kv_caches_ptrs_tensor,
        src_tensor,
        block_token_indices,
        src_block_indices,
        num_tokens_per_block,
        hidden_size_per_token_per_layer,
        sm_count=3
    )

    # 验证结果
    actual_k = kv_caches[0][5, :]  # 位置5的K值
    actual_v = kv_caches[1][5, :]  # 位置5的V值

    expected_k_converted = expected_k.cpu().to(device=actual_k.device, dtype=actual_k.dtype)
    expected_v_converted = expected_v.cpu().to(device=actual_v.device, dtype=actual_v.dtype)

    torch.testing.assert_close(expected_k_converted, actual_k,
                               msg="Mismatch in K values for edge case")
    torch.testing.assert_close(expected_v_converted, actual_v,
                               msg="Mismatch in V values for edge case")

    print("Scatter edge case test passed!")


if __name__ == "__main__":
    test_batch_gather_kv_caches()
    test_batch_gather_kv_caches_with_reference()
    test_batch_gather_edge_cases()
    test_batch_scatter_kv_caches()
    test_batch_scatter_kv_caches_with_reference()
    test_batch_scatter_edge_cases()
    print("All tests passed!")
