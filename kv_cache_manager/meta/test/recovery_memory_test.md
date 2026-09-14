# 恢复后更新的内存回归

`recovery_memory_test` 使用合成数据验证恢复阶段的 arena 轮换对后续更新的影响，不需要 Redis 或 RDB。

## 运行

需要 Linux、启用 stats 的 jemalloc，以及非 ASAN 的正常构建。以 jemalloc 5.3.0 为验证基准。
该目标带 `manual` / `jemalloc` 标签，**不会随默认 `bazel test //...` 或现有 CI 自动执行**；显式运行：

```bash
bazel test //kv_cache_manager/meta/test:recovery_memory_test \
  --test_env=JEMALLOC_LIBRARY=/absolute/path/to/libjemalloc.so.2 \
  --nocache_test_results --test_output=all
```

可追加 `--runs_per_test=3` 检查重复性。测试进程必须能访问指定的库；不满足前提会失败，不会静默跳过。
Python runner 只给 C++ 子进程设置 `LD_PRELOAD` 和 `MALLOC_CONF`，不会给 Bazel/Python 本身预加载分配器。
轮换关闭、开启两组对照顺序执行，各自启动新进程，固定 4 个自动 arena、关闭 per-CPU arena 和后台线程，保留正常 tcache。

## 覆盖范围

- 恢复 262,144 个合成 key：实际 `AsyncRecoverTask → DeserializeFieldMap → PutIfAbsent → LRU`。
  测试替换持久层的 SCAN/Get 和写入响应，不保留第二份全量数据，不模拟网络开销。
- 两轮更新，每轮替换分散的 7/8 key，保留 1/8 旧对象；固定长度 payload 改变内容，保持 charge 不变。
  更新通过实际 `MetaStorageBackendManager::Upsert` 和 local backend，最终逐 key 验证内容。
- 4 个更新线程分别显式绑定 arena 0–3，覆盖全部恢复 arena。
  不测试部分覆盖场景或 HTTP/Admin 启动时的自动绑定策略。
- 在恢复完成、每轮更新完成后，刷新 jemalloc epoch，记录 key 数、charge、allocated、active、RSS
  和各 arena 的 allocated/active。断言 key 数、charge 稳定，allocated 变化小于 5%。
- 全覆盖场景先验证关闭轮换确实产生明显空洞，再要求开启轮换后的 `active - allocated` 至少减少 20%，
  并限制第二轮继续增长。恢复后各 arena 的 allocated 分布也必须符合开关行为。

## 测量边界

每轮更新的线程退出后再取样，使双方都排除未排空的 worker tcache 干扰；不显式调用 purge。
线程退出可能触发 allocator 自身回收，仍有存活对象的 slab 会继续计入 active。
为满足 manager 的调用方串行化约束，写入调用外加互斥锁。该测试衡量内存回归，**不衡量恢复或更新吞吐**。
RSS 只记录，不做阈值断言；主要判据为 allocator 的 active/allocated，减少内核回收时机带来的波动。

本地 x86_64、jemalloc 5.3.0 的 4 KiB 和 64 KiB allocator 页构建均已重复验证。
一次完整测试约 4–5 秒，单个子进程约数百 MiB；具体成本取决于构建与机器。
这些结果不能替代目标 ARM64 环境验证，也不等同于长期线上负载或完整服务压测。
