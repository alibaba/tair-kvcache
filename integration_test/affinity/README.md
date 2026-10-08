# Cache affinity 生命周期验收

当前算法、配置与限制见[整体设计](../../docs/design-cache-affinity-v1.md)。特别注意：保留副本的元数据 RMW 单测不代表节点回收全链路已受保护；当前 `ReclaimByNode` 未传入 `min_retained_replicas`。节点回收还依赖元数据 URI 的 `reclaim_indexer_type=node_lru` 配置。

测试分为三层；控制面返回 URI、释放复制 buffer、物理数据正确是不同的验收条件。

| 层次 | 入口 | 检查内容 |
| --- | --- | --- |
| 策略和组件回归 | `//kv_cache_manager/affinity/test:all`、`ReplicationExecutorTest`、`CacheManagerTest`、`MetaSearcherTest`、`CacheReclaimerTest`、`NodeLruReclaimIndexerTest` | 写入路由、复制门限/抑制、异步复制失败处理、元数据发布、淘汰决策及删除组件 |
| 真实服务的控制面集成 | `//integration_test/affinity:affinity_replication_test` | 25 个用例：本地写、远端读 hint、strict/non-strict 写、未完成/失败写与重试、部分 batch 发布、重复本地命中、逐 key 抑制、实例隔离、空 caller 兼容、noop、删除/重写、多 spec 发布、并发预算/失败回滚、协议滚动升级、显式目标、gRPC/HTTP 批量复制、重启恢复 |
| 双机真实数据链路 | `//integration_test/affinity_piggyback:mock_inference_node` + tair-mempool `.aoneci/kvcm_affinity_test.yaml` | A 写入、B 远端逐字节读、piggyback/async 两种复制、本地物理 Provider 校验、重复读、停 A 后读 B、跨实例 miss、删除和重写 |

`EvictedReplicaCanBeRecreatedAfterCapacityRecovers` 连接策略阶段：远端热读 → hint → 本地候选 → 高水位淘汰 → 滞后估算停止 → 容量恢复 → 再次 hint。它注入候选和节点指标，并不分配内存或执行物理淘汰。真实双机当前测试的是显式删除/重建和源 Provider 停机；控制面已覆盖 Manager 重启后的元数据与容量预算恢复；容量压满导致自动物理淘汰、PACE Provider 重启后的物理数据恢复、真实推理引擎/GPU 读写仍需要专门的集群用例，不能据此宣称已验收。

## Bazel 回归

在包含本次修改的源码版本上，从 `github-opensource` 执行。内/外源及不同 worktree 使用不同 `--output_base`；切换脚本必须从父仓库执行。

```bash
# 父仓库目录：切外源
bash internal_source/scripts/switch_to_open_source.sh
cd github-opensource
bazelisk --output_base=/tmp/bazel-affinity-e2e-os test \
  --config=debug --config=asan \
  --test_env ASAN_OPTIONS=detect_odr_violation=0 --test_output=errors \
  //kv_cache_manager/affinity/test:all \
  //kv_cache_manager/client/test:ReplicationExecutorTest \
  //kv_cache_manager/manager/test:CacheManagerTest \
  //kv_cache_manager/manager/test:MetaSearcherTest \
  //kv_cache_manager/manager/test:CacheReclaimerTest \
  //kv_cache_manager/meta/reclaim_indexer/test:NodeLruReclaimIndexerTest \
  //integration_test/affinity:affinity_replication_test
```

内源切换为 `switch_to_internal_source.sh`，使用 `/tmp/bazel-affinity-e2e-is`，并加：

```text
--copt=-DENABLE_TAIR_MEMPOOL --define ENABLE_TAIR_MEMPOOL=true
--define USER_CLIENT_LOGGER=true
//stub_source/kv_cache_manager/data_storage/test:TairMempoolBackendTest
//stub_source/kv_cache_manager/data_storage/test:PaceServiceResponseTest
//stub_source/kv_cache_manager/client/src/internal/sdk/test:TairMempoolCallerNodeProviderTest
```

双机 CI 已将这些内源回归接在构建前，使用 debug/ASAN；数据链路二进制与服务端 package 随后从同一份源码构建。`KVCM_COMMIT`、`KVCM_GITHUB_COMMIT` 必须指向包含相应变更的远端版本，tair-mempool SDK 固定为本次 CI 的 mempool commit。仅创建本地 worktree 不会自动改变远端 CI 使用的代码。

## 双机验收契约

每种模式使用独立 key 前缀，默认 3 个 block。每个阶段都要求进程退出码为 0，并由 `tair-mempool/tests/affinity/e2e_checks.py` 核验恰好 N 条 `BLOCK_OK` 和唯一、最后出现的 `E2E_OK`。

1. A 执行 `writer_abort`：未 Finish/失败 Finish 的 block 都不可读。
2. A 执行 `writer`：真实 Save、成功 Finish 后能逐字节读回，URI 的物理 node ID 必须属于 A。
3. B 执行 `reader_piggyback` 或 `reader_async`：先读远端原始数据，查询到设定门限才获得 hint；hint 的 key、源 URI、目标 PACE node ID 都必须准确。等 B 副本发布后，验证物理 node ID、稳定 URI、无重复 hint 和完整数据。
4. CI 停掉 A 的 Provider 容器并确认退出，B 执行 `reader_local` 再读全部 block。
5. 另一个 instance 对相同 key 执行 `reader_miss`；原 instance 执行 `remove`、`reader_miss`、`writer`、`reader_local`，验证删除和本地重建。

`--expected-node-id` 必须来自 MetaService `/v1/api/memnode` 中本机唯一、健康 Provider 的 **数值 PACE ID**，不能使用 Provider UUID 或从被测 URI 反推期望值。SDK caller 和 hint 的 `target_node_id` 都使用该 PACE ID 的字符串形式。

```bash
# 已部署的双机环境中，在 A/B 各自 Provider 容器内执行。
mock_inference_node --kvcm-endpoint KVCM_IP:6381 \
  --instance-group affinity_test_group --instance-id affinity_test_instance \
  --role writer --expected-node-id A_NUMERIC_ID \
  --block-key-prefix affinity_piggyback_test_ --num-blocks 3 --block-size 1048576

mock_inference_node --kvcm-endpoint KVCM_IP:6381 \
  --instance-group affinity_test_group --instance-id affinity_test_instance \
  --role reader_piggyback --expected-node-id B_NUMERIC_ID \
  --block-key-prefix affinity_piggyback_test_ --num-blocks 3 --block-size 1048576 \
  --query-rounds 5 --expected-hint-round 2 --wait-seconds 30
```

实例组配置需要同时包含 `write.ops.prefer_local` 和 `read.on_miss`；group override 会整体覆盖 process strategy。测试使用 CPU buffer 和单个 `spec_0`，不需要模型/GPU。禁止关闭校验、读取失败后造数据、将 release callback 当作复制成功。

## 正确性补齐回归

`AffinityPendingContractTest` 保留原目标名称，已移除 `manual`，三个历史缺口现为普通回归：
部分本地命中仍请求完整复制、热度按 instance 隔离、提示抑制按 instance 隔离。

本轮新增覆盖：

- `ReplicationExecutorTest`：多 spec 按名称对齐、真实缓冲区内容、所有目标 URI 发布、缺少源、
  读取/写入失败回滚、成功 Finish 响应失败不重复回滚、单缓冲区回退、调用节点变化、队列上限。
- `CacheAffinityManagerIntegrationTest`：节点指标 TTL、缓存快照不续期、乱序样本、按节点重置删除估算。
- `AffinityProbeTest`：零时间戳重新采样、满容量节点不能因除零提前停止淘汰。
- `CacheReclaimerTest`：instance 策略覆盖实例组，前一个实例 noop 不影响后续实例。
- `GrpcStubTest` / Python 控制面集成：完整 `source_specs` 协议透传、两个已有数据实例的热度/抑制隔离。
- `DataStorageManagerTest` / `EventReportBackendTest`：禁用后端、空 strict 目标和不支持亲和性的后端拒绝分配。
- 内源 `TairMempoolBackendTest`：缓存快照保持原采样时间，刷新失败不能假装容量信息是新数据。

```bash
bazelisk --output_base=/tmp/bazel-affinity-e2e-os test \
  --config=debug --config=asan --test_output=errors \
  --test_env ASAN_OPTIONS=detect_odr_violation=0 \
  //kv_cache_manager/affinity/test:all \
  //integration_test/affinity:AffinityPendingContractTest \
  //kv_cache_manager/client/test:ReplicationExecutorTest \
  //kv_cache_manager/client/src/internal/stub/test:GrpcStubTest \
  //kv_cache_manager/manager/test:CacheReclaimerTest \
  //integration_test/affinity:affinity_replication_test
```

多 spec 复制通过 `replication_capabilities` 协商，旧客户端保留正常读取和单 spec 提示。
`ReplicateWithBuffers` 支持按名称复用部分/全部 spec 的有 owner 缓冲区；旧单指针接口在多 spec 时回退到完整异步读取。

## 与最新主干合并后的回归

`kvcm_affinity_merge` 保留主干 Proto 字段编号：实例的 `default_query_type=8`、
写请求的 `min_replica_count=7` 不变；新增 `affinity_strategy_json` 在实例中为 9、
实例组中为 12，`StartWriteCacheRequest.caller=10`。旧亲和性分支的二进制需要
与服务端一起重新生成协议并构建，不能直接混用旧字段编号。

节点压力淘汰复用主干的异步删除准入，保留 pending 限额、重复过滤及迁移保护。
配置节点索引时使用带索引通知的通用元数据路径。相关回归目标：

```bash
bazelisk --output_base=/tmp/bazel-affinity-e2e-os test \
  --config=debug --config=asan --test_output=errors \
  --test_env ASAN_OPTIONS=detect_odr_violation=0 \
  //kv_cache_manager/manager/test:CacheReclaimerTest \
  //kv_cache_manager/meta/test:meta_indexer_node_lru_test
```

滞回仅计入后端确认成功的物理删除字节，并用采样时间校验，失败/超时/NOENT 不算新增释放。

新增回归覆盖：

- `MetaSearcherTest`：并发副本数量/实例容量准入，按 spec 最低保留数的并发删除校验。
- `SchedulePlanExecutorTest` / `CacheReclaimerTest`：部分成功、重复 URI、晚到反馈与新容量样本竞态。
- `CacheAffinityManagerIntegrationTest` / `FrequencySketchTest`：能力协商、热度衰减、复制成本、前缀位置和拓扑文件刷新/失效。
- `ReplicationExecutorTest`：进程内资源预算、实例轮转、每节点限速、结果统计及具名缓冲区所有权。

原生测试中的 GPU 缓冲区用例验证类型传递与所有权，实际 GPU DMA 仍需双机环境执行。

## 并发、升级和故障回归

真实服务的 25 个用例在 gRPC/HTTP 上执行；新增场景验证：

- 并发 WRITING 预留计入实例字节预算，失败清理后容量可重用；多 spec 按总大小计费。
- 批量请求因单 key 副本上限失败时，不残留其他 key 的半成品。
- 旧客户端和未知能力位正常读取，只有协商成功的客户端获得多 spec 提示。
- 显式复制目标校验和同目标去重；gRPC/HTTP 批量复制独立返回已存在、参数错误、后端不支持的结果，失败预留可再次使用。
- 重复 spec 名拒绝后可重试，元数据保留 WRITING/SERVING 的独立副本身份。
- 持久化元数据在 Manager 重启后恢复，已有副本继续占用预算，删除后恢复写入准入。

NFS 控制面用例中的复制返回 UNSUPPORTED，并不验证 PACE 物理复制成功。
SDK 回归另外验证服务端复制遵守节点限速、UNSUPPORTED 回退不重复计费，以及单个回退抛异常不丢失同批其他结果。
内部适配回归验证 caller、分配结果和容量指标统一使用 PACE 实例 ID；重启换 ID 立即可见，查询失败或 Provider 不唯一时返回空身份。
测试启动等待 RPC、HTTP 和 Admin 三个监听端口以及 Leader 发现就绪，代替固定休眠；多节点用例允许 follower 发现其他 Leader。
