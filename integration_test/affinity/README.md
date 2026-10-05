# Cache affinity 生命周期验收

测试分为三层；控制面返回 URI、释放复制 buffer、物理数据正确是不同的验收条件。

| 层次 | 入口 | 检查内容 |
| --- | --- | --- |
| 策略和组件回归 | `//kv_cache_manager/affinity/test:all`、`ReplicationExecutorTest`、`CacheManagerTest`、`MetaSearcherTest`、`CacheReclaimerTest`、`NodeLruReclaimIndexerTest` | 写入路由、复制门限/抑制、异步复制失败处理、元数据发布、淘汰决策及删除组件 |
| 真实服务的控制面集成 | `//integration_test/affinity:affinity_replication_test` | 13 个用例：本地写、远端读 hint、strict/non-strict 写、未完成/失败写与重试、部分 batch 发布、重复本地命中、逐 key 抑制、实例隔离、空 caller 兼容、noop、删除/重写、多 spec 发布 |
| 双机真实数据链路 | `//integration_test/affinity_piggyback:mock_inference_node` + tair-mempool `.aoneci/kvcm_affinity_test.yaml` | A 写入、B 远端逐字节读、piggyback/async 两种复制、本地物理 Provider 校验、重复读、停 A 后读 B、跨实例 miss、删除和重写 |

`EvictedReplicaCanBeRecreatedAfterCapacityRecovers` 连接策略阶段：远端热读 → hint → 本地候选 → 高水位淘汰 → 滞后估算停止 → 容量恢复 → 再次 hint。它注入候选和节点指标，并不分配内存或执行物理淘汰。真实双机当前测试的是显式删除/重建和源 Provider 停机；容量压满导致自动物理淘汰、进程重启恢复、真实推理引擎/GPU 读写仍需要专门的集群用例，不能据此宣称已验收。

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
```

双机 CI 已将这些内源回归接在构建前，使用 debug/ASAN；数据链路二进制与服务端 package 随后从同一份源码构建。`KVCM_COMMIT`、`KVCM_GITHUB_COMMIT` 必须指向包含相应变更的远端版本，tair-mempool SDK 固定为本次 CI 的 mempool commit。仅创建本地 worktree 不会自动改变远端 CI 使用的代码。

## 双机验收契约

每种模式使用独立 key 前缀，默认 3 个 block。每个阶段都要求进程退出码为 0，并由 `tair-mempool/tests/affinity/e2e_checks.py` 核验恰好 N 条 `BLOCK_OK` 和唯一、最后出现的 `E2E_OK`。

1. A 执行 `writer_abort`：未 Finish/失败 Finish 的 block 都不可读。
2. A 执行 `writer`：真实 Save、成功 Finish 后能逐字节读回，URI 的物理 node ID 必须属于 A。
3. B 执行 `reader_piggyback` 或 `reader_async`：先读远端原始数据，查询到设定门限才获得 hint；hint 的 key、源 URI、目标 UUID 都必须准确。等 B 副本发布后，验证物理 node ID、稳定 URI、无重复 hint 和完整数据。
4. CI 停掉 A 的 Provider 容器并确认退出，B 执行 `reader_local` 再读全部 block。
5. 另一个 instance 对相同 key 执行 `reader_miss`；原 instance 执行 `remove`、`reader_miss`、`writer`、`reader_local`，验证删除和本地重建。

`--expected-node-id` 必须来自 MetaService `/v1/api/memnode` 中本机唯一、健康 Provider 的 **数值 PACE ID**，不能使用 Provider UUID 或从被测 URI 反推期望值。hint 的 `target_node_id` 则使用 SDK caller 的 UUID。

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

## 已知未实现功能的验收

`AffinityPendingContractTest` 包含 3 个预期暴露当前缺口的用例：

- 一个组件已本地命中时，仍应为缺少的远端组件请求复制。
- 相同 caller/key 在不同 instance 的访问热度应独立。
- 一个 instance 的 hint 不应抑制另一个 instance 的复制。

这些断言使用正常失败语义，没有 `skip` 或 `expectedFailure`。target 带 `manual`，需显式运行，不计入默认 CI 的通过用例；实现相关能力后应去掉 manual 并纳入常规验收。

```bash
bazelisk --output_base=/tmp/bazel-affinity-e2e-os test \
  --config=debug --config=asan \
  --test_env ASAN_OPTIONS=detect_odr_violation=0 --test_output=errors \
  //integration_test/affinity:AffinityPendingContractTest
```
