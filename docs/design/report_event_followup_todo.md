# ReportEvent follow-up TODO

本文记录 PR #233 完成后暂缓处理的 Review 项。它们不属于当前 PR 的已实现能力，
也不应在发布说明或测试结论中描述为已覆盖。

后续修改仍应以
[`report_event.md`](../api/report_event.md) 和
[`report_event_snapshot_uri_version.md`](report_event_snapshot_uri_version.md)
中的接口与生命周期契约为准，并为并发问题优先补充可控阻塞点的确定性测试。
小 block / 大批量性能问题的已完成优化、禁止破坏的 HOST_DOWN 抢占窗口和后续目标化 Redis
读取方案记录在 [`report_event_performance.md`](report_event_performance.md)。

## P1：生命周期与重试语义

### 1. Snapshot 最终提交需要覆盖完整 reporter lifecycle

当前 lifecycle generation lease 覆盖 metadata replace，但 replace 完成后的 `Sync`
和 snapshot version commit 不在同一个 generation 栅栏内。旧 snapshot 可能在 metadata
写入完成后被阻塞，期间新 REGISTER 启动新的 reporter lifecycle，随后旧请求恢复并提交
旧 snapshot version。

建议：

- snapshot candidate 绑定创建时的 lifecycle generation；
- lifecycle 变化时取消旧 candidate；
- 提供原子的 `CommitSnapshotVersionIfGeneration(expected_generation)` 或等价操作，
  使 generation 校验、candidate 校验和最终 commit 不可分割；
- 明确 `Sync` 失败、generation 失效和 candidate 被替换时的可重试错误。

验收测试：

- 旧 snapshot 阻塞在 `Sync`；
- 同一 reporter 执行 REGISTER 并进入新 lifecycle；
- 恢复旧 snapshot；
- 断言旧请求不能 commit，新 lifecycle 的 snapshot version 和 metadata 不受影响。

### 2. Delta 部分失败的 `item_results` 需要满足安全重试契约

ADD 和 DELETE 分阶段写 metadata 时，同一 block/location 上的事件可能存在依赖。例如：

```text
event 0: ADD {A, B}
event 1: DELETE {A}
期望最终状态：只有 B
```

若 ADD B 失败、DELETE A 成功，接口可能返回 event 0 失败、event 1 成功。调用方按当前文档
只重试失败的 event 0 后，最终状态会变成 A、B 均存在。

建议优先保持现有“只重试失败项”契约，并将失败传播到同一 block/location 的事件依赖闭包。
如果选择要求调用方重试整个相关 delta batch，需要先评估兼容性并同步修改接口文档。

验收测试应覆盖同一请求内的 ADD→DELETE、DELETE→ADD、重复 ADD/DELETE，以及每个 metadata
阶段分别部分失败的组合。

### 3. EventReport storage 在线变更需要 backend incarnation 栅栏

在线 disable、update 或 remove EventReport storage 时存在以下待完善边界：

- backend lookup 需要检查 backend 是否仍然 available；
- 已持有旧 `shared_ptr` 的请求需要在状态锁内检查 backend 是否仍 open/available；
- HOST_DOWN/liveness cleanup 不能只携带可能从头计数的数值 generation；
- backend 重建后，旧请求和旧 cleanup 不能命中新 backend 中同值 generation 的 reporter。

建议为每次 backend 创建分配不可复用的 incarnation，并让请求 admission、generation lease
和异步 cleanup 同时校验 incarnation + reporter generation。

验收测试：

- 阻塞旧请求或旧 cleanup；
- disable/update/remove storage 并重建 backend；
- 注册 reporter 并写入新 metadata；
- 恢复旧操作；
- 断言旧操作失败或提前退出，且不能修改新 backend 的状态。

### 4. 重复 RegisterInstance 需要校验 instance group

同一 `instance_id` 的重复注册一致性校验还需要包含 `instance_group_name`。否则调用方使用
另一个 instance group 重复注册时可能先收到成功，随后 ReportEvent 仍按持久化的旧 group
查找 backend，并返回不直观的 `EventReportBackend not found`。

建议在重复注册时直接返回明确的冲突错误，并在错误信息中同时给出已有 group 和请求 group。

验收测试应覆盖相同 group 幂等注册、不同 group 冲突，以及冲突后原 instance 仍可正常查询
和上报。

## P2：接口、配置与运维一致性

### 5. 完成 snapshot delta drain timeout 的端到端配置

- 在 `kvcm_ops add_storage` / `update_storage` 暴露
  `snapshot_delta_drain_timeout_ms`；
- Admin Proto 输入负数时返回参数错误，不应静默忽略并回落到默认值；
- 保持配置文件、Admin API、运维 CLI 和运行时 update 使用同一校验范围；
- 为默认值、合法边界、零值和负值补齐端到端测试。

### 6. 明确并实现同一请求内 HEARTBEAT 的事件顺序语义

当前 HEARTBEAT 延迟到事件解析完成后处理。对于 unavailable reporter 的
`[HEARTBEAT, ADD]`，ADD 可能先捕获旧 lifecycle，随后 HEARTBEAT 更新 generation，
导致按输入顺序本应可执行的 ADD 失败。

需要明确 ReportEvent 是严格按序执行，还是仅对 mutation 定义顺序。如果接口保持按序语义，
应让 HEARTBEAT 的 lifecycle 变更在后续 mutation 捕获 lease 前生效，并补
`HEARTBEAT→ADD/DELETE/SNAPSHOT` 组合测试。

### 7. 多个 REGISTER 的逐 item 结果应与实际处理一致

同一请求内多个 REGISTER 当前会预聚合 mediums 并只执行一次。后续非法 REGISTER 可能使
前面的合法 REGISTER 一并失败，与逐 item 的结果契约不完全一致。

后续应选择并文档化一种语义：

- 按输入顺序逐项验证和执行；或
- 将 REGISTER 定义为请求级原子操作，并让所有相关 item 返回一致且明确的结果。

需要覆盖合法/非法 REGISTER 混合、重复 medium 和 REGISTER 与其他事件混合的测试。

### 8. 收敛 ReportEvent 成功路径的逐请求 INFO 日志

HTTP 层和 ServiceImpl 层仍可能为同一 ReportEvent 请求各输出一条 INFO 日志。高 QPS 下
会产生重复日志和额外 I/O。

建议只保留一处必要的成功日志，或改为 DEBUG/采样日志；错误日志仍需保留 trace id、
instance id、reporter、storage type、event type 和错误码，且不能打印完整大 snapshot。

## P3：低优先级维护项

### 9. 避免在热路径依赖 `std::random_device`

评估 snapshot version/token 生成对随机性的真实要求。若不要求密码学安全，可改为进程级
初始化的生成器或无锁/低锁的单调序列与随机前缀组合，避免某些平台上
`std::random_device` 阻塞或产生不稳定延迟。修改后需继续保证 token 格式、进程内唯一性和
并发安全。

### 10. 保持 PR/发布说明中的验证口径可审计

- 带 `manual` 标签的 snapshot/restart HTTP 集成测试不属于默认 GitHub CI；
- 没有实际 ASAN workflow 或本地执行记录时，不应写成“ASAN 已覆盖”；
- force-push 或新增 commit 后，应区分“此前 head 的全量结果”和“当前 head 的定向结果”；
- 发布前记录实际执行的 commit SHA、命令、target、模式（内源/外源、debug/release、
  ASAN/UBSAN）和结果。
