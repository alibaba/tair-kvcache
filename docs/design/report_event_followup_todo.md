# ReportEvent follow-up 状态与 TODO

本文最初记录 PR #233 暂缓处理的 Review 项。后续 hardened/performance 分支已经解决其中多数
问题；为避免后续 AI 重复修改或把已覆盖能力误报为缺口，本文件现在同时维护“仍未完成事项”
和“已解决证据”。以当前工作分支相对 main 的完整 diff 为准，不能只看最早两个 feature commit。

后续修改仍应以
[`report_event.md`](../api/report_event.md) 和
[`report_event_snapshot_uri_version.md`](report_event_snapshot_uri_version.md)
中的接口与生命周期契约为准，并为并发问题优先补充可控阻塞点的确定性测试。
小 block / 大批量性能问题的已完成优化、禁止破坏的 HOST_DOWN 抢占窗口和后续目标化 Redis
读取方案记录在 [`report_event_performance.md`](report_event_performance.md)。

## 仍未完成

### 1. 运维 CLI 的 snapshot delta drain timeout 兼容性

仓库内 Admin proto、proto/config 转换、序列化、默认值和正数/零/负数校验已经覆盖；若生产使用的
`kvcm_ops` 来自外部运维仓库，仍需在该仓库确认 add/update storage 暴露
`snapshot_delta_drain_timeout_ms`，且旧服务端不会因未知字段受影响。

### 2. 避免在热路径依赖 `std::random_device`

评估 snapshot version/token 生成对随机性的真实要求。若不要求密码学安全，可改为进程级
初始化的生成器或无锁/低锁的单调序列与随机前缀组合，避免某些平台上
`std::random_device` 阻塞或产生不稳定延迟。修改后需继续保证 token 格式、进程内唯一性和
并发安全。

### 3. 保持 PR/发布说明中的验证口径可审计

- 带 `manual` 标签的 snapshot/restart HTTP 集成测试不属于默认 GitHub CI；
- 没有实际 ASAN workflow 或本地执行记录时，不应写成“ASAN 已覆盖”；
- force-push 或新增 commit 后，应区分“此前 head 的全量结果”和“当前 head 的定向结果”；
- 发布前记录实际执行的 commit SHA、命令、target、模式（内源/外源、debug/release、
  ASAN/UBSAN）和结果。

## 当前分支已解决的历史项

- Snapshot candidate 绑定 reporter lifecycle generation，并通过
  `CommitSnapshotVersionIfGeneration` 原子校验 generation/candidate 后提交；测试覆盖 REGISTER
  抢占旧 snapshot、cleanup 与后续 attempt 的 fence。
- Delta 同一 block/location 的 ADD/DELETE 建立安全重试依赖闭包；任一阶段失败会传播到相关
  item，避免仅重试失败项反转 last-operation-wins 结果。
- 在线 disable/remove/rebuild 会检查 backend open/available；异步 cleanup 同时携带旧 backend
  `shared_ptr` 和 generation，旧 incarnation 不能清理新 backend。等待中的 delta/snapshot 在
  `Close()` 或 disable 时会被主动唤醒并二次检查 admission；disable 遗留的 candidate 会被 abort，
  re-enable 后写门可继续使用，不能等待完整的 snapshot drain timeout。
- 重复 `RegisterInstance` 校验 `instance_group_name`，不能把既有 instance 移到另一 group。
- unavailable reporter 的同请求 HEARTBEAT 恢复会把已准入 ADD/DELETE/SNAPSHOT 统一收敛到恢复后
  generation；正反事件顺序均有测试。
- 同一请求内多个 REGISTER 逐 item 校验，合法 item 的 mediums 合并并只执行一次实际注册，因此
  每个请求至多推进一次 lifecycle；非法 sibling 只影响自身结果。
- HTTP 与 ServiceImpl 的 ReportEvent 成功入口日志已降为 DEBUG；错误日志仍保留诊断字段。
- ReportEvent delta 热路径已压平为 block 哈希表和 location/spec 小向量，并直接生成最终 metadata task；
  `LocationSpec`/`CacheLocation` 的失效 move、重复 URI parse、BatchMerge task 深拷贝和 ordered-map spec
  merge 已修正。同机 Release/O2、纯 local 的 20k create/update 相对直接父提交下降约 21%/25%；具体
  A/B、语义约束与剩余并行边界见 `report_event_performance.md` 2.4 和 5.5。
- `snapshot_delta_drain_timeout_ms` 已进入 Admin proto/config 转换和运行时验证，非正值会被配置
  校验拒绝；仓库内 round-trip 和非法边界测试已覆盖。
- 查询有界线程池对部分线程创建失败和 `ParallelFor` 分配/入队异常执行显式清理；`noexcept` 热路径
  不会因资源异常直接 `std::terminate`，已入队 helper 也不能越过请求返回边界访问调用方引用。
- cached backend 恢复一旦由 SCAN cursor 得到非空批次，就保留该批精确 key，直到 Get 与
  PutIfAbsent 全部成功后才推进 cursor；失败重试不得重新 SCAN 同一 cursor，否则变化中的 Redis
  集合可能让原失败 key 在发布 Running 前被跳过。
- `MetaStorageBackendManager` 的 `Init()` 只在所有 backend factory 都成功后一次性发布对象，
  `noexcept` 内部的分配/工厂异常会转成 `EC_ERROR`；成功初始化后拒绝重复 Init。`Open()` 同样
  拒绝重复调用，cache open 或 recovery-thread 创建失败会回滚两侧 backend，避免半初始化状态和
  给 joinable `std::thread` 再赋值触发 `std::terminate`。
- Python Manager Client 的 route-refresh 线程启动失败会事务回滚 HTTP/session-discovery 资源；
  `close()` 超时后由 refresh worker 延迟关闭仍在使用的 discovery client。HTTP 非 200 和公共响应
  envelope 损坏统一归类为 `requests.RequestException` 子类（并兼容旧 `AssertionError`），包括
  `check_response=False` 路径，供 Vineyard 等上层正确推进熔断器；明确的 Manager 非 OK 业务状态
  保持原异常语义。

关键回归入口包括 `SnapshotCommitRejectsChangedLifecycleGeneration`、
`TestReportEventDeltaFailureMarksSafeRetryDependencyClosure`、
`TestHostCleanupCannotCrossEventBackendIncarnations`、
`CloseUnblocksSnapshotAndDeltaWaiters`、
`DisableWhileSnapshotDrainsAbortsCandidateAndReopensGate`、
`TestReportEventHeartbeatRecoveryCarriesSameRequestMutationsIntoNewLifecycle`、
`TestReportEventValidatesMultipleRegisterItemsIndependently` 和
`TestReportEventCoalescesMultipleValidRegistersIntoOneLifecycle`。
