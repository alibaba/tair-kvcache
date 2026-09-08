# CacheReclaimer 按用量逐出的跨轮轮转设计

| 项目 | 内容 |
|---|---|
| 状态 | 已实现，单测与共用回收链路端到端回归通过 |
| 更新时间 | 2026-09-08 |
| 代码基线 | `origin/main`，`a6e5d176` |
| 涉及模块 | `manager`、`metrics` |
| 关联能力 | 按用量比例分配预算、异步 credit、水位提前停止 |

本文档只讨论如何避免有逐出预算的小 Instance 长期得不到执行机会。它不改变容量预算的分配算法，也不负责跨 Instance LRU；后者见 [Group LRU 设计](cache_reclaimer_group_lru.md)。整体行为见 [跨 Instance 预算分配设计](cache_reclaimer_instance_fairness.md)，删除安全约束见 [异步删除设计](cache_reclaimer_async_delete.md)。两部分已在本分支实现并统一验证。

## 1. 背景

改造前的 `USAGE_PROPORTIONAL` 策略每轮按实际用量分配采样和逐出预算，再按用量从大到小执行。前面的删除请求被 Executor 接受后立即建立 credit；如果水位已经恢复，剩余 Instance 不再采样或逐出。

提前停止是避免过度逐出的必要保护。问题在于：下一轮又从最大的 Instance 开始，不会优先处理上一轮没轮到的 Instance。当每轮超限缺口较小、前几个大 Instance 的请求已经足够覆盖缺口时，小 Instance 即使有正预算，也可能连续很多轮得不到执行机会。

例如每轮都有正预算的 A、B、C，且 A 的请求足以恢复水位：

```text
改造前：第 1 轮 A 后停止 -> 第 2 轮 A 后停止 -> 第 3 轮 A 后停止
改造后：第 1 轮 A 后停止 -> 第 2 轮从 B 开始 -> 后续接着未处理的 Instance
```

“停止写入”不等于 KVCM 已收到 Instance 下线指令。本改造只解决逐出调度机会，不承诺无容量压力时清空旧数据，也不替代业务显式发起的 Instance 清理。

## 2. 目标与非目标

### 2.1 目标

1. 保留现有权重维度、最大余数分配、采样预算、单 Instance 上限和 Location 过滤规则。
2. 水位恢复仍立即停止，但把尚未执行的 Instance 留在下一轮前面。
3. 单个 Instance 采样失败、候选为空或提交被拒绝，不能长期挡住后续 Instance。
4. 状态只保存在 Reclaimer 内存，不修改 Registry 数据，不增加逐出请求或 credit 的新生命周期。
5. `FIXED_PER_INSTANCE` 的固定预算和注册表遍历路径保持不变。

### 2.2 非目标

- 不保证原始整数预算为 0 的 Instance 获得逐出份额，不增加“至少删 1 个”的修正。
- 不保证实际删除 bytes 严格符合用量比例，也不保证各 Instance 容量相等。
- 不实现 per-instance credit、跨轮预算欠账或最低保有容量。
- 不按 Instance 是否有流量决定删除资格，不实现 Group LRU 或下线自动清空。

本方案解决的是“有名额但长期排不到”，不是所有原因造成的“不下降”。

## 3. 核心方案

### 3.1 预算重新计算，执行顺序跨轮保留

把两个决策分开：

| 决策 | 规则 |
|---|---|
| 当前 Instance 最多采样、逐出多少 | 每轮按最新正式 usage 重新生成现有公平预算 |
| 当前先执行哪个 Instance | 使用跨轮保留的轮转队列 |

每个 Group 维护一个只包含 `instance_id` 的队列。首次建立时，沿用现有计划的排序：权重降序、batch 降序、 `instance_id` 升序、原注册表顺序。之后保留队列顺序，不因每轮权重大小变化而重新排序。

每尝试处理完一个 Instance，就把它从队首移到队尾。水位恢复时直接停止，队首自然指向尚未处理的 Instance。 **水位恢复、随后的无压力轮次都不能清空这个队列**，否则下一次触发仍会从最大的 Instance 重新开始。

```text
初始队列：A -> B -> C
第 1 轮：尝试 A，credit 满足水位；队列变为 B -> C -> A
第 2 轮：从 B 开始；若 B 不足以恢复水位，再尝试 C
若第 2 轮在 C 后停止，下一轮队列为 A -> B -> C
```

轮转不要求“一轮只执行一个 Instance”。同一轮水位持续超限时，仍可依次处理多个 Instance。

### 3.2 为什么不只保存排序数组下标

当前计划会随着用量、有效 Instance 集合和预算取整发生变化。保存上轮的数组下标，不能确定下一轮指向的还是同一个 Instance。即使保存上次处理的 ID，再对每轮新排序的数组取后继，也可能因为排名反复变化而反复跳过同一项。

因此使用稳定的 ID 队列：预算可以变，仍然有效的等待项不因为权重排名变化而被插队。

### 3.3 最小状态

```cpp
// CacheReclaimer 内部状态，不对外暴露接口。
struct FairRotationState {
    std::deque<std::string> instance_ids;
};

std::map<std::string, FairRotationState> fair_rotation_by_group_;
```

队列不保存旧的 `InstanceInfo`、MetaIndexer、权重、预算、候选 key 或 Location。每轮通过当前计划按 ID 取回最新的 Instance 信息和预算，因此跨轮状态不能成为删除授权。

队列只由 Reclaimer cron 线程访问，不需要在采样 worker 或 Executor 中增加锁。

### 3.4 与当前计划同步

只有成功构造了非空公平计划，才同步该 Group 的轮转队列：

1. 删除当前计划中已不存在的 ID，例如 Instance 被删除、Indexer 不存在、当前权重为 0 或最终 batch 为 0。
2. 保留其余 ID 的相对顺序。
3. 将当前计划中新出现的 ID 按现有计划顺序追加到队尾，不插到仍在等待的项之前。
4. 若原来没有队列，则用当前计划顺序初始化。

本轮未超水位、配置预算为 0、计划构造失败或查询 Registry 失败，不应被解释为“等待项都已处理”，不推进队列。某个 ID 暂时失去正预算后重新进入计划，作为新有效项追加到队尾；这不构成对零预算 Instance 的保底逐出。

### 3.5 执行骨架

```text
读取本轮策略、Instance 列表、水位和参数快照
  -> 未触发：返回，保留轮转位置
  -> 按当前用量构造公平计划
  -> 同步 Group 的 ID 队列
  -> 最多尝试本轮计划项数次：
       检查运行状态、最新 credit 水位及触发范围
       若不再允许继续：停止，不移动尚未尝试的队首
       读取队首 ID 在本轮计划中的采样 / batch 预算
       调用现有 Instance 内 LRU 逐出路径
       将已尝试的队首移动到队尾
       请求 accepted 后，继续按最新 credit 检查是否停止
```

每轮的尝试上限固定为本轮同步后的计划项数，避免队列循环导致同一 Instance 在同一轮重复执行。当前 Group 执行完成后，仍沿用既有“Reclaim 准入先于 Migration 准备”的顺序。

## 4. 失败、暂停与范围变化

| 情况 | 队列处理 | 删除 / 调度处理 |
|---|---|---|
| 请求 accepted | 本 Instance 移到队尾 | 保留现有 pending、credit、Future 记账；水位恢复即停止 |
| 采样失败、超时、没有可删除 Location、请求被拒绝 | 已进入该 Instance 的处理函数，视为一次尝试并移到队尾 | 不新增 credit；水位仍超限时尝试后续项 |
| 处理函数内部遇到暂停 / 停止后返回 | 已开始的尝试允许移到队尾 | 不再次尝试该项；遵守已有取消和退出检查 |
| 调用处理函数前发现暂停、水位恢复或水位读取失败 | 未尝试的队首不移动 | 结束当前计划 |
| 当前超限维度或 Storage Type 集合变化 | 停止旧计划，不因范围变化整体重排等待项 | 下一轮重新计算权重和 Location 范围，再按第 3.4 节同步队列 |
| 全部尝试都没有 accepted 请求 | 不能把轮转本身算作进展 | `made_progress=false`，保留现有无进展退避 |

这里的“获得机会”表示处理函数获得执行机会，不代表一定能采样成功、提交成功或最终释放空间。在有效集合稳定、每轮至少能够开始一次尝试的前提下，队列中的每个 Instance 在最多 `N` 次计划项尝试内会轮到一次。它不是固定秒数内的删除承诺，也不覆盖预算长期为 0、持续故障或没有水位压力的情况。

## 5. 配置与生命周期

### 5.1 配置保持不变

本实现不新增开关，也不改变已有枚举编号：

- `instance_reclaim_budget_policy=USAGE_PROPORTIONAL`：按用量分配预算，执行顺序升级为跨轮轮转。
- `instance_reclaim_budget_policy=FIXED_PER_INSTANCE`：保持现有固定预算、注册表顺序路径。

因此选择容量比例模式的 Group 升级后执行顺序会变化，但不需要迁移 Registry 数据。切换到 `FIXED_PER_INSTANCE` 可以回退到旧固定预算路径，**并不等于恢复本改造前“用量比例预算 + 每轮从大到小”的完全相同行为**。

本分支同时提供 `instance_reclaim_budget_policy=GROUP_LRU` 的独立候选选择路径，不消费本容量策略的执行队列。该路径成为新的缺省模式，显式配置 `USAGE_PROPORTIONAL` 的 Group 继续使用容量比例轮转；默认值变更和旧配置升级规则见 [Group LRU 设计](cache_reclaimer_group_lru.md)。

### 5.2 状态清理

- 策略在每轮开始时读取。切换离开容量比例路径后，由 cron 清理该 Group 的轮转状态；切回时重新初始化。
- Group 删除：在成功获取完整 Group 列表后，清理列表中已不存在的 Group 队列。列表读取失败时不能据此清空状态。
- Instance 删除：由下一次成功计划同步移除。重新读取 Instance 信息和 Location，不能使用队列中的 ID 重放旧请求。
- 普通 Pause / Resume：保留队列；Stop 并 join 后可以安全清空。运行线程之外不得并发修改队列。
- 重启、重新创建 Reclaimer 或主备切换：允许重新初始化，不持久化、复制或恢复轮转状态，不承诺跨 Leader 的连续公平性。

以上状态清理只针对 ID 队列，不因此重置已有 pending、credit 或删除 Future；异步资源仍按原生命周期处理。

## 6. 成本与取舍

- 预算构造仍使用现有算法。同步队列需要建立本轮 ID 查找表并遍历 `N` 个计划项，预期为 `O(N)`；单次队列轮转为 `O(1)`。
- 常驻状态为每个活跃 Group 的有效 ID 队列，空间为 `O(sum(N * ID长度))`，不保存候选 metadata 或在途删除副本。
- 额外查找表只服务当前计划，可以使用已有 `FairReclaimPlanItem` 的索引，不复制大对象。
- 小 Instance 开始获得执行机会后，可能更早减少容量；这是本次改造的行为变化，不是最低容量保护。
- 不再承诺每轮由最大 Instance 先执行。用量比例继续决定请求预算，执行机会通过轮转分配，实际 bytes 仍受候选和 credit 影响。

## 7. 可观测性

保留现有 `fair_plan_count`、计划 / 采样 / 提交 Instance 数、计划截断及无进展退避指标。截断仍是正常的过度逐出保护，不能把“截断指标不再增长”作为本改造的验收标准。

新增两个低基数 counter，通过现有 metrics registry 和 KMonitor 链路上报，不增加按 `instance_id` 展开的指标：

| 指标 | 含义 |
|---|---|
| `fair_rotation_resume_count` | 一轮实际开始尝试，且起始项不是本轮原始用量排序第一项的次数；未开始尝试的轮次不计数 |
| `fair_rotation_advance_count` | 处理函数返回后实际推进队列的次数，含失败尝试，不等同成功提交数 |

DEBUG 日志记录 Group、本轮原始最大权重项、实际起始 ID、已尝试项数和停止后的下一 ID。跨轮日志用于验证同一小 Instance 是否获得机会；进程聚合指标只能佐证轮转确实发生，不能单独证明某个 Instance 已被删除。

## 8. 验证计划

已补充 12 个轮转专项单测，并扩展已有触发范围变化和配置切换用例，随 159 个 CacheReclaimer 全量单测通过。扩展回归与共用回收链路端到端测试已通过，结果见 [Group LRU 功能验证记录](cache_reclaimer_group_lru.md#102-本次功能验证结果2026-09-08)。验证覆盖以下范围，异步与迁移安全约束同时沿用已有回归用例：

1. **稳定复现并修复饥饿**：三个正预算 Instance，usage stub 在多轮保持不变，每轮第一个 accepted 请求足以恢复水位；验证执行顺序依次覆盖 A、B、C，而不是反复 A。
2. **无压力轮次不丢位置**：A 后停止，中间多轮 credit 使水位不再触发，恢复压力后从 B 开始。
3. **一轮执行多个项**：A 后仍超限，B 后恢复；下一轮从 C 开始，同一轮不重复 A。
4. **权重交换**：A、B 的排名反复交换，等待中的 C 不被重新排到队尾；所有预算仍按当轮权重重算。
5. **成员变化**：等待项删除、重新加入、新增更大 Instance、Indexer 缺失、正预算变 0 再恢复，均无旧对象引用或重复项。
6. **局部失败**：队首采样失败、空候选或提交拒绝，后续项仍被尝试；全部失败仍进入无进展退避。
7. **credit 安全**：accepted 后及时停止；轮转不伪造 credit，不等待物理删除完成，也不突破 pending 硬上限。
8. **触发范围变化**：bytes / keys / type 范围变化时停止旧计划，下一轮按新范围同步；仍有效的等待项保留相对顺序。
9. **策略和生命周期**：Group 之间队列隔离、固定策略不受影响、配置下一轮生效、Pause / Resume 保留位置、Stop / 重建安全。
10. **极小预算边界**：原始 batch 为 0 不进入队列，不因轮转强制删除；单 Instance 与现有预算执行基本一致。
11. **异步与迁移回归**：多轮 credit、pending 去重、同轮 Migration 快照互斥保持原契约。

后续端到端用例应模拟持续向大 Instance 写入、小 Instance 停止写入、Group 长期位于触发阈值附近，观察小 Instance 在持续压力下获得删除提交并最终用量下降。验收不要求其在没有容量压力时被清空。

## 9. 本次实现的范围与取舍

1. `USAGE_PROPORTIONAL` 默认升级执行顺序，不新增独立的轮转开关。
2. 预算仍按用量，执行机会按队列，不继续保留每轮最大 Instance 优先。
3. 本次仅覆盖有正预算的计划项；如果还要覆盖原始 batch 长期取整为 0，应作为额外的整数预算累积方案评审。

## 10. 实现位置

- `CacheReclaimer::PrepareFairExecutionOrder` / `TryReclaimOnGroupFair`：计划同步、按队列取项、尝试后推进；不修改整数分配实现。
- `CacheReclaimer::PruneFairRotationStates` / `ReclaimCron` / `Stop`：Group 状态回收和线程归属。
- `CacheReclaimer` 与 `kmonitor_metrics_reporter.cc`：轮转指标注册、采集和上报。
- `cache_reclaimer_test.cc`：跨轮顺序、预算更新、失败让位、成员同步和生命周期回归代码，已通过。
- 当前预算分配文档已同步执行顺序和提前停止行为；Group LRU 通过独立路径实现，两种策略可按配置切换。
