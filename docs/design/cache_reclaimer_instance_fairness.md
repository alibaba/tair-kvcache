# CacheReclaimer 跨 Instance 预算分配设计

| 项目 | 内容 |
|---|---|
| 状态 | 已实现，待合入 |
| 更新时间 | 2026-09-01 |
| 涉及模块 | `manager`、`meta`、`config`、`metrics`、`service`、`protocol`、`kvcm_ops` |
| 关联能力 | Instance Group 水位回收、异步删除、分层存储迁移 |

本文档描述 CacheReclaimer 当前的跨 Instance 预算分配行为。异步删除的 pending、credit、Future 和反压语义见
[CacheReclaimer 异步删除设计](cache_reclaimer_async_delete.md)，模块职责与分层迁移顺序见
[模块架构与关联关系](module_architecture.md)。

## 1. 背景与目标

CacheReclaimer 以 Instance Group 为水位和配额边界。Group 总 bytes、总 key count 或某个 Storage Type 的
bytes 超过阈值后，Reclaimer 在各 Instance 内采样 key、读取 LRU 属性，并提交异步删除请求。

如果每个 Instance 都使用相同的采样量和逐出上限，小 Instance 可能与大 Instance 承担相同的回收量；固定遍历
顺序还会决定异步 credit 满足水位前由谁先承担逐出。当前默认策略因此按本轮超水位维度上的实际用量，为各
Instance 分配采样和逐出预算，并让用量更大的 Instance 优先执行。

本设计遵守以下边界：

1. 只在同一个 Instance 内采样、排序和删除，不跨 Instance 复用或比较 key。
2. 调整的是预算分配，不改变 LRU 候选、Location 过滤、删除请求和异步生命周期。
3. 继续以 Group/Type 水位恢复为停止条件，不要求完整执行理论计划。
4. 不为低用量 Instance 提供最低保有容量，也不保证单轮实际删除量严格等于用量比例。

## 2. 配置与兼容模式

`CacheReclaimStrategy.instance_reclaim_budget_policy` 选择跨 Instance 的预算策略：

| 值 | Admin API | 行为 |
|---|---|---|
| `0` | `USAGE_PROPORTIONAL` | 默认。按当前超水位维度上的 per-instance 用量分配 Group 预算 |
| `1` | `FIXED_PER_INSTANCE` | 兼容模式。按注册表顺序遍历，每个 Instance 使用固定配置预算 |

内部持久化 JSON 使用整数，Admin protobuf/JSON 和 `kvcm_ops` 使用枚举名。旧配置缺少该字段时按
`USAGE_PROPORTIONAL` 处理；配置加载会拒绝未知枚举值。运行期若遇到未知值，Reclaimer 记录告警并保守回退到
默认的用量比例策略。

预算使用以下进程级参数：

- `key_sampling_size_total`：单个有效 Instance 的基准采样量 `S_cfg`；
- `del_batch_size`：单个有效 Instance 的基准逐出上限 `B_cfg`；
- `key_sampling_size_per_task`：单个采样任务的大小；
- sampling worker 数和 future timeout：约束并行采样波次及共享截止时间。

`FIXED_PER_INSTANCE` 完整复用原有执行路径，不做采样量归一化、Group 预算构造、用量排序或公平模式的单项
裁剪。切换策略只影响下一轮规划；已有 pending、credit 和删除 Future 不会被清空或重建。

## 3. 水位信号与权重维度

`GetWaterLevelExceed` 使用正式 usage 减去有效的 Group/Type credit，分别计算三类信号：

- `storage_type_exceeded[type]`：某个 Storage Type 的 bytes 超水位；
- `group_bytes_exceeded`：Group 总 bytes 超水位；
- `group_keys_exceeded`：Group 总 key count 超水位。

计划只使用本轮实际需要回收的维度，优先级为：

1. 任意 Storage Type 超水位：按每个 Instance 在所有超水位、且可由通用 Reclaimer 删除的 Storage Type 上的
   bytes 之和加权；
2. 否则 Group bytes 超水位：按每个 Instance 的可回收 storage usage bytes 加权；
3. 否则 Group key count 超水位：按每个 Instance 的 used key count 加权；
4. 都未超水位：不生成计划。

该优先级保证权重和本轮 Location 过滤范围一致。例如 NFS 超水位时，不会用 Instance 的总 bytes 为仅针对 NFS
的回收分配责任。Group bytes 和 key count 同时超限时优先使用 bytes；仅 key count 超限时固定使用 key count。

EventReport Location 由独立链路管理，不进入通用 Reclaimer 的 bytes 权重；`VCNS_HF3FS` 与 HF3FS 共用统计槽，
计算时跳过别名以避免重复计数。当前权重为 0、缺少 Instance 信息或缺少 MetaIndexer 的 Instance 不进入有效计划。

## 4. Group 预算与整数分配

设有效 Instance 数为 `N`。用量比例策略保留固定策略的理论总量：

```text
B_group = B_cfg * N
S_group = max(S_cfg, B_cfg) * N
```

`S_cfg < B_cfg` 时先把本轮采样基准归一化为 `B_cfg`，保证每个计划项的采样量不小于逐出量。任一配置为 0 时
不生成计划。Group 预算只检查整数乘法溢出，不受单次请求上限约束。

### 4.1 最大余数分配

batch 预算按权重占比分配。每个 Instance 先取得向下取整的整数份额，再把剩余名额依次给小数余数最大的
Instance；余数相同时按 `instance_id` 排序，结果可复现。权重求和及乘法使用无符号 128 位整数，避免大容量
Group 的 64 位溢出和浮点误差。

采样预算分两步分配：先让每个获得 batch 的 Instance 得到等量采样名额，再把 `S_group - B_group` 的额外采样
预算按相同权重分配给这些 Instance。由此保持 `sample_i >= batch_i`，且 batch 为 0 的 Instance 不产生无效采样。

分配不会强制“非零权重至少一个 batch”。极小 Instance 因整数取整得到 0 时，本轮不处理，避免其每轮固定失血。

### 4.2 单 Instance 上限

Group 预算分配完成后，每个计划项分别受单 Instance 配置值和 `kSizeLimit - 1` 限制。Group 总预算可以大于
`kSizeLimit`，因此 Instance 数较多不会导致整组计划失败。

采样预算被硬上限裁剪时，batch 同步按配置的 `S_cfg:B_cfg` 比例收缩，避免采样放大倍数退化。若一个已经分配
到非零 batch 的极小计划项因整数比例再次向下取整为 0，则保留 1 个 batch，避免裁剪造成永久无进展；这不会为
原始 batch 为 0 的 Instance 新增最低份额。裁掉的预算不在同轮转移给其他 Instance。

计划项最终按以下顺序执行：

1. 权重降序；
2. batch 降序；
3. `instance_id` 升序；
4. 原注册表顺序。

## 5. 采样与逐出执行

每个计划项仍在自己的 Instance 内执行：

```text
随机采样 sample_i 个 key
  -> 批量读取 LRU 属性
  -> 在该 Instance 内按 LRU 选择最多 batch_i 个 key
  -> 按当前超水位维度过滤可删除 Location
  -> 提交异步删除请求
```

这里的“按 LRU”用于从随机候选中选择较老的 victim，不是从全量 key 中直接取全局最老项。不同 Instance 的 LRU
时间不参与统一排序。

公平预算可能集中到大 Instance。为避免一次提交超过 sampling worker 数，采样按
`key_sampling_size_per_task` 拆分为有界波次：每个波次最多占用当前可用 worker，完成后再提交下一波。所有波次
共享同一 deadline；任一任务失败、超时、Reclaimer 暂停或 worker pool 已饱和时，本计划项整体失败，不提交部分
采样结果。

## 6. 异步 credit 与提前停止

公平计划每轮只构造一次。执行每个计划项前，Reclaimer 都重新读取正式 Group/Type usage，并减去最新 credit：

```text
effective_group_usage = official_group_usage - credited_group_usage
effective_type_usage  = official_type_usage  - credited_type_usage
```

当前请求被 Executor 接受后，pending Location、credited bytes、predicted deleted keys、DeleteHandler 和 Future 按既有
异步删除流程立即建立。Reclaimer 随后再次判断水位：

- 水位已经恢复：结束本 Group，不再为剩余 Instance 采样；
- 水位仍超限且触发维度未变化：继续下一个计划项；
- 请求为空或未被接受：不产生 credit，继续尝试后续计划项；
- 触发维度或超水位 Storage Type 集合变化：停止旧计划，由下一轮按新范围重新规划。

因此 `B_group`、`S_group` 以及各 Instance 预算都是上限，不是必须完成的工作量。用量较大的 Instance 先执行，其
accepted 请求可能已经覆盖全部水位缺口，使后续 Instance 不再承担本轮逐出。这一行为保留了异步 credit 对过度
逐出的保护。

当前 credit 仍按 Group/Type 维护，不拆分为 per-instance credit。计划权重读取当轮正式 per-instance usage；实际
删除因候选不足、Location 过滤、提交拒绝或异步失败产生的偏差，由后续轮次重新读取 usage 后修正。

## 7. 与分层存储迁移的协同

同一 cron round 内，Reclaim 准入先于 Migration 准备。删除请求 accepted 后会同步登记
`pending_locations_`，随后构造的 Migration Job 快照排除这些 Location，避免同一 Location 同时进入删除和 Copy。

这一顺序只保证准入互斥，不等待物理删除完成。若水位只达到 migration threshold、尚未达到 reclaim threshold，
Migration 仍可独立触发。公平预算不改变 Migration strategy、Copy 并发、reservation 生命周期或共享 Executor 的
任务优先级。

## 8. 边界与近似

当前实现存在以下有意边界：

1. bytes 权重最终分配的是 key 数预算。Instance 间平均 KV 大小差异较大时，实际删除 bytes 比例会偏离计划比例；
2. 不提供 per-instance 最低容量或保底份额，极小权重可以因取整在某轮获得 0 batch；
3. 不跨 Instance 合并候选，不实现严格全局 LRU；
4. 候选不足、过滤或提交失败后不在同轮重新分配预算；
5. Group/Type credit 防止整体过度逐出，但不精确描述每个 Instance 的在途删除量；
6. LFU 和 TTL 在公平路径中仍回退为当前的 LRU 实现。

这些边界不改变 Instance 隔离、Location 状态机及异步删除的安全约束。

## 9. 可观测性

公平路径提供以下指标：

| 指标 | 含义 |
|---|---|
| `fair_plan_count` | 成功生成的计划数 |
| `fair_planned_batch_count` / `fair_planned_sample_count` | 计划逐出和采样总量 |
| `fair_effective_instance_count` / `fair_planned_instance_count` | 有效权重 Instance 数和最终计划项数 |
| `fair_sampled_instance_count` / `fair_submitted_instance_count` | 实际进入采样和成功提交的 Instance 数 |
| `fair_zero_weight_skip_count` | 因当前权重为 0 而跳过的 Instance 数 |
| `fair_item_capped_count` | 预算被裁剪后仍保留的计划项数 |
| `fair_plan_truncated_count` / `fair_plan_truncated_instance_count` | 水位恢复或范围变化造成的计划截断次数和剩余项数 |
| `fair_sampling_size_normalized_count` | 因 `S_cfg < B_cfg` 执行采样量归一化的次数 |

DEBUG 日志记录权重维度、Group 理论预算、各计划项的原始/最终预算和提前停止原因。结合计划、采样和提交三组
Instance 数，可以区分预算取整、采样失败、提交拒绝和 credit 提前满足水位等情况。

## 10. 发布与回退

缺少策略字段的配置默认进入 `USAGE_PROPORTIONAL`。发布后重点观察计划量、实际采样/提交 Instance 数、计划截断
和无进展退避，确认预算分配及 credit 提前停止符合预期。

需要回退预算行为时，将策略改为 `FIXED_PER_INSTANCE`；下一轮开始按固定 per-instance 预算和原注册表顺序执行。
切换不会取消、重置或重复提交已经在途的删除请求，异步 credit 和 pending Location 继续按原生命周期收敛。

## 11. 自动化验证

当前自动化测试覆盖以下行为：

- 最大余数分配的确定性、128 位大数计算、Group 预算溢出和单 Instance 上限；
- bytes、key count、Storage Type 权重以及多维度同时超限的优先级；
- 采样与 batch 联动、`S_cfg < B_cfg` 归一化、极小份额取整和裁剪比例保护；
- 异步 credit 满足水位后的提前停止、继续执行、触发范围变化和提交失败；
- 有界采样波次的成功、失败、超时、暂停和 worker 饱和；
- 两种预算策略、缺省值、配置/Proto 转换和运行期切换；
- Reclaim pending Location 与同轮 Migration 快照的互斥。

`kvcm_ops` 单测覆盖枚举参数解析、JSON round-trip、缺省值和 create/update payload。通用 Reclaimer 冒烟测试用于
验证整体回收链路，但不作为跨 Instance 公平分配的专项覆盖依据。
