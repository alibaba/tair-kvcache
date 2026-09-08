# CacheReclaimer Instance Group 级 LRU 逐出设计

| 项目 | 内容 |
|---|---|
| 状态 | 已实现，单测与端到端功能回归通过；性能压测待开展 |
| 更新时间 | 2026-09-08 |
| 代码基线 | `origin/main`，`a6e5d176` |
| 涉及模块 | `manager`、`meta`、`common`、`config`、`metrics`、`service`、`protocol`、`kvcm_ops` |
| 关联能力 | Instance Group 水位回收、LRU、异步删除、分层存储迁移 |

本文档定义一种可按 Group 选择的新逐出策略：从各 Instance 采样，把候选放在一起按最近访问时间排序，再选择最冷的数据提交删除。采样采用“基础份额加按 key 数分配的额外份额”，删除不按容量分配份额；资源不足时轮转收集部分 Instance，不因 Group 太大永久停止回收。

当前容量比例策略见 [跨 Instance 预算分配设计](cache_reclaimer_instance_fairness.md)，该策略的独立改进见 [跨轮轮转设计](cache_reclaimer_cross_round_rotation.md)，与 Group LRU 一起在当前分支实现和验证。删除生命周期和模块关系分别见 [异步删除设计](cache_reclaimer_async_delete.md)、[模块架构与关联关系](module_architecture.md)。

## 1. 背景

### 1.1 两种不同的逐出目标

现有容量比例策略根据各 Instance 在超限维度上的实际用量分配采样和逐出预算，LRU 比较发生在 Instance 内。它解决的是“各 Instance 应承担多少回收量”，但不会跨 Instance 比较哪份数据更冷。

在部署切换或多个部署共享 Group 时，旧 Instance 的数据可能很久没有被访问，新 Instance 的数据仍很活跃。即使都获得逐出机会，按容量分配责任仍可能同时删除旧 Instance 的冷数据和新 Instance 的较热数据。

Group LRU 则希望：容量水位决定是否需要回收，最近访问时间决定优先删谁。活跃 Instance 可以占更多容量，不要求各 Instance 的容量趋于相等，也不预先约定每个 Instance 必须承担多少删除份额。

### 1.2 “TTL 对齐”的含义

这里的“TTL 对齐”指不同 Instance 剩余数据的 LRU 保留边界近似一致，不是新增固定过期时间。

- 比较的是 block 的最近访问时间，不是 Instance 创建时间，也不是数据首次写入时间。
- 旧 Instance 停止访问后，数据通常会变冷，在持续容量压力下优先被淘汰。
- 若旧 Instance 仍有最近访问的数据，不因为它已缩容就无条件先删除。
- 只比较采样且可删除的候选，不能承诺全量 keyspace 的严格全局 LRU。

### 1.3 与下线清理的边界

Group LRU 不负责判断部署是否永久下线。业务明确废弃某个 Instance 后，可以走独立的主动清理流程；不能把“无流量就主动清空”或“下线后限定时间内腾空”作为本策略的承诺。

## 2. 目标与非目标

### 2.1 V1 目标

1. 以 Instance Group 为配额、水位和逐出边界，不跨 Group 比较候选。
2. 资源足够时为每个当前可能有可回收数据的 Instance 分配非零采样预算；资源不足时跨轮轮转，避免小用量 Instance 因 batch 取整或固定顺序长期不进入候选集。
3. 先按当前超限维度过滤可删除 Location，再在 Group 内统一按 LRU 选择 victim。
4. 允许本轮 victim 全部来自一个 Instance，不受容量比例逐出份额约束。
5. 复用现有异步删除、pending、credit、反压、删除延迟和无进展退避。
6. 支持按 Group 灰度开启，与容量比例策略和固定 per-instance 兼容路径互斥选择。
7. 控制采样量、候选内存、任务并发、等待时间和单轮提交数，不扫描完整 keyspace。

### 2.2 V1 非目标

- 不修改 Instance 隔离：相同 block key 在不同 Instance 中仍是不同对象，不互相匹配或复用。
- 不实现严格 TTL、跨 Group 公平性、最低保有容量或 byte-exact 逐出。
- 不实现 Instance 下线感知与主动清空，不取消水位恢复后的提前停止。
- 不新增 per-instance credit，不重构 Executor 的删除状态机，也不改变 Migration policy。
- 不在采样失败时自动切回容量比例策略；降级结果必须保持“已取得候选中的近似 Group LRU”语义。

## 3. 配置与策略共存

### 3.1 用一个字段选择逐出模式

在已有 `InstanceReclaimBudgetPolicy` 中增加 `GROUP_LRU = 2`，通过 `instance_reclaim_budget_policy` 在三种逐出模式间切换。新版本默认使用 `GROUP_LRU`，不再新增 `POLICY_GROUP_LRU`。保留原字段名以兼容既有配置，其含义由“跨 Instance 的预算分配方式”扩展为“跨 Instance 的逐出模式”。

| `instance_reclaim_budget_policy` | 编号 | 执行行为 |
|---|---|---|
| `GROUP_LRU`（新增，默认） | `2` | 合并各 Instance 的采样候选，按 LRU 统一选择 victim，不预先分配各 Instance 的删除份额 |
| `USAGE_PROPORTIONAL` | `0` | 各 Instance 独立 LRU，按用量分配预算，使用跨轮轮转执行 |
| `FIXED_PER_INSTANCE` | `1` | 保留固定 per-instance 预算和注册表遍历顺序 |

`reclaim_policy` 的现有枚举及编号保持不变，正常配置仍使用 `POLICY_LRU`。`GROUP_LRU` 模式只接受 `POLICY_LRU` 或 `POLICY_UNSPECIFIED`（按 LRU 解释）；与 `POLICY_LFU`、`POLICY_TTL` 的组合明确拒绝。两个旧模式保留当前 LFU / TTL 告警并回退到 Instance 内 LRU 的行为，不顺带修改兼容路径。

例如以下是 Admin JSON 中的策略字段片段，不是完整的 update 请求：

```json
{
  "reclaim_policy": "POLICY_LRU",
  "instance_reclaim_budget_policy": "GROUP_LRU"
}
```

切回容量比例策略时，只需把 `instance_reclaim_budget_policy` 改为 `USAGE_PROPORTIONAL`；选择旧固定预算路径时改为 `FIXED_PER_INSTANCE`，不需要联动修改 `reclaim_policy`。不同 Group 可使用不同模式；同一 Group 的一轮只进入一条逐出路径，不能同时执行容量比例删除和 Group LRU 删除。

### 3.2 默认值与旧配置升级

默认值由 `USAGE_PROPORTIONAL` 调整为 `GROUP_LRU`。这里的默认值用于“没有配置该字段”的情况，不覆盖已有的显式配置，也不改变枚举值 `0`、`1` 的含义：

| 配置情况 | 新版本行为 |
|---|---|
| 新建 Group 未指定该字段 | 使用 `GROUP_LRU` |
| 旧 Registry 配置缺少该字段 | 恢复时使用 `GROUP_LRU`，逐出行为随默认值升级 |
| 已明确配置 `USAGE_PROPORTIONAL` / `0` | 保持容量比例模式，使用跨轮轮转 |
| 已明确配置 `FIXED_PER_INSTANCE` / `1` | 保持固定预算模式 |
| 已明确配置 `GROUP_LRU` / `2` | 使用 Group LRU 模式 |

因此，默认值变更不等于把所有旧 Group 强制切换到 Group LRU。若希望已有显式配置的 Group 也使用新策略，需要明确更新该字段；若希望历史缺字段的 Group 保持容量比例模式，应在升级前显式配置 `USAGE_PROPORTIONAL`。

实现统一修改 C++ 默认成员值、构造参数默认值、Registry JSON 缺字段处理、Admin 请求转换和 `kvcm_ops` 新建默认值。GET 返回及后续持久化显式携带解析后的策略；GET→修改→UPDATE 保留该值，不能因为更新无关字段意外切换模式。

改造前的 Admin protobuf 字段是普通 proto3 enum，无法区分“没有传值”和“显式选择编号 0”。本次保留字段编号 `8` 和原 enum 编码，将该字段放入单字段 `oneof`，增加字段存在性判断：未传时使用业务默认值 `GROUP_LRU`，明确传入 0 时仍是 `USAGE_PROPORTIONAL`。不改变原枚举编号，也不把所有读到的 0 当成缺省。

旧 protobuf 客户端可能在序列化时省略值为 0 的字段，服务端无法还原它原本是否想显式选择容量比例模式。需要显式选择该模式的调用方必须使用支持字段存在性的新客户端，不能依赖旧客户端的隐式零值；该差异需要兼容性测试和升级说明。

### 3.3 运行期切换、发布与回退

- 每轮读取一次策略和预算快照。运行中切换从下一轮规划生效，不重新解释正在执行的计划。
- 已 accepted 的请求继续按原 pending / credit / Future 生命周期完成，不因策略切换取消、重置或重复提交。
- 容量轮转队列与 Group LRU 的采样顺序状态独立。切换路径后由 cron 丢弃离开路径的调度提示，不清理删除状态。
- 内部 Registry JSON 仍按整数持久化；Admin protobuf/JSON 和 `kvcm_ops` 使用枚举名。配置校验拒绝未知模式和非法组合，不把非法值当成“未配置”；Group LRU 采样失败时也不自动切回容量比例模式。
- 旧二进制和旧工具不承诺识别 `GROUP_LRU = 2`。由于缺字段也会默认进入新模式，混合版本发布前应将待保留旧行为的 Group 显式设为旧模式，并限制旧客户端创建缺省策略的 Group；可能成为 Leader 的所有副本及相关管理工具升级后，再切换已有 Group 或使用新默认值。
- 升级前检查历史 `reclaim_policy`：若为 LFU / TTL 且缺少预算策略字段，需先明确选择旧模式，或改为 LRU 后启用 Group LRU，避免新默认值与非法组合校验冲突。
- 回退策略时显式设为 `USAGE_PROPORTIONAL` 或 `FIXED_PER_INSTANCE`。回退旧二进制前，应把所有 `GROUP_LRU` 以及仍依赖缺省值的 Group 显式设为旧版支持的值，并回读验证后再降级。

## 4. 整体执行流程

```text
读取 Group 配置、Instance 列表、正式 usage 和最新 credit
  -> 判断本轮超限范围
  -> 构造 Group 总预算及本轮覆盖的 Instance 的非零采样预算
  -> 有界采样并读取 LRU 属性
  -> 按本轮超限范围批量过滤 Location
  -> 合并可删除候选，按 (LRU 时间, instance_id, block key) 排序
  -> 选择最旧的 Top B_group
  -> 沿排序顺序，按连续的同 Instance 段拆分请求
  -> 提交前复核 Location 和反压，只为 accepted 请求建立 credit
  -> 水位恢复、范围变化或达到本轮上限后停止
  -> 再进入当前 Group 的 Migration 准备逻辑
```

不能直接对各 Instance 调用当前 `ReclaimByLRUWithBudget`：该函数会在 Instance 内选完 victim 后立即提交，没有机会与其他 Instance 的候选作比较。需要拆开“候选收集”和“删除提交”，而不是在旧路径外再套一个排序。

### 4.1 超限范围

复用现有 `GetWaterLevelExceed` 和 official usage / Group、Type credit 口径，不在新策略中另写水位判断。本轮范围沿用 Storage Type 优先、其次 Group bytes、最后 Group keys 的规则：

1. Storage Type 超限：只把匹配这些 Type 的可删除 Location 作为候选。
2. 否则 Group bytes 或 keys 超限：沿用通用 Reclaimer 的 Location 选择规则。
3. 没有超限：不采样、不逐出。

EventReport Location 不进入通用逐出，Type 别名按现有 BaseType 口径处理。仅 Group key count 超限时，也不能把删除部分 Location 等同于删除整个 key；key credit 仍只计入预计能完全删除的 key。

提交前和 accepted 后均复查水位。超限维度或 Type 集合变化时停止旧计划，下一轮重新采样规划，不把旧 NFS 候选直接拿来处理新的其他 Type 压力。

## 5. 预算与采样

### 5.1 有效 Instance

本轮有效 Instance 至少满足：注册信息和 MetaIndexer 存在，在当前超限维度上有正用量，且理论上可能提供普通 Reclaimer 候选。bytes 维度排除 EventReport 用量；keys 维度使用 key count，实际是否有普通可删除 Location 仍需过滤。

这里不依赖旧公平计划中的 `batch_i > 0`。旧算法即使给某个小 Instance 分配了 0 batch，新策略仍应给它采样机会。用量为 0 的 Instance 不放大 Group 预算；使用当前有效数 `N`，不是注册项总数。

### 5.2 Group 总预算

`S_cfg` 和 `B_cfg` 仍是现有配置中的单 Instance 基准，不是 Group 总预算。为保持与旧策略相同的理论总量，分别乘以有效 Instance 数 `N`，再对新模式增加独立的聚合采样上限：

```text
S = max(S_cfg, B_cfg)
S_theory = checked_multiply(S, N)
B_theory = checked_multiply(B_cfg, N)
S_plan = min(S_theory, group_lru_max_sampling_size)
```

`S_cfg`、`B_cfg` 任一为 0 时不生成计划，不能通过归一化意外开启逐出。乘法溢出时计划失败并记录原因。`group_lru_max_sampling_size` 是新增的独立 Group 总量限制，默认 65536；不是把现有单 Instance 的 `kSizeLimit` 当成 Group 上限。该限制只影响新模式，不能限制旧容量比例计划的 Group 总量。新增保护参数必须为正数；非法值在参数校验时拒绝，运行期若仍读到 0 则拒绝该计划并记录原因，不能当作无限制。

例如 `B_cfg=100`、三个有效 Instance 时，Group 理论上限为 300 个 block；需求示例中的“一次计划逐出 100 个 block”指已经算出的 `B_group=100`，不是说配置中的 `B_cfg=100`。Group LRU 不再限制每个 Instance 整轮最多删除 `B_cfg` 个，因此相同理论总量下，实际集中删除量和请求数也可能比旧策略大，仍受水位、pending 和单轮请求上限约束。

### 5.3 基础采样加按 key 数分配

大小 Instance 都采相同数量时，小 Instance 的采样覆盖比例会更高，大 Instance 中一些更冷的数据可能没有机会参加比较。V1 采用折中方式：先给每个 Instance 一份基础采样，再按 key 数分配额外采样。这里调整的是“看多少”，最终“删谁”仍只由候选的访问时间决定。

1. 按 Group 独立的采样轮转队列选择本轮覆盖项，数量为 `K = min(N, S_plan)`。`K < N` 时标记 partial plan，未覆盖项下轮优先，不返回整组失败。
2. 基础采样池为 `S_base = max(K, ceil(S_plan / 2))`，均匀分给本轮的 `K` 个 Instance，余数按 `instance_id` 补齐。正常预算下不是象征性地只采一个；预算非常紧张时允许每个本轮覆盖项仅分到一个。
3. 剩余 `S_plan - S_base` 按各 Instance 当前 used key count 使用 128 位最大余数法分配。权重不用 bytes，因为采样单位是 key；所有 key count 都为 0 而 bytes 表明仍有候选时，额外池也均分，不把统计暂时不一致变成永久无进展。
4. 每项最终受单 Instance `kSizeLimit - 1` 限制，裁掉的预算不重新分配。实际总采样预算记为 `S_effective`。

key count 只是可取得的采样规模估计，不代表这些 key 都含本轮可删除的 Location；资格仍在候选阶段检查。“一半基础、一半按 key 数”的固定 V1 折中缓解覆盖偏差，但不是严格的全局随机采样承诺，后端本身也可能使用冷候选采样。基础份额保留对小 Instance 冷数据的覆盖，后续可依据冷候选产出调整补采样。均不引入“每个 Instance 至少删除一个”的约束。

采样总量被裁剪时，同步缩小 Group batch，保留配置的采样放大倍数：

```text
B_group = min(B_theory, max(1, floor(uint128(S_effective) * B_cfg / S)))
```

该式仅用于 `N > 0`、原始配置和 `S_effective` 均非零的成功计划。`S >= B_cfg` 保证 `B_group <= S_effective`；宽整型乘法避免溢出；`max(1, ...)` 只防止已有非零 Group 预算被二次取整清零。正常未裁剪时，仍有 `B_group = B_cfg * N`。单个 Instance 的采样预算不是其逐出份额，最终只受它实际提供的候选数限制。

### 5.4 并发、截止时间与局部失败

复用现有采样 worker pool，不创建新线程池。Group 按 Instance 轮流派发采样子任务，优先让各 Instance 获得第一批采样机会。收集器不等待整批任务全部结束：哪个任务完成，就回收哪个任务的名额并继续派发；慢 Instance 不能挡住健康 Instance 的后续分片。

将本轮开始时可用 worker 数除以计划 Instance 数，向下取整且至少为 1，作为每个 Instance 的在途分片上限；所有任务仍受进程级 in-flight 上限约束。这样慢 Instance 不能反复占用健康 Instance 释放的 worker，单 Instance 场景仍可并行使用可用 worker。这个限制只分配采样并发，不分配删除份额；本轮不动态借用其他 Instance 的并发份额。

新旧有界路径共用采样任务提交能力，由 Group 收集器统一管理新模式的任务和结果，不能并发调用多个各自认为可以占满整个 pool 的采样循环。同一 Instance 的分片结果按实际完成状态合并，不能跨 Instance 混淆。

本轮候选收集共享一个绝对 deadline，时长沿用 `future_timeout_ms` 的当轮快照。派发和等待都使用剩余时间，不能为后续分片或 Instance 重新获得完整 timeout。截止时间限制的是等待与继续派发，不承诺取消不支持取消的底层 I/O；超时任务继续占用已有 in-flight 计数直到真实退出。

某个 Instance 完成采样后，由 cron 分批完成 Location 资格过滤，再将该 Instance 的候选加入 Group 列表；不必等所有 Instance 采样结束才开始过滤。每批过滤前后也检查收集 deadline，未完成资格过滤的 Instance 不进入成功结果。deadline 到达后可以对已经完整收集的候选排序并进入准入阶段，但不能继续补采样；最终准入读取仍受现有后端 I/O 约束、请求数量上限和运行状态检查保护。这不是对整轮墙钟耗时的硬性承诺。

失败处理采用显式的局部降级：

- 一个 Instance 的任一采样分片失败或超时，丢弃该 Instance 的全部部分结果；其他已完成 Instance 的候选可以保留。
- 属性或 Location 批量读取发生整体错误时，也跳过该 Instance，不能把调用失败伪装成所有 key 的时间都是 0。
- 任务饱和或 deadline 造成某些 Instance 未完成，记录 incomplete / skipped 数；仍允许使用成功收集的候选，但必须标记为不完整计划，不能声称覆盖了全部 Instance。
- 用独立的 Group 采样 ID 队列保留首次尚未获得派发机会的位置，下一轮先尝试该处；单个故障 Instance 也要让出顺序，避免总在相同前缀耗尽 deadline。队列是调度提示，不保留上轮候选，不能复用上轮未完成的半份结果。
- Group 全部失败时不提交删除，`made_progress=false`，沿用无进展退避。Reclaimer 停止或暂停时不使用已有候选继续提交。

不完整计划应以成功完成收集的 Instance 的采样预算之和，替换第 5.3 节的 `S_effective` 重新收缩 `B_group`，不能把失败 Instance 对应的理论批量也压到剩余成功候选上。这只是在异常时降低总量，不为各成功 Instance 再分配固定逐出份额；完整计划仍允许全部 victim 来自一个 Instance。

正常无故障且资源足够时，本轮为所有有效 Instance 收集候选；局部失败时只能保证成功候选范围内的 LRU。失败率持续较高、deadline 频繁截断时，应视为策略覆盖不足并告警，而不是把热点数据提前删除解释为理想 Group LRU。

### 5.5 Local 后端的分片覆盖

no-touch 采样只推进分片内部的采样游标，不移动物理 LRU 尾部。若每次只按尾部时间选择最冷的几个分片，一个不可回收的旧尾部可能让同一批分片反复入选，其他分片一直没有采样机会。

新路径每次选取 `K = min(sample_times, 非空分片数, 本次采样数)` 个分片：其中 `ceil(K / 2)` 个名额按分片 ID 跨调用轮转，剩余名额从尚未选中的分片里按尾部时间选最冷者。`K=1` 时该名额也轮转，不会一直固定在最冷分片。轮转部分先采样，分片内部仍沿独立游标继续读取。

轮转复用公共候选采样接口中每个 Local backend 独立的 64 位原子计数，支持并发采样；非空分片集合稳定时会持续覆盖各分片。三种逐出策略共用该采样基础，不修改业务访问时间或物理 LRU 顺序。冷分片优先仍保留，但采样不是全量扫描，最终只对实际取得且可删除的候选按访问时间排序。

## 6. 候选表示、过滤与排序

### 6.1 候选身份与访问时间

候选可以使用紧凑结构，不为每个 key 复制完整 Instance 信息：

```cpp
// 示意结构，instance_index 指向本轮持有的 Instance 快照。
struct GroupLruCandidate {
    size_t instance_index;
    int64_t block_key;
    int64_t lru_time_us;
};
```

去重身份为 `(instance_id, block_key)`，不能只按 block key 去重。一个 Instance 内重复采到同一 key 时，若读取到的有效时间不同，采用较新的访问时间，避免把后来已变热的重复项按旧时间选中。

排序统一使用 `PROPERTY_LRU_TIME` 的微秒时间戳升序，时间相同时按 `instance_id`、block key 排序，保证可复现。V1 已确认：成功读取但属性缺失 / 解析失败的单个 key 沿用历史 LRU 的时间 0 退化规则；新模式将非正时间也归一化为 0，并记录异常时间计数。这不是证明异常 key 一定最冷，而是避免它们因时间不可用长期无法回收的兼容性取舍；优先排序范围从原来的 Instance 内扩大到了整个 Group。批量读取错误必须走局部失败，不能当成所有 key 的时间都是 0。LRU 时间可能在采样后更新，V1 不增加阻塞前台访问的全局快照锁，因此只承诺采样时刻的近似次序。

候选采样复用公共 `SampleReclaimCandidates` 接口，一次返回 key 和访问时间；local 在分片锁内读取，不晋升 LRU 或修改业务时间。cached 在恢复期间从完整的持久层采样，再以 no-touch 精确读取的热缓存时间覆盖命中项；恢复完成后使用完整缓存。Group LRU 启用 `require_read_success=true`：批量读取失败时丢弃该 Instance，已消失的 key 跳过，成功读取但时间缺失 / 非法仍按 0 处理。容量比例 / 固定策略保持公共接口默认的 best-effort 读取降级语义。

Group LRU 的候选资格检查和最终准入另通过 `GetLocationMapsForMaintenance` 无副作用读取 Location；cached 恢复期间优先读热缓存，仅对缺 key 回查持久层且不回填。独立测试覆盖重复采样和 Location 读取后业务时间与物理 LRU 顺序不变，避免维护操作把冷数据读热。

local 采样使用 shard 内独立游标推进下一段候选，不移动 LRU 节点；否则多个采样子任务可能反复拿到同一批最旧 key。节点因业务访问、删除或淘汰移出链表时同步维护游标，游标只影响后续采样覆盖，不把已采到的 key 标记为变热。

### 6.2 Location 资格与删除准入分开

当前 `FilterLocID` 同时包含 Location 资格判断和本批 pending 配额裁剪，不能直接按 Instance 顺序对全部采样项调用它，再把已经按该顺序裁掉的结果当成完整 Group 候选。否则反压额度会先被前几个 Instance 的候选占掉，重新引入顺序偏差。

通过公共 `FilterLocIDImpl` 复用资格判断，分为两步：

1. **候选阶段**：按小批次读取 Location，判断 block 是否至少有一个符合本轮范围的 Location。排除 pending、EventReport、活跃 WRITING session、活跃 Migration Copy target，以及现有规则不允许删除的副本。此时不占 pending 配额，不建立 credit，也不为全部采样项预留待删除 bytes。Location map 用完即可释放，只保留紧凑候选。
2. **提交阶段**：对已经选中的 block，按当前 Location 状态重新执行完整过滤、反压裁剪和计数。只把最终请求的 Location、bytes、Type count 和预计完全删除的 key 数交给 `SubmitDelReq`。

多层存储 `keep_both` 的冷副本保护、热副本 spec 覆盖检查、Type 硬逐出等规则必须共用现有逻辑，不能在新路径复制出一套不一致的规则。普通容量策略和固定策略仍保留原来调用顺序，公共抽取不能顺带改变它们的选中集合。

在候选阶段保留 EventReport 以外的可删除性，不意味着可以忽略 EventReport 对 key 存活的影响：若 key 仍有任何不会随本次请求删除的有效 Location，不能计入“预计删除完整 key”的 credit。

### 6.3 Top B 的含义

过滤后对候选统一排序，最多选择 `B_group` 个不同的 `(instance_id, block_key)`。它是 block 数上限，不是 bytes，也不是 Location 数：一个 block 可以对应多个待删除 Location，部分 Location 删除后 key 也可能继续存在。

例如 A 提供 200 个很冷的候选，B 提供 200 个较热候选，`B_group=100`，允许 100 个 victim 全来自 A。反过来，如果 A 本轮只采到了 20 个冷候选，就不能假装知道 A 中其余未采到的数据也更冷；本轮可能继续选中 B。采样覆盖限制是近似算法的一部分，可通过后续自适应补采样优化，不属于本轮全量扫描承诺。

## 7. 保序拆批与异步 credit

### 7.1 不能按 Instance 一次性归并全部 victim

选中的全局顺序如果是：

```text
A-oldest -> B-older -> A-newer
```

不能因为 A 出现两次就提交 `A-oldest + A-newer`，再提交 B。若第一个 A 请求的 credit 已恢复水位，B 的更老数据就会留下。

V1 只合并全局有序列表中**连续属于同一个 Instance**的候选段，并按以下单次请求上限继续拆分：

```text
B_request = min(B_cfg, kSizeLimit - 1)
```

上例应依次尝试 `A-oldest`、`B-older`、`A-newer`。若最旧候选全部来自 A，则允许连续向 A 提交多批，直至 Group 水位恢复、本轮预算耗尽或反压生效；单次请求上限不是 A 整轮的逐出份额上限。

### 7.2 执行和停止

对每个连续段或分片：

1. 检查运行状态、最新 credit 水位和本轮触发范围；不再允许继续时停止。
2. 对这些已选 block 重新读取 / 过滤 Location，按当前剩余 pending 额度生成最终请求。
3. 请求为空或被拒绝：不建立 credit，继续尝试后续选中段；若是全局反压已满则直接结束本轮。
4. accepted：立即沿用 `SubmitDelReq` 登记 pending、Group/Type bytes credit、predicted deleted keys 和 Future。
5. 再检查水位；恢复即停止，不等物理删除完成，不要求执行完 Top B。

已选 block 在提交前消失、状态改变或被反压裁剪时，不在同一轮从 Top B 以外重新补足；下一轮重新采样。因此提交仍可能不是严格的 LRU 前缀：较老项可能已不可删除或提交失败。必须区分“按顺序尝试准入”和“按顺序完成物理删除”，异步任务的完成顺序不作保证。

### 7.3 请求数量上限

候选来自不同 Instance 且时间交错时，连续段拆分可能把一个大 batch 变成大量小请求。新模式增加进程级参数 `group_lru_max_delete_requests_per_round`，默认 128，限制一轮实际尝试提交的非空请求数；拒绝的非空请求也计入这个预算。它与现有 pending handler / bytes / Type 上限同时生效，不放宽任何既有上限。

达到该上限就结束当前有序前缀，剩余候选下轮重新采样，不改为按 Instance 全量归并。这个取舍以提交顺序正确和单轮工作有界为优先，可能牺牲吞吐；参数初值需用不同 Instance 数和候选交错程度的测试评估。

## 8. 与现有模块的衔接

| 位置 | 改动 / 保持的契约 |
|---|---|
| `TryReclaimOnGroup` | 根据本轮 `instance_reclaim_budget_policy` 互斥选择 Group LRU、容量比例或固定预算路径 |
| 候选采样与属性读取 | 复用采样后端和 worker pool，抽取 Group 有界调度、共享 deadline 和 no-touch 读取 |
| `FilterLocID` 相关逻辑 | 提取共同资格规则；候选资格与最终准入计数分离，旧路径行为保持 |
| `SubmitDelReq` | 继续接收单 Instance 请求，只按最终选中的 Location 建立状态；不使用全体采样项的统计 |
| `SchedulePlanExecutor` | 不新增跨 Instance 删除请求，不改状态迁移、delay、Future 或物理删除链路 |
| `TryMigrateOnGroup` | 仍在本轮 Reclaim 准入之后运行，通过已有 pending 快照排除被接受的删除 Location |
| `config` / proto / 转换 / `kvcm_ops` | 同步支持新策略枚举、校验、持久化和管理接口 round-trip |

如果 Group 尚未到 reclaim threshold、只达到 migration threshold，迁移仍可独立触发。采样失败或没有 accepted 删除不能通过提前返回外层函数而意外跳过原有迁移入口。Group LRU 不控制后台 GC 或 EventReport 的独立清理策略。

## 9. 成本与可观测性

### 9.1 成本边界

设实际采样预算为 `S_effective`、最终候选数为 `M <= S_effective`：

- 排序采用简单确定性的 `O(M log M)` 全排序；在总采样上限内优先保持实现易懂，后续再评估 top-k heap。
- 紧凑候选只保存 Instance 索引、key、时间。65536 个候选按典型 24～32 字节结构估算约 1.5～2 MiB，最终以实现的 `sizeof` 为准；该估算不含属性 map、临时 Location、Instance 字符串和未结束采样任务的内存。
- Location 读取按小批次处理，不能持有 `S_effective` 份完整 Location map；Group 采样总量上限也不能替代 metadata 返回大小及 in-flight worker 的既有资源约束。
- 相比容量策略，Group LRU 需要在删除前收集多个 Instance 的候选，并且先对采样项过滤 Location，前置 I/O 会明显增加；不能宣称只是增加一次排序。二次准入读取仅覆盖 Top B 中即将提交的部分。
- 基础采样加按 key 数分配会改变各 Instance 的 metadata 访问分布；在不完整计划、超大 Group 或很低的采样放大倍数下，LRU 精度会下降。

### 9.2 可观测性

新策略不把“未采样”混同“采样后没有选中”，独立记录以下聚合计数及阶段耗时，不新增高基数 Instance 标签：

| 指标组 | 用途 |
|---|---|
| `group_lru_plan_count` / `group_lru_partial_plan_count` | 成功计划与因局部失败、deadline 等造成的不完整计划 |
| 有效、已开始采样、采样成功的 Instance 累计数 | 区分预算覆盖、调度机会和实际结果 |
| 采样 key、可删除候选、选中 victim、accepted block 累计数 | 区分采样、Location 资格、LRU 排名和删除准入 |
| 采样失败 / LRU 属性缺失或非法计数 | 识别近似排序质量退化；批量 I/O 失败不能伪装成属性缺失 |
| 提前停止原因计数 | 区分水位恢复、范围变化、request 上限、deadline 和全局反压 |
| 候选收集 / 排序 / 准入耗时 | 定位前置 I/O 和小请求交错带来的成本 |

复用现有删除 bytes、Location、Future、credit 和反压指标。Group LRU 的 DEBUG 汇总包含 Group、有效与预算覆盖的 Instance 数、是否 partial、计划与成功收集的采样预算、Top B 数量和请求尝试数；单请求沿用现有提交日志。暂停和停止由运行状态检查保护，未增加独立的暂停计数。上述信息不能被表述为全量 keyspace 的冷热分布。

## 10. 验证范围与结果

### 10.1 验证范围

实现与跨轮轮转一并执行以下验证，实际完成情况另行记录，不能把验收计划当作已通过结果：

1. **跨 Instance 冷热选择**：大小不同的 Instance 使用可控 LRU 时间，最旧候选允许全部来自小 Instance；同容量热 Instance 不因固定份额被强制选中。
2. **低用量采样资格与采样权重**：旧容量算法 raw batch 为 0 的 Instance，在资源足够时仍有基础采样预算；额外预算按 key count 而非 bytes 分配，覆盖 bytes / key count 比例相反、极端倾斜、全零 key 统计和基础池取整。
3. **身份隔离**：不同 Instance 的相同 key 分别参与比较、过滤和删除；重复采样在 Instance 内去重并采用最新有效时间。
4. **排序确定性**：LRU 相同、属性缺失 / 非法、采样返回顺序变化，仍按既定规则输出；批量属性错误走失败路径。
5. **no-touch**：重复采样、属性和 Location 查询不刷新业务 LRU 或 backend 候选次序；覆盖 local、cached / persistent 组合。
6. **先过滤再 Top B**：最旧项全部是不可删 Location 时，后续可删除候选仍能进入 Top B；EventReport、活跃写入、Copy target、 keep_both 与 spec 覆盖规则不变。
7. **水位维度**：Group bytes、keys、单 / 多 Type、同时超限优先级、范围变化，均使用匹配的 Location 集合和停止逻辑。
8. **保序拆批**：`A-oldest -> B-older -> A-newer` 不能被归并成 A 全部先提交；每个 accepted 后恢复水位都应停止于对应位置。
9. **集中逐出**：同一 Instance 连续占据 Top B，能够按单请求上限提交多个请求，不被旧 per-instance 比例预算截断。
10. **异步安全**：候选计划不建立 credit；accepted 才记账；过期 / 失败 Future、pending 去重、反压及 key 保留规则保持。
11. **有界资源**：Group 乘法溢出、`N > S_plan` 时多轮覆盖全部有效 Instance 而非永久停回收、采样和 batch 为 0、联合裁剪取整、单请求 / 单轮请求上限、共享 deadline、超时 worker 不提前减 in-flight、局部失败不提交半个 Instance 的结果。
12. **覆盖退化**：一个 Instance 故障不导致其他健康 Instance 永久无进展，包含健康 Instance 需要多次补充采样分片的场景；分片旧尾部长期保留时，其余分片仍获得采样机会；多轮 deadline 截断时未开始项优先获得下一轮机会， incomplete 计数和成功覆盖对应的 batch 收缩准确，暂停 / Stop 后不发起新删除。
13. **配置兼容**：新建和旧 Registry 缺字段时默认 Group LRU，显式旧值 0 / 1 保持原模式；覆盖协议字段缺省与显式 0 的区别、旧客户端省略零值、新枚举 round-trip、非法 LFU / TTL 组合、CLI 更新无关字段不丢策略，以及下一轮切换和在途状态不变。
14. **单 Instance 回归**：合法完整候选下与原 LRU 选择基本一致。新路径过滤前移、确定性 tie-break 和显式 I/O 失败处理造成的差异单独断言，不承诺输出逐项完全相同。
15. **迁移回归**：本轮 accepted Location 出现在 Migration pending 排除快照中；仅达到迁移水位时仍能迁移。

端到端场景应包括“旧 Instance 无访问、新 Instance 持续读写”的稳定容量压力：验证冷旧数据优先回收、各 Instance 容量允许不同、新 Instance 不受固定份额限制。另测水位恢复后停止，明确不会在无压力时继续清空旧 Instance。

性能验证覆盖 Instance 数 1 / 3 / 100 / 512、LRU 高度集中与交错、采样总量裁剪、慢 metadata 后端，比较原容量策略的回收吞吐、首次提交延迟、采样 I/O、请求数和峰值内存。构建验证使用隔离测试环境；不需要的 Mooncake 依赖保持禁用。

非正 LRU 时间的归一化仅用于新模式，不能通过公共函数重构改变旧模式对历史异常值的处理而不提供单独回归说明。

## 11. 已确认取舍与后续优化

1. **V1 采样**：采用基础份额加按 key 数分配额外份额；资源不足时轮转收集部分 Instance，明确标记 partial plan。按冷候选产出自适应补采样留待后续优化。
2. **局部失败**：以健康且完整收集的候选继续缓解压力，并收缩 Group batch；不把一个 Instance 故障扩散成整个 Group 停止回收。
3. **资源保护**：Group 采样总量 65536、单轮请求 128 以及共享 timeout 为初值；后续按实际瓶颈调整，不以扩大采样量替代吞吐验证。
4. **属性退化**：保留单个 key 缺失 / 非法时间按 0 排序并统计异常；整体读取故障不进入这一退化分支。

本分支同时实现容量策略轮转和 Group LRU。两条路径共享删除基础设施，但调度状态独立，同一 Group 通过配置互斥选择。
