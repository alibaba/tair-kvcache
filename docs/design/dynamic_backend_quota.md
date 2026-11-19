# 动态后端容量与 quota

Instance Group 可以限制整个 Group 的字节用量，也可以给 DRAM、SSD 等存储类型分别设额度。分类型额度可以填固定字节数，也可以按后端总容量的比例计算。后端扩缩容后，自动额度随最新一次有效的容量采样更新。

## 配置额度

### Group 总额度

`quota.capacity` 控制整个 Group 的字节额度。Admin API 省略这个字段时取 `0`。

| `quota.capacity` | 新写入 | 按 Group 字节水位回收 |
| --- | --- | --- |
| 正数 | 已用字节数达到上限时拒绝新分配 | 按配置的回收水位检查 |
| `0` | 停止整个 Group 的新分配 | 只要扣除预计释放的空间后还有用量，就触发回收 |
| `-1` | 关闭 Group 总字节上限 | 跳过 Group 总字节水位检查 |

总额度为 `-1` 时，分类型额度、key 数量限制和已有的缓存位置保护规则仍然生效。Group 和分类型的字节回收水位都由 `cache_config.reclaim_strategy.trigger_strategy.used_percentage` 设置。例如额度为 100 GiB、水位设为 `0.8` 时，计入回收判断的用量达到 80 GiB 就会触发回收。

### 分类型额度

在 `quota.quota_config` 中为每个需要限额的存储类型加一项。省略的类型按 Group 总额度和其他准入规则检查。

- `mode: "fixed"` 使用 `capacity` 指定的字节额度。这是省略 `mode` 时的默认值。填正数表示该类型的上限。填 `0` 会停止该类型的新分配，已有用量按该类型回收。
- `mode: "auto"` 将该类型的后端总容量乘以 `ratio`，结果向下取整到字节。`ratio` 默认是 `1.0`，可填大于 `0` 且不超过 `1` 的值。

每个基础存储类型只能配置一项。VCNS 3FS 和 3FS 共用同一类型额度。按类型计算回收时，CacheReclaimer 处理普通存储，EventReport 有独立的清理机制。

下面是 Admin API 请求中 `instance_group.quota` 的示例。这个 Group 的总字节不限额，DRAM 和 SSD 分别使用各自后端总容量的 50% 和 80%。示例适用于 DRAM 和 SSD 独立分配的主动分层模式。

```json
{
  "capacity": -1,
  "quota_config": [
    {"storage_type": "ST_TAIRMEMPOOL", "mode": "auto", "ratio": 0.5},
    {"storage_type": "ST_TAIRMEMPOOL_SSD", "mode": "auto", "ratio": 0.8}
  ]
}
```

### Auto 从哪些后端取容量

默认取 Group 的 `storage_candidates` 和迁移策略中配置的源、目标后端，再按存储类型筛选。因此，只用作迁移目标的后端也能给该类型提供容量。仅列在 `event_report_storage_candidates` 里的 EventReport 后端独立处理。

需要指定容量来源时，可以在该类型的配置里填 `storage_names`，用后端名称列表替换默认范围。这个列表只决定用哪些后端来算额度。用量仍然统计整个 Group 在该类型上的字节数。

同一类型的独立池容量相加，同一个后端名称只算一次。同一物理池有多个别名时，KVCM 按池标识去重。TairMempool 使用规范化后的域名或服务发现 URL 加存储介质作为标识。多个 URL 指向同一个池时，可以用 `storage_names` 只选其中一个。Meta 的多个副本提供的是同一个池的视图。

多个 Group 共用一个池时，可以给各 Group 配置不同的比例。比例决定各 Group 的额度，实际空间仍由后端分配。

创建或更新 Group 时，如果已注册的后端尚未实现容量查询，Admin 会拒绝该类型的 auto 配置，并在错误中给出后端名称。这类后端可以用固定额度。已实现容量查询的后端即使暂时连不上 Meta，也可以配置 auto。还未注册的容量来源按未知处理。

## TairMempool 的容量怎么算

| 后端模式 | KVCM 存储类型和分配介质 | Auto 的容量基数 |
| --- | --- | --- |
| 纯 DRAM（`off`） | `pace`，AUTO 或 DRAM（2） | 汇总 DRAM 的 `total_bytes` |
| 透明分层（`transparent`） | `pace`，AUTO | 分别汇总同一池的 DRAM 和 SSD，总量取较大值 |
| 主动分层（`explicit`） | `pace`，AUTO 或 DRAM（2） | 汇总 DRAM 的 `total_bytes`。此模式下 AUTO 分配在 DRAM |
| 主动分层（`explicit`） | `pace_ssd`，SSD（5） | 汇总 SSD 的 `ssd_total_bytes` |

透明分层下，一份缓存按 GA 的逻辑大小计入 `pace` 用量，数据在 DRAM 和 SSD 之间移动时仍计入同一类型。为这个逻辑池设置额度时，先按介质汇总所有 provider，再取 DRAM 和 SSD 两个总量中较大的值。比如两个 provider 分别有 100 GiB DRAM / 20 GiB SSD 和 20 GiB DRAM / 100 GiB SSD，池的容量基数就是 120 GiB。多个独立池各自算出基数后再相加，最后乘 Group 的比例。

为透明池配置 auto 时使用表中的 `pace` + AUTO 组合。显式指定 DRAM 或给同一个透明池配置独立的 `pace_ssd` 自动额度，会得到 `unsupported` 状态。实际分配还受双份驻留、SSD slot 大小、碎片和后端水位影响。可以用较小的 `ratio` 预留余量。

Meta 明确报告某节点关闭分层时，该节点只提供 DRAM 容量。需要计入 SSD 时，KVCM 会校验分层模式和 SSD 容量。此时收到 `0` 表示 SSD 尚未完成初始化，KVCM 保留上次容量。老 Meta 缺少分层字段时，纯 DRAM 仍可用 `total_bytes`，SSD 则无法计算。共享远端层需要按 owner 统计容量，当前这类 SSD 容量返回 `unsupported`。

## Meta 故障、重启和扩缩容时的行为

**短暂查询失败时沿用上次成功的容量。超过 30 秒仍未刷新成功，暂停该类型的新分配，并跳过该类型额度触发的回收。** 这样，Meta 断连或 KVCM 重启后还没取到容量时，已有缓存不会因为容量读失败而被当成超额。

| 当前状态 | 查询和监控看到的有效额度 | 新分配 | 按该类型额度回收 |
| --- | --- | --- | --- |
| 固定额度（`fixed`） | 配置中的字节数 | 按配置检查 | 按配置检查 |
| 本次刷新成功（`fresh`） | 最近成功的容量乘比例 | 按有效额度检查 | 按有效额度检查 |
| 刷新失败，上次成功距今不超过 30 秒（`refresh_error`） | 上次成功的容量乘当前比例 | 继续按该额度检查，后端仍须可写 | 继续按该额度和已有回收保护规则检查 |
| 超过 30 秒未刷新成功（`stale`） | 上次成功的容量乘当前比例，标记为不可用 | 暂停该类型新分配 | 跳过 |
| 尚未拿到完整容量，比如启动、切主或后端尚未注册 | `-1`，表示未知 | 暂停该类型新分配 | 跳过 |
| 当前介质无法计量（`unsupported`） | 有历史值时继续展示，否则为 `-1` | 立即暂停该类型新分配 | 跳过 |
| 确认容量已降为零 | `0`，容量已知 | 拒绝该类型新分配 | 扣除预计释放的空间后还有用量时触发回收 |

这里的暂停和跳过按存储类型生效。Group 总额度、key 数量水位及其他类型的额度继续检查。按水位发起的自动迁移需要源类型有正数可用额度，迁移目标也需要通过写入准入。

同一类型有多个容量来源时，KVCM 需要拿到完整容量才能计算额度。有池从未采样成功时，整个类型暂为未知。有池的历史采样过期时，整个类型额度暂不可用。对于同池别名，优先采用最近一次成功的容量。所选范围内有别名返回 `unsupported` 时，整个类型额度仍标记为不可用。

### 什么时候接受容量变化

- **确认扩缩容。** TairMempool 汇总健康、运行中且分层模式匹配的 provider。Consumer、正在下线的节点和已删除节点不计入容量。Meta 确认移除节点后，KVCM 按剩余节点重算额度。用量超过新额度的类型按新水位回收。Group 总额度为 `-1` 时，其他有余量的类型仍可使用。
- **节点暂时不可达。** Provider 心跳异常、连接失败或报告错误时，本次刷新失败，继续沿用上次容量。HTTP 失败、必要字段缺失或非法、重复 provider、字节数溢出等情况也走同样的保护。
- **确认空池。** Meta 可以返回空的 `memnodes`。如果响应附带汇总数据，容量、用量和已分配 GA 数都必须明确为零。Meta 省略节点列表时，这三个汇总值也是确认空池所必需的。收到不完整或互相矛盾的响应时，KVCM 保留上次容量和节点集合。

大幅缩容前的排空操作由后端或部署侧安排。

### 什么时候用节点列表判断缓存已丢失

节点健康状态和 GA 是否还存在是两件事。暂时断连的 provider 仍可能持有有效 GA，所以节点集合会保留这类 provider，直到 Meta 确认删除或在完整视图中移除它。节点列表刷新失败时，已有缓存位置保留为「可能存在」，等待下一次完整视图。

扩容期间，成功分配有时会先于探活轮询发现新节点。KVCM 在发起轮询前记录已知节点集合的版本号。如果请求期间发现了新节点，这次响应就过时了，节点和容量等下一轮再更新。分配到已知节点时，当前轮询仍可正常提交。

## 怎么查看额度和水位

Admin 的 Get/List 在 `instance_group` 里同时返回两项。写入配置看 `quota`，排查当前额度看同级的只读字段 `runtime_quota`。整对象发给 Update 时，服务端会忽略传回的 `runtime_quota`。[Admin API](../api/admin_service.md#update-instance-group)

- `total_bytes_limit_enabled` 对应 Group 总字节限制是否开启。配置 `quota.capacity=-1` 时为 `false`。
- `reference_capacity` 是所有已配置类型的有效额度之和，用来观察整个 Group 的容量量级。过期但仍有历史值的额度也计入。没有配置分类型额度、任一已配置类型容量未知或求和溢出时，返回 `-1`。
- `storage_types` 给出每个已配置类型的模式、比例、有效额度、是否已知（`known`）、当前是否能用于额度判断（`usable`）和刷新状态。Auto 类型还显示比例应用前的后端总容量及最近的成功时间。多池的成功时间取各池成功时间中最早的一项。

| 指标 | 怎么读 |
| --- | --- |
| `cache_manager_group_storage.usage_ratio` | 该 Group 在这个类型上的用量除以该类型的有效额度。结合 `capacity_known`、`capacity_usable` 和刷新时间判断这个值是否使用了历史容量 |
| `cache_manager_group.usage_ratio` | 固定正数总额度取总用量除以该额度。总额度为 `-1` 时，用量和参考容量须覆盖同一批普通存储类型，才显示总用量除以参考容量。EventReport 用量独立统计 |
| `data_storage.total_capacity_bytes` | 按 `type, unique_name` 上报单个后端的容量基数。未拿到容量时为 `-1`，刷新失败或过期时保留最近成功值。同池别名各有一条指标，查看物理总量时按池去重 |
| `data_storage.storage_usage_ratio` | 后端自身的水位。TairMempool 上报物理 DRAM 使用率。换算已用字节时，用这个比例乘同口径的 DRAM 总容量 |

分类型或不限额 Group 的使用率无法计算时显示 NaN。分母为零且仍有用量时显示 +Inf，用量和分母都为零时显示 0。固定总额度为 `0` 时，Group 使用率沿用原有值 `0`。参考容量只用于观测，实际写入和回收使用配置的总额度及各类型的有效额度。

存储总容量和后端水位由 Leader 上报，降级和卸载时沿用存储指标的清理流程。更多指标名称见 [Prometheus 指标](../prometheus-zh_CN.md)。

## 实现与调用关系

后端负责采集容量，通过 `GetStorageCapacity()` 提供最近成功的总字节数、成功时间和最近一次刷新结果。这个接口只读内存快照，首次采样成功前立即返回未知。后端通过 `SupportsStorageCapacity()` 声明是否实现该接口。未实现时返回 `EC_UNIMPLEMENTED`。

TairMempool 复用已有的探活循环，健康检查通过后查询一次 memnode。循环正常间隔约 10 秒，探活重试和请求耗时会拉长间隔。一次响应分别更新节点集合、自动额度所用的容量和物理 DRAM 水位。容量字段有问题时，可以继续使用响应中有效的节点信息。Quota 和水位指标都从这次采集留下的内存快照读取。监控还可能有约 5 秒的采样延迟。

后端用单调时钟记录成功时间，额度计算用它判断 30 秒过期，Admin 和指标同时提供便于排查的 Unix 秒时间。刷新失败时保留成功值和成功时间。移除、替换或 Leader 清理时会清掉后端的内存快照，重新打开或切主后等待新的成功采样。关闭后端时等待当前有超时限制的查询结束。

RegistryManager 在写入时校验配置。运行时由 `config/effective_quota` 的 `ResolveEffectiveQuota()` 使用调用方持有的 Group 配置计算额度。固定额度直接取配置，显式 auto 才通过 DataStorageManager 读相关后端的快照。一次计算始终使用同一版 Group 配置，避免比例和后端范围分别取自不同版本。

写入准入和迁移目标使用 `ForAdmission()`，回收和迁移源水位使用 `ForReclaim()`。前者遇到不可用的 auto 额度时停分配，后者跳过这个类型的容量判断。准入沿用按已统计用量检查水位的方式，多个并发请求可能在用量更新前同时获得分配。

## CLI、Bootstrap 和版本兼容

- **kvcmops。** 新建仍默认使用固定额度。要配置上面的主动分层示例，可以传入 `--quota_capacity -1 --quota_configs 'ST_TAIRMEMPOOL,auto,0.5;ST_TAIRMEMPOOL_SSD,auto,0.8'`。指定容量来源时用 `type,auto,ratio,backend1|backend2`。更新采用 auto 或总额度 `-1` 的 Group 时，请使用支持这两个配置的 kvcmops。
- **Bootstrap。** 新建默认总额度为 1,000,000,000 字节，主存储类型使用相同的固定额度。Reconcile 省略 `quota_capacity` 时保留已有额度。对已有 Group 显式设 `-1` 时只关闭总字节限制，分类型额度保留。分类型 auto 通过 Admin API 或 CLI 配置。
- **Optimizer。** 按真实布局订阅时，首次注册不限额 Group 使用理论最大命中率分析。已有 Group 切到不限额时继续使用已注册的模拟容量档位及其统计状态。按模拟布局订阅时使用订阅配置中的档位。
- **回滚。** 回到旧版二进制前，把已使用 auto 或 `-1` 的 Group 恢复为正数固定总额度和固定分类型额度。
