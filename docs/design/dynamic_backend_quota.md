# 动态分类型 quota 与无限 Group 总容量

## 配置

`quota.capacity > 0` 保留固定总字节上限，`0` 保留原有的禁止新分配及非空用量触发回收语义，`-1` 关闭总字节准入和总字节水位回收。key 数量、分类型水位和 Location 保护规则独立生效。参考容量不写回配置，不参与总量限制。

持久化 JSON 示例（Admin API 的类型名称分别为 `ST_TAIRMEMPOOL`、`ST_TAIRMEMPOOL_SSD`）：

```json
{
  "capacity": -1,
  "quota_config": [
    {"storage_type": "pace", "mode": "auto", "ratio": 0.5},
    {"storage_type": "pace_ssd", "mode": "auto", "ratio": 0.8}
  ]
}
```

`mode` 省略为 `fixed`，`ratio` 省略为 `1.0`，必须在 `(0, 1]` 内。自动有效 quota 为后端总容量乘比例，向下取整到字节。固定 `capacity=0` 不解释为自动。每个基础类型只允许一个配置；VCNS 3FS 与 3FS 使用同一基础类型。

自动容量默认关联本 Group 的 `storage_candidates`，以及 migration strategy 的 source、target，包含仅用于迁移的后端。独立的 EventReport 上报后端不自动加入普通存储的容量范围。可选 `storage_names` 替换该类型的默认关联范围。它仅选择容量来源，用量仍按整个 Group × 基础类型统计。

独立池容量相加，重复 storage name 只计一次。同物理池的别名通过后端 capacity identity 去重；TairMempool 使用规范化的 domain 或 service discovery URL 加介质。相同物理池应使用相同发现标识；不同 URL 指向同一池时无法自动推断，应通过 `storage_names` 明确选择一次。不会累加多个 meta 副本的容量。共享后端的份额通过比例表达，不要求独占后端。

新建 Group 省略整个 `quota` 时默认总量 `-1`；总量 `-1` 时，没有显式覆盖的关联类型默认自动 100%，因此只调整 DRAM 比例不会使 SSD 意外变成无类型限额。旧的显式固定总量和分类型配置保持不变。已有固定总量下，未配置的类型沿用不限制该类型的旧语义；自动默认模式下，未关联的显式写目标不能绕过类型 quota。

## 容量口径与失败保护

`DataStorageBackend::GetStorageCapacity()` 读取指定介质**总字节数**的内存快照，包含最近成功容量、成功时间和最近刷新结果。接口禁止网络/磁盘 I/O，不触发同步刷新，不等待首次采集或正在执行的刷新。后端负责异步采集及保留最近成功值；首次未知立即返回未知，未实现时立即返回 `EC_UNIMPLEMENTED`，不能以成功零值代替。`DataStorageManager` 只汇集各后端快照，不保存第二份容量缓存，也不创建采集线程。

TairMempool 从完整成功的 memnode 响应聚合健康且运行中的 provider，排除 consumer、下线/draining 节点及 tiering mode mismatch 节点。DRAM 使用 `total_bytes`；SSD 使用显式分层模式下的 `ssd_total_bytes`，明确禁用 tiering 的节点不提供 SSD 容量。启用 tiering 却报告 SSD 容量为零是初始化/上报未完成，按查询失败处理，不作为真实零容量。旧 meta 缺失 SSD/tiering 信息时，SSD 容量未知，不能使用 DRAM 容量替代。缺失字段、负值、数值溢出、重复 provider、HTTP 或解析失败都不提交部分和。成功的空 memnode 集合，或 API 对空集群省略集合并明确返回全零 aggregate，是确认的零容量。

Provider 的心跳或连接异常不能视为确认缩容：运行中但不健康、连接可疑/失败或错误状态会使整个容量样本失败，沿用最近成功值及过期策略，避免异常节点被过滤后发布部分容量甚至零容量。明确删除状态及成功完整快照中消失的节点仍正常减少容量。拓扑中的不健康 provider 仍可能持有有效 GA，不能仅凭健康状态判定数据不存在；memnode 刷新失败后，旧拓扑也不再用于否定新节点上的 location，直到下一次成功刷新。

分配可能比 memnode 轮询更早得知扩容节点。后台查询期间若成功分配发现了新节点，该查询的响应不再覆盖拓扑或容量，避免较早生成、较晚到达的快照抹掉新分配。此时保留最近容量并按刷新失败处理，等待下一次完整快照；普通已知节点上的分配不会使查询失效。

容量与实际分配模式对齐：

| 后端模式 | KVCM 类型与分配介质 | 自动容量基数 |
| --- | --- | --- |
| 纯 DRAM（`off`） | `pace`，AUTO 或 DRAM（2） | DRAM `total_bytes` |
| 透明分层（`transparent`） | `pace`，AUTO | `max(Σ total_bytes, Σ ssd_total_bytes)`，分别汇总同一后端池的两层容量后取较大值 |
| 主动分层（`explicit`） | `pace`，AUTO 或 DRAM（2） | DRAM `total_bytes`；主动模式的 AUTO 实际分配 DRAM |
| 主动分层（`explicit`） | `pace_ssd`，SSD（5） | SSD `ssd_total_bytes` |

透明模式下 KVCM 按 GA 逻辑大小统计 `pace` 用量，不随 DRAM/SSD 驻留变化重新归类。自动 quota 选择两层总容量的较大值作为保守预算，避免把两层当作完全独立的额度相加；这是一项预算策略，不等价于后端物理总量或某个 GA 此刻可分配的空闲量。`max` 在每个后端池分别汇总两层之后计算，不按节点取 `max` 再相加；例如两个节点分别为 DRAM/SSD 100/20 GiB、20/100 GiB，该池基数为 120 GiB。不同独立池仍按前述规则相加，然后应用 Group 的类型比例。比例还可预留余量；双份驻留、SSD slot 放大、碎片和后端水位门禁仍可能造成 alloc fail。

透明模式不支持显式 DRAM 或独立 `pace_ssd` 自动预算，避免同一 SSD 容量重复用于两个类型。透明模式明确报告 tiering 已关闭时只计 DRAM；开启时 SSD 数据缺失、为零或溢出，整个样本失败，不能发布较小的 DRAM 部分值。分层已开启但缺少模式时，容量未知。共享远端 tier 需要独立的 owner quota 口径，不能把每个 provider 的 SSD 字段相加，当前自动容量返回不支持。

TairMempool 复用原有探活/节点同步循环，每轮读取一次 memnode，分别更新节点集合、容量和物理 DRAM 使用率。`GetStorageCapacity()` 与原有 `GetStorageUsageRatio()` 都只读缓存；监控和 quota 读取不再各自请求 meta。容量字段无效时保留成功容量及其时间，仍可更新有效的节点信息和 DRAM 水位；原有后端水位的统计口径和错误返回语义保留。正常循环约 10 秒，失败探活的重试、查询超时会增加间隔，过期判定仍以最近成功时间为准。监控另有最多约 5 秒的 recorder 采样延迟。

| 状态 | 展示容量 | 新分配 | 按该类型容量水位回收/迁移 |
| --- | --- | --- | --- |
| `fixed` | 配置值 | 按配置检查 | 按配置检查 |
| `fresh` | 最近成功值 × 比例 | 按有效值检查 | 按有效值检查 |
| `refresh_error`，成功值未超过 30 秒 | 保留最近成功值 × 比例 | 按保留值检查，Writable 仍须通过 | 按保留值检查，后端原保护规则仍生效 |
| `stale`，超过 30 秒未成功刷新 | 保留最近成功值，usable=false | 暂停该类型新分配 | 跳过该类型的容量触发 |
| 首次未知、缺失、不支持或无效配置 | `-1`，known=false | 暂停该类型新分配 | 跳过该类型的容量触发 |
| 确认的零容量 | `0`，known=true | 拒绝新分配 | 有用量时按该类型回收 |

有效期用单调时钟判定，查询另提供最近成功的 Unix 秒时间。失败不覆盖容量和成功时间。多后端中任一个首次未知，则整个类型未知；已有成功值的失败后端保留其贡献。任一容量来源过期则整个类型过期。恢复成功后自动使用新容量。移除/替换后端或 Leader cleanup 清理对应快照，同名新后端不继承旧容量。关闭后端等待当前有界查询完成。重启/切主后的首次未知遵循同样策略，不猜测默认容量。

`unsupported` 表示当前介质无法可靠计量；即使此前有成功值，也只保留展示，立即停止使用该类型容量作分配/回收判断。相同池的另一个别名查询成功不能掩盖这一介质配置问题。

真实缩容后用量可以超过有效 quota：超配类型暂停分配并按自身水位回收。总量为 `-1` 时，Group 用量超过参考容量也不会阻止其他有余量类型的新分配。直接大比例缩容仍可能出现后端 alloc fail，缩容前排空编排不属于此功能。

## 消费与观测

`RegistryManager::ResolveQuota` 统一解析快照。普通写入和显式迁移目标使用 `ForAdmission()`；回收和迁移源水位使用 `ForReclaim()`。未知/过期容量在两条路径分别转换为拒绝分配、跳过容量回收，不修改展示值。沿用原有水位准入，不新增并发字节 reservation。

每次解析使用调用方持有的同一份 InstanceGroup 配置快照，比例和后端范围不会分别读取不同配置版本，避免更新时拼接“旧比例 + 新后端”而制造一次不存在的缩容。

Group 参考容量为各类型有效 quota 之和，包含固定与自动模式；任一类型未知或求和溢出时为未知 `-1`，不发布部分和。过期但已知的容量保留参考贡献，并标明陈旧性。DRAM 100 GiB × 50% 和 SSD 200 GiB × 80% 的参考容量为 210 GiB；用量 20、80 GiB 对应总使用率约 47.62%，分类型为 40%、50%。

`getInstanceGroup` / `listInstanceGroup` 的只读 `runtime_quota` 与配置分开：

- `quota.capacity=-1`、`total_bytes_limit_enabled=false` 表示“Group 总容量上限：无限”。
- `reference_capacity` 表示“汇总参考容量”，只供观测。
- `storage_types` 提供模式、比例、有效 quota、后端容量、known、usable、刷新状态及成功时间。

`cache_manager_group.usage_ratio` 在固定正数总 quota 下沿用原分母，总量 `-1` 时使用参考容量。未知使用率为 NaN；确认零容量且有用量为 +Inf，空用量为 0，通过容量和 known 区分无容量状态。后端原有 `data_storage.storage_usage_ratio` 含义不变。

`data_storage.total_capacity_bytes` 按 `type, unique_name` 上报每个后端的容量基数，使用同一内存快照，不新增采集请求。它未应用 Group 比例，也不汇总多个 Group；未知/不支持为 -1，失败或过期保留最近成功值，成功的零容量为 0。透明分层取两层总量的 max，主动分层按后端类型取对应介质。现有 TairMempool `storage_usage_ratio` 保留物理 DRAM 水位口径及原有过滤规则，新总量不保证是旧 ratio 的分母，不能据此相乘推算用量。同池别名各自上报，求物理总量时需避免重复相加。follower 省略该指标，降级和卸载沿用存储指标的清理机制。

CLI 新建默认总量 `-1`，支持 `--quota_configs 'ST_TAIRMEMPOOL,auto,0.5;ST_TAIRMEMPOOL_SSD,auto,0.8'`，显式范围格式为 `type,auto,ratio,backend1|backend2`。Bootstrap 新建采用相同默认；升级同步未显式配置 quota 时保留已有配置，显式 `quota_capacity=-1` 才改为默认自动模式。不支持容量查询的后端需要显式固定 quota。

启用自动模式或总量 `-1` 后，回滚到不支持此功能的旧二进制前，应先恢复正数固定总 quota 和固定分类型 quota；旧版本无法正确解释这些新配置语义。
